from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from _bootstrap import PROJECT_ROOT  # noqa: F401
from igra_forecast.revision2_data import (
    ensure_data_directories,
    load_data_config,
    month_keys,
    sha256_file,
    stable_fingerprint,
    write_json,
)


ALIASES = {
    "pressure": ["Pres", "pres", "pressure", "Pressure"],
    "refractivity": ["Ref", "ref", "refractivity", "Refractivity"],
    "height": ["MSL_alt", "msl_alt", "height", "Height", "altitude"],
    "latitude": ["Lat", "lat", "latitude", "Latitude"],
    "longitude": ["Lon", "lon", "longitude", "Longitude"],
}


def choose(dataset: xr.Dataset, logical: str) -> str:
    name = next((candidate for candidate in ALIASES[logical] if candidate in dataset.variables), None)
    if name is None:
        raise KeyError(f"Cannot locate COSMIC-2 {logical}; tried {ALIASES[logical]}")
    return name


def scalar(dataset: xr.Dataset, name: str) -> float:
    values = np.asarray(dataset[name].values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    return float(np.nanmedian(values)) if len(values) else np.nan


def profile_time(dataset: xr.Dataset) -> pd.Timestamp:
    attrs = dataset.attrs
    keys = ["year", "month", "day", "hour", "minute", "second"]
    if all(key in attrs for key in keys[:3]):
        values = [int(float(attrs.get(key, 0))) for key in keys]
        return pd.Timestamp(*values, tz="UTC").tz_convert(None)
    for name in ["time", "occultation_time", "datetime"]:
        if name in dataset.variables:
            return pd.to_datetime(np.asarray(dataset[name].values).reshape(-1)[0])
    return pd.NaT


def interpolate(
    pressure_hpa: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
    *,
    positive_values: bool = False,
) -> np.ndarray:
    valid = np.isfinite(pressure_hpa) & np.isfinite(values) & (pressure_hpa > 0)
    if positive_values:
        valid &= values > 0
    if valid.sum() < 2:
        return np.full(targets.shape, np.nan)
    x = np.log(pressure_hpa[valid])
    y = values[valid]
    order = np.argsort(x)
    x, unique = np.unique(x[order], return_index=True)
    return np.interp(np.log(targets), x, y[order][unique], left=np.nan, right=np.nan)


def interpolate_longitude(pressure_hpa: np.ndarray, longitude: np.ndarray, targets: np.ndarray) -> np.ndarray:
    valid = np.isfinite(pressure_hpa) & np.isfinite(longitude) & (pressure_hpa > 0)
    if valid.sum() < 2:
        return np.full(targets.shape, np.nan)
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(longitude[valid])))
    interpolated = interpolate(pressure_hpa[valid], unwrapped, targets)
    return (interpolated + 180.0) % 360.0 - 180.0


def fallback_longitude(longitude: np.ndarray) -> float:
    finite = np.asarray(longitude, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.nan
    value = np.rad2deg(np.median(np.unwrap(np.deg2rad(finite))))
    return float((value + 180.0) % 360.0 - 180.0)


def cosmic_profile_id(member_name: str) -> str:
    name = Path(member_name).name
    if name.lower().endswith("_nc"):
        return name[:-3]
    return Path(name).stem


def is_netcdf_member(member: tarfile.TarInfo) -> bool:
    """UCAR atmPrf archives use extension-less names ending in ``_nc``."""
    name = member.name.lower()
    return member.isfile() and name.endswith((".nc", ".nc4", "_nc"))


def open_cosmic_payload(payload: bytes) -> xr.Dataset:
    if payload[:3] == b"CDF":
        return xr.open_dataset(io.BytesIO(payload), engine="scipy")
    raise ValueError(
        "Unsupported COSMIC-2 NetCDF payload. Expected a classic CDF file; "
        f"magic={payload[:8].hex()}"
    )


def schema(dataset: xr.Dataset) -> dict[str, object]:
    return {
        "sizes": dict(dataset.sizes), "attributes": dict(dataset.attrs),
        "variables": {name: {"dims": list(value.dims), "dtype": str(value.dtype), "attrs": dict(value.attrs)}
                      for name, value in dataset.variables.items()},
    }


def main() -> None:
    default_output = Path("data/revision2/processed/cosmic2_atmprf_profiles.parquet")
    parser = argparse.ArgumentParser(description="Inspect or standardize downloaded COSMIC-2 atmPrf tar archives.")
    parser.add_argument("--config", default="configs/revision2/data_pipeline.yaml")
    parser.add_argument("--input-root", default="data/revision2/raw/cosmic2/nrt/atmPrf")
    parser.add_argument("--output", default=str(default_output))
    parser.add_argument("--schema-only", action="store_true")
    parser.add_argument("--max-profiles", type=int)
    parser.add_argument("--resume", action="store_true", help="Reuse completed per-archive Parquet caches.")
    args = parser.parse_args()
    if args.max_profiles is not None and Path(args.output) == default_output:
        args.output = "data/revision2/smoke/processed/cosmic2_atmprf_profiles.parquet"
    config = load_data_config(args.config)
    paths = ensure_data_directories(config)
    archives = sorted(Path(args.input_root).rglob("*.tar.gz"))
    if not archives:
        raise FileNotFoundError(f"No COSMIC-2 archives below {args.input_root}")
    expected_archive_count = len(month_keys(config["study"]["start"], config["study"]["end"]))
    if not args.schema_only and args.max_profiles is None and len(archives) != expected_archive_count:
        raise ValueError(
            f"Formal COSMIC-2 preparation requires one archive per study month; "
            f"expected={expected_archive_count}, found={len(archives)}"
        )
    targets = np.asarray(config["study"]["pressure_levels_hpa"], dtype=float)
    parser_fingerprint = stable_fingerprint({
        "parser_version": 2,
        "pressure_levels_hpa": targets.astype(int).tolist(),
        "min_valid_levels_per_profile": int(config["study"]["min_valid_levels_per_profile"]),
        "member_suffixes": [".nc", ".nc4", "_nc"],
        "horizontal_location": "pressure-level interpolated Lat/Lon",
    })
    download_manifest_path = paths["manifests"] / "cosmic2_download_manifest.json"
    download_hashes: dict[str, str] = {}
    if download_manifest_path.is_file():
        download_manifest = json.loads(download_manifest_path.read_text(encoding="utf-8"))
        download_hashes = {
            os.path.normcase(str(Path(record["path"]).resolve())): str(record["sha256"])
            for record in download_manifest.get("files", [])
        }
    cache_root = paths["interim"] / "cosmic2_atmprf_archive_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    schemas: list[dict[str, object]] = []
    archive_audit: list[dict[str, object]] = []
    processed = 0
    reused_profiles = 0
    stop = False
    for archive in tqdm(archives, desc="COSMIC-2 archives"):
        archive_key = os.path.normcase(str(archive.resolve()))
        archive_sha256 = download_hashes.get(archive_key)
        if archive_sha256 is None:
            archive_sha256 = sha256_file(archive)
        cache_path = cache_root / f"{archive.name}.{archive_sha256[:16]}.{parser_fingerprint[:12]}.parquet"
        if args.resume and not args.schema_only and args.max_profiles is None and cache_path.is_file():
            cached = pd.read_parquet(cache_path)
            frames.append(cached)
            reused_profiles += int(cached["profile_id"].nunique())
            archive_audit.append({
                "archive": str(archive.resolve()), "archive_sha256": archive_sha256,
                "cache": str(cache_path.resolve()), "cache_reused": True,
                "profiles_retained": int(cached["profile_id"].nunique()), "rows": int(len(cached)),
            })
            continue
        archive_rows: list[dict[str, object]] = []
        netcdf_members = 0
        examined_in_archive = 0
        # Streaming mode reads each multi-GB gzip archive only once.
        with tarfile.open(archive, "r|gz") as bundle:
            for member in tqdm(bundle, desc=archive.name, leave=False, unit="members"):
                if not is_netcdf_member(member):
                    continue
                netcdf_members += 1
                extracted = bundle.extractfile(member)
                if extracted is None:
                    continue
                payload = extracted.read()
                with open_cosmic_payload(payload) as dataset:
                    if len(schemas) < 3:
                        schemas.append({"archive": str(archive), "member": member.name, **schema(dataset)})
                    if args.schema_only:
                        processed += 1
                    else:
                        names = {logical: choose(dataset, logical) for logical in ALIASES}
                        pressure = np.asarray(dataset[names["pressure"]].values, dtype=float).reshape(-1)
                        unit = str(dataset[names["pressure"]].attrs.get("units", "")).lower()
                        if ("pa" in unit and "hpa" not in unit) or np.nanmedian(pressure) > 2000:
                            pressure = pressure / 100.0
                        ref = np.asarray(dataset[names["refractivity"]].values, dtype=float).reshape(-1)
                        height = np.asarray(dataset[names["height"]].values, dtype=float).reshape(-1)
                        height_unit = str(dataset[names["height"]].attrs.get("units", "")).lower()
                        if "km" in height_unit or np.nanmedian(np.abs(height)) < 200:
                            height = height * 1000.0
                        latitude_raw = np.asarray(dataset[names["latitude"]].values, dtype=float).reshape(-1)
                        longitude_raw = np.asarray(dataset[names["longitude"]].values, dtype=float).reshape(-1)
                        lengths = {"pressure": len(pressure), "refractivity": len(ref), "height": len(height),
                                   "latitude": len(latitude_raw), "longitude": len(longitude_raw)}
                        if len(set(lengths.values())) != 1:
                            raise ValueError(f"COSMIC-2 profile vectors have inconsistent lengths in {member.name}: {lengths}")
                        values = interpolate(pressure, ref, targets, positive_values=True)
                        heights = interpolate(pressure, height, targets)
                        latitudes = interpolate(pressure, latitude_raw, targets)
                        longitudes = interpolate_longitude(pressure, longitude_raw, targets)
                        mask = np.isfinite(values) & np.isfinite(heights) & np.isfinite(latitudes) & np.isfinite(longitudes)
                        if mask.sum() >= config["study"]["min_valid_levels_per_profile"]:
                            timestamp = profile_time(dataset)
                            if pd.isna(timestamp):
                                raise ValueError(f"COSMIC-2 profile has no usable timestamp: {member.name}")
                            latitude_fallback = float(np.nanmedian(latitude_raw))
                            longitude_fallback = fallback_longitude(longitude_raw)
                            profile_id = cosmic_profile_id(member.name)
                            for level_index, level in enumerate(targets.astype(int)):
                                archive_rows.append(
                                    {"profile_id": profile_id, "time": timestamp,
                                     "latitude": latitudes[level_index] if np.isfinite(latitudes[level_index]) else latitude_fallback,
                                     "longitude": longitudes[level_index] if np.isfinite(longitudes[level_index]) else longitude_fallback,
                                     "pressure_hpa": level, "level_mask": int(mask[level_index]),
                                     "observed_n": values[level_index] if mask[level_index] else np.nan,
                                     "height_m": heights[level_index] if mask[level_index] else np.nan,
                                     "source_archive": str(archive), "source_member": member.name, "source": "COSMIC2_atmPrf_nrt"}
                                )
                        processed += 1
                        examined_in_archive += 1
                if args.max_profiles and processed >= args.max_profiles:
                    stop = True
                    break
        if netcdf_members == 0:
            raise ValueError(
                f"No NetCDF members found in {archive}. UCAR atmPrf members are expected to end in _nc, .nc, or .nc4."
            )
        if not args.schema_only:
            archive_frame = pd.DataFrame(archive_rows)
            if args.max_profiles is None:
                if archive_frame.empty:
                    raise ValueError(f"No profiles met the six-level data contract in {archive}")
                temporary_cache = cache_path.with_suffix(cache_path.suffix + ".tmp")
                archive_frame.to_parquet(temporary_cache, index=False, compression="zstd")
                os.replace(temporary_cache, cache_path)
            if not archive_frame.empty:
                frames.append(archive_frame)
            archive_audit.append({
                "archive": str(archive.resolve()), "archive_sha256": archive_sha256,
                "cache": str(cache_path.resolve()) if args.max_profiles is None else None,
                "cache_reused": False, "netcdf_members": int(netcdf_members),
                "profiles_examined": int(examined_in_archive),
                "profiles_retained": int(archive_frame["profile_id"].nunique()) if len(archive_frame) else 0,
                "rows": int(len(archive_frame)),
            })
        if stop:
            break
    schema_path = paths["manifests"] / "cosmic2_atmprf_schema.json"
    write_json(schema_path, {"sample_profiles": schemas, "netcdf_member_suffixes": ["_nc", ".nc", ".nc4"]})
    if args.schema_only:
        if processed == 0 or not schemas:
            raise ValueError("COSMIC-2 schema inspection examined zero NetCDF profiles")
        print(f"Schema: {schema_path.resolve()}")
        return
    if not frames:
        raise ValueError("COSMIC-2 preparation retained zero profiles; no output was written")
    result = pd.concat(frames, ignore_index=True)
    required = {"profile_id", "time", "latitude", "longitude", "pressure_hpa", "level_mask", "observed_n", "height_m"}
    missing = required - set(result.columns)
    if missing or result.empty:
        raise ValueError(f"COSMIC-2 prepared table is incomplete; missing={sorted(missing)}, rows={len(result)}")
    profile_sizes = result.groupby(["profile_id", "time"], observed=True).size()
    profile_levels = result.groupby(["profile_id", "time"], observed=True)["pressure_hpa"].nunique()
    if not (profile_sizes == len(targets)).all() or not (profile_levels == len(targets)).all():
        raise ValueError("COSMIC-2 output does not contain exactly six unique pressure-level rows per profile")
    if result[["time", "latitude", "longitude"]].isna().any().any():
        raise ValueError("COSMIC-2 output contains missing time or horizontal coordinates")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    result.to_parquet(temporary_output, index=False, compression="zstd")
    os.replace(temporary_output, output)
    manifest_path = (
        paths["manifests"] / "cosmic2_prepare_manifest.json"
        if args.max_profiles is None
        else paths["root"] / "smoke" / "manifests" / "cosmic2_prepare_manifest.json"
    )
    write_json(manifest_path, {
        "input_root": str(Path(args.input_root).resolve()), "output": str(output.resolve()),
        "complete": True, "parser_fingerprint": parser_fingerprint,
        "archives_found": len(archives), "archives_completed": len(archive_audit),
        "profiles_examined_this_run": processed, "profiles_reused_from_cache": reused_profiles,
        "profiles_retained": int(result["profile_id"].nunique()),
        "rows": int(len(result)), "output_sha256": sha256_file(output),
        "time_min": str(pd.to_datetime(result["time"]).min()), "time_max": str(pd.to_datetime(result["time"]).max()),
        "pressure_levels_hpa": targets.astype(int).tolist(), "rows_per_profile": len(targets),
        "horizontal_location": "Lat/Lon interpolated independently to each target pressure level",
        "archive_audit": archive_audit,
        "interpretation": "Cross-platform comparison; ERA5 assimilation means this is not an assimilation-independent truth set.",
    })
    print(
        f"COSMIC-2 profiles: {output.resolve()} | profiles={result['profile_id'].nunique():,} "
        f"rows={len(result):,} archives={len(archive_audit)}"
    )


if __name__ == "__main__":
    main()
