from __future__ import annotations

import argparse
import json
import os
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from _bootstrap import PROJECT_ROOT  # noqa: F401
from prepare_cosmic2 import cosmic_profile_id, is_netcdf_member, open_cosmic_payload, profile_time
from igra_forecast.revision2_data import ensure_data_directories, load_data_config, sha256_file, stable_fingerprint, write_json
from igra_forecast.revision2_publication import (
    cosmic_official_qc_passes,
    interpolate_longitude_no_extrapolation,
    interpolate_no_extrapolation,
    validate_geometric_height_schema,
)


def _integer_attribute(value: object) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _read_publication_config(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _profile_rows(dataset, member_name: str, targets_m: np.ndarray) -> tuple[list[dict[str, object]], str]:
    qc_passes, qc_reason = cosmic_official_qc_passes(dict(dataset.attrs))
    if not qc_passes:
        return [], qc_reason
    errstr = str(dataset.attrs.get("errstr", "")).strip().strip("\x00")
    required = {"MSL_alt", "Ref", "Lat", "Lon"}
    if not required.issubset(dataset.variables):
        return [], "missing_required_variable"
    height = np.asarray(dataset["MSL_alt"].values, dtype=float).reshape(-1)
    unit = str(dataset["MSL_alt"].attrs.get("units", "")).lower()
    if "km" in unit or (np.isfinite(height).any() and np.nanmedian(np.abs(height)) < 200):
        height = height * 1000.0
    refractivity = np.asarray(dataset["Ref"].values, dtype=float).reshape(-1)
    latitude = np.asarray(dataset["Lat"].values, dtype=float).reshape(-1)
    longitude = np.asarray(dataset["Lon"].values, dtype=float).reshape(-1)
    if len({len(height), len(refractivity), len(latitude), len(longitude)}) != 1:
        return [], "inconsistent_vector_lengths"
    source_valid = (
        np.isfinite(height) & np.isfinite(refractivity) & np.isfinite(latitude) & np.isfinite(longitude)
        & (height >= 0.0) & (refractivity > 0.0)
        & (latitude >= -90.0) & (latitude <= 90.0)
        & (longitude >= -180.0) & (longitude <= 180.0)
    )
    if source_valid.sum() < 2:
        return [], "insufficient_native_levels"
    height = height[source_valid]
    refractivity = refractivity[source_valid]
    latitude = latitude[source_valid]
    longitude = longitude[source_valid]
    observed = interpolate_no_extrapolation(height, refractivity, targets_m)
    target_latitude = interpolate_no_extrapolation(height, latitude, targets_m)
    target_longitude = interpolate_longitude_no_extrapolation(height, longitude, targets_m)
    height_mask = np.isfinite(observed) & np.isfinite(target_latitude) & np.isfinite(target_longitude)
    timestamp = profile_time(dataset)
    if pd.isna(timestamp):
        return [], "missing_timestamp"
    fallback_latitude = float(np.nanmedian(latitude))
    fallback_longitude = float(np.nanmedian(longitude))
    rows: list[dict[str, object]] = []
    profile_id = cosmic_profile_id(member_name)
    for index, target in enumerate(targets_m):
        valid = bool(height_mask[index])
        rows.append({
            "profile_id": profile_id,
            "time": timestamp,
            "latitude": float(target_latitude[index]) if valid else fallback_latitude,
            "longitude": float(target_longitude[index]) if valid else fallback_longitude,
            "height_m": float(target),
            "height_mask": int(valid),
            "observed_n": float(observed[index]) if valid else np.nan,
            "source_member": member_name,
            "cosmic_bad": 0,
            "cosmic_errstr": errstr,
            "native_height_min_m": float(np.min(height)),
            "native_height_max_m": float(np.max(height)),
            "source": "COSMIC2_atmPrf_geometric_height",
        })
    return rows, "retained"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare QC-passing COSMIC-2 atmPrf refractivity on fixed MSL heights.")
    parser.add_argument("--config", default="configs/revision2/publication.yaml")
    parser.add_argument("--input-root")
    parser.add_argument("--download-manifest", help="Use exactly the archives listed by an audited download manifest.")
    parser.add_argument("--output")
    parser.add_argument("--manifest", help="Optional isolated preparation-manifest path.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-profiles", type=int)
    args = parser.parse_args()

    publication = _read_publication_config(args.config)
    settings = publication["cosmic2_height"]
    data_config = load_data_config(publication["data_config"])
    data_paths = ensure_data_directories(data_config)
    input_root = Path(args.input_root or settings["input_root"])
    formal_output = Path(settings["prepared_profiles"])
    output = Path(args.output or formal_output)
    smoke = args.max_profiles is not None
    if smoke and args.output is None:
        output = data_paths["root"] / "smoke" / "processed" / "cosmic2_atmprf_height_profiles.parquet"
    targets_m = np.asarray(settings["height_levels_km"], dtype=float) * 1000.0
    minimum_valid = int(settings["minimum_valid_heights"])
    download_manifest = None
    if args.download_manifest:
        download_manifest_path = Path(args.download_manifest)
        download_manifest = json.loads(download_manifest_path.read_text(encoding="utf-8"))
        archives = [Path(item["path"]) for item in download_manifest.get("files", [])]
        if not archives or any(not path.is_file() for path in archives):
            raise FileNotFoundError("COSMIC download manifest contains missing archive paths")
        expected = int(download_manifest.get("expected_file_count", len(archives)))
        if len(archives) != expected or len({path.resolve() for path in archives}) != len(archives):
            raise ValueError("COSMIC download manifest archive count or uniqueness check failed")
        for item, path in zip(download_manifest["files"], archives):
            if item.get("sha256") and sha256_file(path) != item["sha256"]:
                raise ValueError(f"COSMIC archive hash mismatch: {path}")
    else:
        archives = sorted(input_root.rglob("*.tar.gz"))
    if not archives:
        raise FileNotFoundError(f"No COSMIC-2 tar.gz archives below {input_root}")
    expected_archives = int(
        download_manifest.get("expected_file_count", len(archives)) if download_manifest
        else 24
    )
    if not smoke and len(archives) != expected_archives:
        raise ValueError(f"Formal geometric-height preparation requires {expected_archives} archives, found {len(archives)}")

    # The profile parser and QC rules are unchanged from the one-archive-day
    # preparation.  Keep the parser fingerprint stable so a conditional
    # three-day expansion can reuse the already audited day-15 archive cache.
    # The exact archive selection remains bound separately by the download
    # manifest and its file hashes.
    fingerprint = stable_fingerprint({
        "revision": 1,
        "coordinate": "MSL_geometric_height",
        "height_levels_m": targets_m.tolist(),
        "qc": "official bad == 0",
        "minimum_valid_heights": minimum_valid,
        "dry_pressure_used": False,
    })
    cache_root = data_paths["interim"] / "cosmic2_height_archive_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    archive_audit: list[dict[str, object]] = []
    total_examined = 0
    total_reused = 0
    stop = False

    for archive in tqdm(archives, desc="COSMIC-2 height archives"):
        archive_hash = sha256_file(archive)
        cache = cache_root / f"{archive.name}.{archive_hash[:16]}.{fingerprint[:12]}.parquet"
        audit_cache = cache.with_suffix(".json")
        if args.resume and not smoke and cache.is_file() and audit_cache.is_file():
            frame = pd.read_parquet(cache)
            audit = json.loads(audit_cache.read_text(encoding="utf-8"))
            frames.append(frame)
            archive_audit.append(audit)
            total_reused += int(audit.get("profiles_examined", 0))
            continue
        rows: list[dict[str, object]] = []
        reasons: dict[str, int] = {}
        examined = 0
        retained = 0
        with tarfile.open(archive, "r|gz") as bundle:
            for member in tqdm(bundle, desc=archive.name, leave=False, unit="members"):
                if not is_netcdf_member(member):
                    continue
                extracted = bundle.extractfile(member)
                if extracted is None:
                    continue
                payload = extracted.read()
                with open_cosmic_payload(payload) as dataset:
                    profile_rows, reason = _profile_rows(dataset, member.name, targets_m)
                examined += 1
                total_examined += 1
                reasons[reason] = reasons.get(reason, 0) + 1
                if profile_rows and sum(int(row["height_mask"]) for row in profile_rows) >= minimum_valid:
                    for row in profile_rows:
                        row["source_archive"] = str(archive.resolve())
                    rows.extend(profile_rows)
                    retained += 1
                elif profile_rows:
                    reasons["insufficient_fixed_heights"] = reasons.get("insufficient_fixed_heights", 0) + 1
                    reasons[reason] -= 1
                if args.max_profiles is not None and total_examined >= int(args.max_profiles):
                    stop = True
                    break
        frame = pd.DataFrame(rows)
        audit = {
            "archive": str(archive.resolve()),
            "archive_sha256": archive_hash,
            "cache": str(cache.resolve()) if not smoke else None,
            "cache_reused": False,
            "profiles_examined": examined,
            "profiles_retained": retained,
            "rows": int(len(frame)),
            "exclusion_reasons": reasons,
        }
        if not frame.empty:
            frames.append(frame)
        archive_audit.append(audit)
        if not smoke:
            temporary = cache.with_suffix(cache.suffix + ".tmp")
            frame.to_parquet(temporary, index=False, compression="zstd")
            os.replace(temporary, cache)
            write_json(audit_cache, audit)
        if stop:
            break

    if not frames:
        raise ValueError("COSMIC-2 geometric-height preparation retained zero profiles")
    result = pd.concat(frames, ignore_index=True)
    validate_geometric_height_schema(result)
    expected_rows = len(targets_m)
    sizes = result.groupby(["profile_id", "time"], observed=True).size()
    if not (sizes == expected_rows).all():
        raise ValueError("Each retained COSMIC-2 height profile must contain every fixed-height row")
    if not (result.groupby(["profile_id", "time"], observed=True)["height_mask"].sum() >= minimum_valid).all():
        raise ValueError("A retained COSMIC-2 profile violates the minimum fixed-height requirement")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    result.to_parquet(temporary_output, index=False, compression="zstd")
    os.replace(temporary_output, output)
    manifest_path = (
        Path(args.manifest or settings["preparation_manifest"])
        if not smoke
        else data_paths["root"] / "smoke" / "manifests" / "cosmic2_height_prepare_manifest.json"
    )
    write_json(manifest_path, {
        "complete": True,
        "smoke": smoke,
        "input_root": str(input_root.resolve()),
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "parser_fingerprint": fingerprint,
        "coordinate": "MSL_geometric_height",
        "dry_pressure_used": False,
        "official_qc_rule": "bad == 0",
        "height_levels_m": targets_m.tolist(),
        "profiles_examined_this_run": total_examined,
        "profiles_reused_from_cache": total_reused,
        "profiles_retained": int(result[["profile_id", "time"]].drop_duplicates().shape[0]),
        "rows": int(len(result)),
        "archives_found": len(archives),
        "archives_processed": len(archive_audit),
        "archive_audit": archive_audit,
        "download_manifest": str(Path(args.download_manifest).resolve()) if args.download_manifest else None,
        "download_manifest_sha256": sha256_file(args.download_manifest) if args.download_manifest else None,
        "interpretation": "Primary geometric-height cross-platform input; dry pressure is not used as actual pressure.",
    })
    print(f"COSMIC-2 height profiles: {output.resolve()} | profiles={len(sizes):,} rows={len(result):,}")
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
