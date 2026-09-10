from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from _bootstrap import PROJECT_ROOT  # noqa: F401
from collocate_era5_igra import TOKENS, find_one, standardize_da
from igra_forecast.revision2_data import (
    G0,
    ensure_data_directories,
    load_data_config,
    refractivity_components,
    require_era5_audit,
    resolve_era5_root,
    sha256_file,
    write_json,
)


def concatenate_time_boundaries(
    current: xr.DataArray,
    target_times: pd.Series,
    previous: xr.DataArray | None = None,
    following: xr.DataArray | None = None,
) -> tuple[xr.DataArray, dict[str, bool]]:
    """Add only the adjacent ERA5 timestamps needed for linear interpolation."""
    current_times = pd.DatetimeIndex(pd.to_datetime(current["time"].values))
    target = pd.DatetimeIndex(pd.to_datetime(target_times))
    pieces: list[xr.DataArray] = []
    used_previous = bool(target.min() < current_times.min())
    used_following = bool(target.max() > current_times.max())
    if used_previous:
        if previous is None:
            raise ValueError(
                f"External profile at {target.min()} precedes ERA5 support {current_times.min()} and the previous month is unavailable"
            )
        boundary = previous.isel(time=-1).load()
        boundary_time = pd.Timestamp(boundary["time"].values)
        if target.min() < boundary_time:
            raise ValueError(f"Previous ERA5 boundary {boundary_time} does not bracket target {target.min()}")
        pieces.append(boundary)
    pieces.append(current)
    if used_following:
        if following is None:
            raise ValueError(
                f"External profile at {target.max()} exceeds ERA5 support {current_times.max()} and the next month is unavailable"
            )
        boundary = following.isel(time=0).load()
        boundary_time = pd.Timestamp(boundary["time"].values)
        if target.max() > boundary_time:
            raise ValueError(f"Following ERA5 boundary {boundary_time} does not bracket target {target.max()}")
        pieces.append(boundary)
    combined = current if len(pieces) == 1 else xr.concat(pieces, dim="time").sortby("time")
    return combined, {"previous_month_timestamp_used": used_previous, "next_month_timestamp_used": used_following}


def open_era5_variable(
    root: Path,
    period: pd.Period,
    config: dict[str, object],
    logical: str,
) -> xr.DataArray:
    year, month = f"{period.year:04d}", f"{period.month:02d}"
    yyyymm = f"{period.year:04d}{period.month:02d}"
    if logical == "surface_pressure":
        folder = root / year / month / "single_levels"
        return standardize_da(find_one(folder, f"*{yyyymm}*.nc"), "sp")
    folder = root / year / month / "pressure_levels"
    return standardize_da(
        find_one(folder, f"*{TOKENS[logical]}*{yyyymm}*.nc"),
        config["era5"]["required_pressure_variables"][logical],
    )


def validate_external_input(frame: pd.DataFrame) -> None:
    required = {"profile_id", "time", "latitude", "longitude", "pressure_hpa", "level_mask", "observed_n"}
    missing = required - set(frame.columns)
    if frame.empty or missing:
        raise ValueError(
            "External profile input is empty or incomplete. Run the formal preparation step (not --schema-only) first; "
            f"rows={len(frame)}, missing_columns={sorted(missing)}"
        )


def _indices(da: xr.DataArray, frame: pd.DataFrame, has_level: bool) -> tuple[dict[str, np.ndarray], np.ndarray]:
    times = pd.to_datetime(da["time"].values).to_numpy(dtype="datetime64[ns]").astype("int64")
    target_time = pd.to_datetime(frame["time"]).to_numpy(dtype="datetime64[ns]").astype("int64")
    hi = np.searchsorted(times, target_time, side="left")
    hi = np.clip(hi, 1, len(times) - 1)
    lo = hi - 1
    time_weight = (target_time - times[lo]) / (times[hi] - times[lo])
    if np.any((target_time < times[0]) | (target_time > times[-1])):
        raise ValueError("External profile time falls outside its monthly ERA5 file; boundary-month support is required")
    lat_values = np.asarray(da["latitude"].values, dtype=float)
    lon_values = np.asarray(da["longitude"].values, dtype=float)
    lat_step = abs(lat_values[1] - lat_values[0])
    target_lat = frame["latitude"].to_numpy(float)
    lat_pos = (lat_values[0] - target_lat) / lat_step if lat_values[0] > lat_values[-1] else (target_lat - lat_values[0]) / lat_step
    lat_pos = np.clip(lat_pos, 0, len(lat_values) - 1)
    lat0 = np.floor(lat_pos).astype(int)
    lat1 = np.minimum(lat0 + 1, len(lat_values) - 1)
    lon_step = abs(lon_values[1] - lon_values[0])
    lon_pos = np.mod((np.mod(frame["longitude"].to_numpy(float), 360) - lon_values[0]) / lon_step, len(lon_values))
    lon0 = np.floor(lon_pos).astype(int) % len(lon_values)
    lon1 = (lon0 + 1) % len(lon_values)
    result = {"lo": lo, "hi": hi, "lat0": lat0, "lat1": lat1, "lon0": lon0, "lon1": lon1,
              "wy": lat_pos - lat0, "wx": lon_pos - np.floor(lon_pos)}
    if has_level:
        lookup = {int(value): index for index, value in enumerate(np.asarray(da["level"].values))}
        result["level"] = np.asarray([lookup[int(value)] for value in frame["pressure_hpa"]], dtype=int)
    return result, time_weight


def preload_interpolation_window(
    da: xr.DataArray,
    frame: pd.DataFrame,
    has_level: bool,
    max_values: int = 50_000_000,
) -> tuple[xr.DataArray, dict[str, object]]:
    """Load the smallest rectangular ERA5 window needed by one target chunk.

    Vectorized point indexing against a netCDF4-backed array can degrade into
    thousands of small random reads.  RAPSODI is spatially compact, while the
    pre-registered COSMIC-2 sample is temporally compact.  In both cases the
    required time/space/level bounding window is normally small enough to load
    once and perform all interpolation from NumPy memory.  Very large windows
    retain the original point-index path to keep memory bounded.
    """
    index, _ = _indices(da, frame, has_level)
    time_indices = np.concatenate([index["lo"], index["hi"]])
    latitude_indices = np.concatenate([index["lat0"], index["lat1"]])
    longitude_indices = np.concatenate([index["lon0"], index["lon1"]])
    time_start, time_stop = int(time_indices.min()), int(time_indices.max()) + 1
    latitude_start, latitude_stop = int(latitude_indices.min()), int(latitude_indices.max()) + 1
    longitude_start, longitude_stop = int(longitude_indices.min()), int(longitude_indices.max()) + 1
    if latitude_stop - latitude_start == 1:
        if latitude_start > 0:
            latitude_start -= 1
        else:
            latitude_stop = min(da.sizes["latitude"], latitude_stop + 1)
    if longitude_stop - longitude_start == 1:
        if longitude_start > 0:
            longitude_start -= 1
        else:
            longitude_stop = min(da.sizes["longitude"], longitude_stop + 1)
    selections: dict[str, object] = {
        "time": slice(time_start, time_stop),
        "latitude": slice(latitude_start, latitude_stop),
        "longitude": slice(longitude_start, longitude_stop),
    }
    selected_sizes = {
        "time": time_stop - time_start,
        "latitude": latitude_stop - latitude_start,
        "longitude": longitude_stop - longitude_start,
    }
    if has_level:
        level_indices = np.unique(index["level"])
        selections["level"] = level_indices
        selected_sizes["level"] = int(len(level_indices))
    estimated_values = int(np.prod(list(selected_sizes.values()), dtype=np.int64))
    audit: dict[str, object] = {
        "preloaded": estimated_values <= int(max_values),
        "estimated_values": estimated_values,
        "selected_sizes": selected_sizes,
        "max_values": int(max_values),
    }
    if not audit["preloaded"]:
        return da, audit
    return da.isel(selections).load(), audit


def _gather(da: xr.DataArray, index: dict[str, np.ndarray], time_key: str, lat_key: str, lon_key: str) -> np.ndarray:
    selectors = {
        "time": xr.DataArray(index[time_key], dims="points"),
        "latitude": xr.DataArray(index[lat_key], dims="points"),
        "longitude": xr.DataArray(index[lon_key], dims="points"),
    }
    if "level" in index:
        selectors["level"] = xr.DataArray(index["level"], dims="points")
    return np.asarray(da.isel(selectors).values, dtype=float)


def spatiotemporal_points(da: xr.DataArray, frame: pd.DataFrame, has_level: bool) -> np.ndarray:
    index, wt = _indices(da, frame, has_level)
    results = []
    for time_key in ["lo", "hi"]:
        v00 = _gather(da, index, time_key, "lat0", "lon0")
        v01 = _gather(da, index, time_key, "lat0", "lon1")
        v10 = _gather(da, index, time_key, "lat1", "lon0")
        v11 = _gather(da, index, time_key, "lat1", "lon1")
        spatial = (1 - index["wy"]) * ((1 - index["wx"]) * v00 + index["wx"] * v01)
        spatial += index["wy"] * ((1 - index["wx"]) * v10 + index["wx"] * v11)
        results.append(spatial)
    return (1 - wt) * results[0] + wt * results[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collocate standardized RAPSODI or COSMIC-2 profiles with 6-hourly ERA5.")
    parser.add_argument("--config", default="configs/revision2/data_pipeline.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--source-name", required=True, choices=["rapsodi", "cosmic2"])
    parser.add_argument("--era5-root", help="Optional override for era5.root in the data config.")
    parser.add_argument("--era5-audit", help="Optional ERA5 audit manifest; defaults to the revision-2 manifests directory.")
    parser.add_argument("--prepare-manifest", help="Optional COSMIC-2 preparation manifest override (primarily for smoke tests).")
    parser.add_argument("--output-manifest", help="Optional collocation manifest override (primarily for smoke tests).")
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-rows", type=int, default=20000)
    args = parser.parse_args()
    config = load_data_config(args.config)
    paths = ensure_data_directories(config)
    input_path = Path(args.input)
    frame = pd.read_parquet(input_path)
    validate_external_input(frame)
    if args.source_name == "cosmic2":
        prepare_manifest_path = (
            Path(args.prepare_manifest)
            if args.prepare_manifest
            else paths["manifests"] / "cosmic2_prepare_manifest.json"
        )
        if not prepare_manifest_path.is_file():
            raise FileNotFoundError(f"COSMIC-2 formal preparation manifest is missing: {prepare_manifest_path}")
        prepare_manifest = json.loads(prepare_manifest_path.read_text(encoding="utf-8"))
        if not prepare_manifest.get("complete"):
            raise ValueError("COSMIC-2 formal preparation manifest is incomplete")
        if Path(prepare_manifest.get("output", "")).resolve() != input_path.resolve():
            raise ValueError("COSMIC-2 input does not match the formal preparation manifest output")
        if prepare_manifest.get("output_sha256") != sha256_file(input_path):
            raise ValueError("COSMIC-2 prepared Parquet hash does not match its formal manifest")
    frame["time"] = pd.to_datetime(frame["time"])
    frame = frame.loc[frame["time"].notna()].copy()
    frame["yyyymm"] = frame["time"].dt.strftime("%Y%m")
    outputs, audit = [], []
    root = resolve_era5_root(config, args.era5_root)
    audit_manifest_path = Path(args.era5_audit) if args.era5_audit else paths["manifests"] / "era5_archive_manifest.json"
    era5_audit = require_era5_audit(
        config,
        audit_manifest_path,
        root,
        sorted(frame["yyyymm"].unique().tolist()),
        require_full_study=True,
    )
    for yyyymm, month_frame in tqdm(frame.groupby("yyyymm"), desc=f"Collocate {args.source_name}"):
        arrays = {}
        source_handles: list[xr.DataArray] = []
        boundary_audit: dict[str, dict[str, object]] = {}
        try:
            period = pd.Period(yyyymm, freq="M")
            for logical in [*TOKENS, "surface_pressure"]:
                current = open_era5_variable(root, period, config, logical)
                source_handles.append(current)
                current_times = pd.DatetimeIndex(pd.to_datetime(current["time"].values))
                target_times = pd.DatetimeIndex(pd.to_datetime(month_frame["time"]))
                previous = following = None
                if target_times.min() < current_times.min():
                    previous = open_era5_variable(root, period - 1, config, logical)
                    source_handles.append(previous)
                if target_times.max() > current_times.max():
                    following = open_era5_variable(root, period + 1, config, logical)
                    source_handles.append(following)
                arrays[logical], boundary_audit[logical] = concatenate_time_boundaries(
                    current, month_frame["time"], previous=previous, following=following
                )
                boundary_audit[logical]["preload_chunks"] = 0
                boundary_audit[logical]["direct_index_chunks"] = 0
                boundary_audit[logical]["maximum_window_values"] = 0
            month_parts = []
            for start in range(0, len(month_frame), args.chunk_rows):
                part = month_frame.iloc[start : start + args.chunk_rows].copy()
                chunk_arrays: dict[str, xr.DataArray] = {}
                owned_chunk_arrays: list[xr.DataArray] = []
                try:
                    for logical, has_level in [
                        ("temperature", True), ("specific_humidity", True),
                        ("geopotential", True), ("surface_pressure", False),
                    ]:
                        chunk_arrays[logical], preload_audit = preload_interpolation_window(
                            arrays[logical], part, has_level
                        )
                        if preload_audit["preloaded"]:
                            owned_chunk_arrays.append(chunk_arrays[logical])
                        counter = "preload_chunks" if preload_audit["preloaded"] else "direct_index_chunks"
                        boundary_audit[logical][counter] = int(boundary_audit[logical][counter]) + 1
                        boundary_audit[logical]["maximum_window_values"] = max(
                            int(boundary_audit[logical]["maximum_window_values"]),
                            int(preload_audit["estimated_values"]),
                        )
                    part["era5_temperature_k"] = spatiotemporal_points(chunk_arrays["temperature"], part, True)
                    part["era5_specific_humidity"] = np.maximum(
                        spatiotemporal_points(chunk_arrays["specific_humidity"], part, True), 0
                    )
                    part["era5_height_m"] = spatiotemporal_points(chunk_arrays["geopotential"], part, True) / G0
                    part["era5_surface_pressure_pa"] = spatiotemporal_points(
                        chunk_arrays["surface_pressure"], part, False
                    )
                finally:
                    for chunk_da in owned_chunk_arrays:
                        chunk_da.close()
                dry, wet, total, _ = refractivity_components(
                    part["pressure_hpa"].to_numpy(), part["era5_temperature_k"].to_numpy(), part["era5_specific_humidity"].to_numpy()
                )
                part["era5_n_dry"], part["era5_n_wet"], part["era5_n"] = dry, wet, total
                part["below_ground_level"] = part["era5_surface_pressure_pa"] < part["pressure_hpa"] * 100
                part["residual_total"] = part["observed_n"] - part["era5_n"]
                if {"observed_n_dry", "observed_n_wet"}.issubset(part.columns):
                    part["residual_dry"] = part["observed_n_dry"] - part["era5_n_dry"]
                    part["residual_wet"] = part["observed_n_wet"] - part["era5_n_wet"]
                month_parts.append(part)
            result = pd.concat(month_parts, ignore_index=True)
            outputs.append(result)
            audit.append({
                "yyyymm": yyyymm, "rows": len(result), "profiles": result["profile_id"].nunique(),
                "target_time_min": str(pd.to_datetime(month_frame["time"]).min()),
                "target_time_max": str(pd.to_datetime(month_frame["time"]).max()),
                "boundary_time_support": boundary_audit,
            })
        finally:
            for da in arrays.values():
                da.close()
            for da in source_handles:
                da.close()
    result = pd.concat(outputs, ignore_index=True)
    result = result.drop(columns=["yyyymm"])
    expected_levels = set(map(int, config["study"]["pressure_levels_hpa"]))
    profile_keys = ["profile_id", "time"]
    profile_sizes = result.groupby(profile_keys, observed=True).size()
    profile_level_counts = result.groupby(profile_keys, observed=True)["pressure_hpa"].nunique()
    if not (profile_sizes == len(expected_levels)).all() or not (profile_level_counts == len(expected_levels)).all():
        raise ValueError("External collocation output does not contain exactly one row per configured pressure level and profile")
    if set(map(int, result["pressure_hpa"].unique())) != expected_levels:
        raise ValueError("External collocation pressure levels do not match the configured six-level contract")
    evaluation_mask = result["level_mask"].astype(bool) & ~result["below_ground_level"].astype(bool)
    core_columns = [
        "observed_n", "era5_temperature_k", "era5_specific_humidity", "era5_height_m",
        "era5_surface_pressure_pa", "era5_n_dry", "era5_n_wet", "era5_n", "residual_total",
    ]
    nonfinite_core = {
        column: int((~np.isfinite(result.loc[evaluation_mask, column].to_numpy(float))).sum())
        for column in core_columns
    }
    if any(nonfinite_core.values()):
        raise ValueError(f"External collocation produced non-finite core values on evaluable rows: {nonfinite_core}")
    residual_identity_max = None
    if {"residual_dry", "residual_wet"}.issubset(result.columns):
        identity = result.loc[evaluation_mask, "residual_total"].to_numpy(float)
        identity -= (
            result.loc[evaluation_mask, "residual_dry"].to_numpy(float)
            + result.loc[evaluation_mask, "residual_wet"].to_numpy(float)
        )
        residual_identity_max = float(np.max(np.abs(identity))) if len(identity) else 0.0
        if residual_identity_max > 1e-9:
            raise ValueError(f"External dry/wet/total residual identity failed: {residual_identity_max}")
    output_audit = {
        "rows": int(len(result)),
        "profiles": int(result[profile_keys].drop_duplicates().shape[0]),
        "evaluable_rows": int(evaluation_mask.sum()),
        "below_ground_rows": int(result["below_ground_level"].astype(bool).sum()),
        "pressure_levels_hpa": sorted(expected_levels, reverse=True),
        "rows_per_profile": len(expected_levels),
        "nonfinite_core_on_evaluable_rows": nonfinite_core,
        "dry_wet_total_residual_identity_max_abs": residual_identity_max,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False, compression="zstd")
    output_manifest_path = Path(args.output_manifest) if args.output_manifest else paths["manifests"] / f"{args.source_name}_era5_collocation_manifest.json"
    write_json(output_manifest_path, {
        "source_name": args.source_name,
        "input": str(Path(args.input).resolve()), "era5_root": str(root.resolve()), "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "era5_audit_manifest": str(audit_manifest_path.resolve()),
        "era5_archive_fingerprint": era5_audit["archive_fingerprint"],
        "interpolation": "linear in time and bilinear in latitude-longitude; exact pressure levels",
        "output_audit": output_audit,
        "monthly_audit": audit,
    })


if __name__ == "__main__":
    main()
