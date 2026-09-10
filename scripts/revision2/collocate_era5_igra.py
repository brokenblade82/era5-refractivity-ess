from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import xarray as xr
from tqdm import tqdm

from _bootstrap import PROJECT_ROOT  # noqa: F401
from igra_forecast.revision2_data import (
    G0,
    data_namespace,
    ensure_data_directories,
    load_data_config,
    month_keys,
    refractivity_components,
    require_era5_audit,
    require_igra_profile_manifest,
    resolve_era5_root,
    revision2_config_fingerprint,
    write_json,
)


TOKENS = {"temperature": "temperature", "specific_humidity": "specific_humidity", "geopotential": "geopotential"}


def find_one(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one file for {folder / pattern}; found {len(matches)}")
    return matches[0]


def standardize_da(path: Path, variable: str) -> xr.DataArray:
    dataset = xr.open_dataset(path)
    if variable not in dataset:
        dataset.close()
        raise KeyError(f"{variable!r} is missing in {path}")
    da = dataset[variable]
    rename = {}
    for candidates, target in [
        (("valid_time", "time"), "time"), (("pressure_level", "level", "isobaricInhPa"), "level"),
        (("latitude", "lat"), "latitude"), (("longitude", "lon"), "longitude"),
    ]:
        found = next((name for name in candidates if name in da.dims or name in da.coords), None)
        if found and found != target:
            rename[found] = target
    return da.rename(rename)


def _point_indices(da: xr.DataArray, frame: pd.DataFrame, has_level: bool) -> dict[str, np.ndarray]:
    times = pd.to_datetime(da["time"].values).to_numpy(dtype="datetime64[ns]").astype("int64")
    time_lookup = {int(value): index for index, value in enumerate(times)}
    target_time = pd.to_datetime(frame["time"]).to_numpy(dtype="datetime64[ns]").astype("int64")
    try:
        time_index = np.asarray([time_lookup[value] for value in target_time], dtype=np.int64)
    except KeyError as exc:
        raise KeyError(f"IGRA time is absent from ERA5 monthly file: {pd.Timestamp(exc.args[0])}") from exc
    lat_values = np.asarray(da["latitude"].values, dtype=float)
    lon_values = np.asarray(da["longitude"].values, dtype=float)
    if len(lat_values) < 2 or len(lon_values) < 2:
        raise ValueError("ERA5 grid must contain at least two latitudes and longitudes")
    lat_descending = lat_values[0] > lat_values[-1]
    lat_step = float(abs(lat_values[1] - lat_values[0]))
    target_lat = frame["latitude"].to_numpy(float)
    raw_lat_pos = (lat_values[0] - target_lat) / lat_step if lat_descending else (target_lat - lat_values[0]) / lat_step
    raw_lat_pos = np.clip(raw_lat_pos, 0.0, len(lat_values) - 1.0)
    lat0 = np.floor(raw_lat_pos).astype(np.int64)
    lat1 = np.minimum(lat0 + 1, len(lat_values) - 1)
    lat_weight = raw_lat_pos - lat0
    lon_step = float(abs(lon_values[1] - lon_values[0]))
    target_lon = np.mod(frame["longitude"].to_numpy(float), 360.0)
    lon_pos = np.mod((target_lon - lon_values[0]) / lon_step, len(lon_values))
    lon0 = np.floor(lon_pos).astype(np.int64) % len(lon_values)
    lon1 = (lon0 + 1) % len(lon_values)
    lon_weight = lon_pos - np.floor(lon_pos)
    result = {
        "time": time_index, "lat0": lat0, "lat1": lat1, "lat_weight": lat_weight,
        "lon0": lon0, "lon1": lon1, "lon_weight": lon_weight,
    }
    if has_level:
        level_values = np.asarray(da["level"].values).astype(int)
        lookup = {value: index for index, value in enumerate(level_values)}
        result["level"] = np.asarray([lookup[int(value)] for value in frame["pressure_hpa"]], dtype=np.int64)
    return result


def _gather(da: xr.DataArray, indices: dict[str, np.ndarray], lat_key: str, lon_key: str) -> np.ndarray:
    point = "points"
    indexers = {
        "time": xr.DataArray(indices["time"], dims=point),
        "latitude": xr.DataArray(indices[lat_key], dims=point),
        "longitude": xr.DataArray(indices[lon_key], dims=point),
    }
    if "level" in indices:
        indexers["level"] = xr.DataArray(indices["level"], dims=point)
    return np.asarray(da.isel(indexers).values, dtype=np.float64)


def bilinear_points(da: xr.DataArray, frame: pd.DataFrame, has_level: bool) -> np.ndarray:
    idx = _point_indices(da, frame, has_level)
    v00 = _gather(da, idx, "lat0", "lon0")
    v01 = _gather(da, idx, "lat0", "lon1")
    v10 = _gather(da, idx, "lat1", "lon0")
    v11 = _gather(da, idx, "lat1", "lon1")
    wx = idx["lon_weight"]
    wy = idx["lat_weight"]
    return (1 - wy) * ((1 - wx) * v00 + wx * v01) + wy * ((1 - wx) * v10 + wx * v11)


def bilinear_loaded_array(
    values: np.ndarray,
    indices: dict[str, np.ndarray],
    *,
    has_level: bool,
    chunk_rows: int,
) -> np.ndarray:
    """Sample an in-memory monthly grid without repeated NetCDF advanced indexing."""
    if has_level and values.ndim != 4:
        raise ValueError(f"Expected time-level-latitude-longitude array, got {values.shape}")
    if not has_level and values.ndim != 3:
        raise ValueError(f"Expected time-latitude-longitude array, got {values.shape}")
    count = len(indices["time"])
    output = np.empty(count, dtype=np.float64)
    for start in range(0, count, max(1, int(chunk_rows))):
        stop = min(start + max(1, int(chunk_rows)), count)
        sl = slice(start, stop)
        time_index = indices["time"][sl]
        lat0, lat1 = indices["lat0"][sl], indices["lat1"][sl]
        lon0, lon1 = indices["lon0"][sl], indices["lon1"][sl]
        if has_level:
            level_index = indices["level"][sl]
            v00 = values[time_index, level_index, lat0, lon0]
            v01 = values[time_index, level_index, lat0, lon1]
            v10 = values[time_index, level_index, lat1, lon0]
            v11 = values[time_index, level_index, lat1, lon1]
        else:
            v00 = values[time_index, lat0, lon0]
            v01 = values[time_index, lat0, lon1]
            v10 = values[time_index, lat1, lon0]
            v11 = values[time_index, lat1, lon1]
        wx = indices["lon_weight"][sl]
        wy = indices["lat_weight"][sl]
        output[sl] = (1 - wy) * ((1 - wx) * v00 + wx * v01) + wy * ((1 - wx) * v10 + wx * v11)
    return output


def load_and_sample_month_variable(
    path: Path,
    variable: str,
    frame: pd.DataFrame,
    *,
    has_level: bool,
    chunk_rows: int,
    progress_label: str,
) -> tuple[np.ndarray, dict[str, float]]:
    """Load only requested monthly times/levels contiguously, then sample in NumPy."""
    started = time.perf_counter()
    da = standardize_da(path, variable)
    try:
        target_times = np.sort(pd.to_datetime(frame["time"]).to_numpy(dtype="datetime64[ns]").astype("int64").astype("datetime64[ns]"))
        target_times = np.unique(target_times)
        subset = da.sel(time=target_times)
        if has_level:
            target_levels = sorted(frame["pressure_hpa"].dropna().astype(int).unique().tolist())
            subset = subset.sel(level=target_levels).transpose("time", "level", "latitude", "longitude")
        else:
            subset = subset.transpose("time", "latitude", "longitude")
        expected_mb = int(np.prod(subset.shape)) * np.dtype(subset.dtype).itemsize / 1024**2
        tqdm.write(f"[IGRA] {progress_label}: load {expected_mb:.1f} MiB contiguous subset")
        load_started = time.perf_counter()
        loaded = subset.load()
        load_seconds = time.perf_counter() - load_started
        indices = _point_indices(loaded, frame, has_level)
        sample_started = time.perf_counter()
        result = bilinear_loaded_array(
            np.asarray(loaded.values), indices, has_level=has_level, chunk_rows=chunk_rows
        )
        sample_seconds = time.perf_counter() - sample_started
        diagnostics = {
            "loaded_mib": float(expected_mb),
            "load_seconds": float(load_seconds),
            "sample_seconds": float(sample_seconds),
            "total_seconds": float(time.perf_counter() - started),
        }
        del loaded, subset
        return result, diagnostics
    finally:
        da.close()
        gc.collect()


def _month_partition(output: Path, year: int, month: int) -> Path:
    return output / f"year={year:04d}" / f"month={month}"


def _read_month_success(path: Path) -> dict | None:
    marker = path / "_MONTH_SUCCESS.json"
    if not marker.is_file():
        return None
    with marker.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def promote_month_staging_with_retry(
    staging: Path,
    target: Path,
    *,
    retries: int = 10,
    initial_delay_seconds: float = 0.5,
) -> None:
    """Handle transient Windows file locks while preserving an atomic month promotion."""
    target.parent.mkdir(parents=True, exist_ok=True)
    last_error: PermissionError | None = None
    for attempt in range(max(1, int(retries))):
        try:
            if target.exists():
                shutil.rmtree(target)
            os.replace(staging, target)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt + 1 >= max(1, int(retries)):
                break
            delay = min(float(initial_delay_seconds) * (2**attempt), 5.0)
            tqdm.write(
                f"[IGRA] Windows temporarily locked {staging.name}; retry promotion "
                f"{attempt + 2}/{retries} after {delay:.1f}s"
            )
            time.sleep(delay)
    raise PermissionError(
        f"Could not promote completed month after {retries} attempts. The computed data remain in {staging}; "
        "close Explorer/antivirus handles and rerun with --resume."
    ) from last_error


def recover_completed_staging_month(
    output: Path,
    year: int,
    month: int,
    expected: dict[str, str],
) -> dict | None:
    """Promote a fully written staging month left by an interrupted Windows rename."""
    yyyymm = f"{year:04d}{month:02d}"
    staging = output / ".building" / yyyymm
    audit = _read_month_success(staging)
    if audit is None or not any(staging.glob("*.parquet")):
        return None
    matches = all(audit.get(key) == value for key, value in expected.items()) and audit.get("status") == "complete"
    if not matches:
        return None
    target = _month_partition(output, year, month)
    promote_month_staging_with_retry(staging, target)
    return audit


def write_month_partition_atomic(
    output: Path,
    result: pd.DataFrame,
    year: int,
    month: int,
    audit: dict,
) -> None:
    """Write one month to staging and expose it only after its success marker exists."""
    yyyymm = f"{year:04d}{month:02d}"
    staging = output / ".building" / yyyymm
    target = _month_partition(output, year, month)
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    payload = result.drop(columns=[name for name in ["year", "month"] if name in result])
    pq.write_table(
        pa.Table.from_pandas(payload, preserve_index=False),
        staging / f"part-{yyyymm}.parquet",
        compression="zstd",
    )
    write_json(staging / "_MONTH_SUCCESS.json", audit)
    promote_month_staging_with_retry(staging, target)


def collocate_month_fast(
    frame: pd.DataFrame,
    pressure_dir: Path,
    single_dir: Path,
    yyyymm: str,
    config: dict,
    chunk_rows: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Vectorized monthly ERA5 sampling with one contiguous read per scientific variable."""
    result = frame.copy()
    timings: dict[str, dict[str, float]] = {}
    specs = [
        ("temperature", pressure_dir, f"*temperature*{yyyymm}*.nc", config["era5"]["required_pressure_variables"]["temperature"], True),
        ("specific_humidity", pressure_dir, f"*specific_humidity*{yyyymm}*.nc", config["era5"]["required_pressure_variables"]["specific_humidity"], True),
        ("geopotential", pressure_dir, f"*geopotential*{yyyymm}*.nc", config["era5"]["required_pressure_variables"]["geopotential"], True),
        ("surface_pressure", single_dir, f"*{yyyymm}*.nc", "sp", False),
    ]
    sampled: dict[str, np.ndarray] = {}
    for logical, folder, pattern, variable, has_level in tqdm(specs, desc=f"{yyyymm} variables", leave=False):
        path = find_one(folder, pattern)
        sampled[logical], timings[logical] = load_and_sample_month_variable(
            path, variable, frame, has_level=has_level, chunk_rows=chunk_rows,
            progress_label=f"{yyyymm} {logical}",
        )
        tqdm.write(
            f"[IGRA] {yyyymm} {logical}: load={timings[logical]['load_seconds']:.1f}s, "
            f"sample={timings[logical]['sample_seconds']:.1f}s"
        )
    result["era5_temperature_k"] = sampled["temperature"]
    result["era5_specific_humidity"] = np.maximum(sampled["specific_humidity"], 0.0)
    result["era5_height_m"] = sampled["geopotential"] / G0
    result["era5_surface_pressure_pa"] = sampled["surface_pressure"]
    dry, wet, total, _ = refractivity_components(
        result["pressure_hpa"].to_numpy(), result["era5_temperature_k"].to_numpy(),
        result["era5_specific_humidity"].to_numpy(),
    )
    result["era5_n_dry"], result["era5_n_wet"], result["era5_n"] = dry, wet, total
    valid = result["level_mask"].astype(bool).to_numpy()
    result["below_ground_level"] = result["era5_surface_pressure_pa"] < result["pressure_hpa"] * 100.0
    result["residual_dry"] = np.where(valid, result["igra_n_dry"] - result["era5_n_dry"], np.nan)
    result["residual_wet"] = np.where(valid, result["igra_n_wet"] - result["era5_n_wet"], np.nan)
    result["residual_total"] = np.where(valid, result["igra_n"] - result["era5_n"], np.nan)
    result["source"] = "IGRA_v2.2__ERA5_global_0p5_bilinear_vectorized"
    return result, timings


def read_igra_month(dataset: pads.Dataset, year: int, month: int) -> pd.DataFrame:
    table = dataset.to_table(filter=(pads.field("year") == year) & (pads.field("month") == month))
    return table.to_pandas()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collocate six-level IGRA profiles with an existing global ERA5 archive.")
    parser.add_argument("--config", default="configs/revision2/data_pipeline.yaml")
    parser.add_argument("--era5-root", help="Optional override for era5.root in the data config.")
    parser.add_argument("--era5-audit", help="Optional ERA5 audit manifest; defaults to the revision-2 manifests directory.")
    parser.add_argument("--igra-profile-manifest", help="Optional IGRA profile manifest; defaults to the selected namespace.")
    parser.add_argument("--chunk-rows", type=int, default=20000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip months with matching atomic success markers.")
    parser.add_argument("--max-months", type=int, help="Smoke-test limit.")
    parser.add_argument("--smoke", action="store_true", help="Use isolated smoke inputs and outputs.")
    args = parser.parse_args()
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume are mutually exclusive")
    if args.max_months is not None and not args.smoke:
        parser.error("--max-months is only allowed with --smoke so partial products cannot enter the formal namespace")
    if args.smoke and args.max_months is None:
        parser.error("--smoke requires --max-months")
    config = load_data_config(args.config)
    paths = ensure_data_directories(config)
    namespace = data_namespace(paths, smoke=args.smoke)
    root = resolve_era5_root(config, args.era5_root)
    igra_root = namespace["processed"] / "igra_profiles"
    output = namespace["processed"] / "era5_igra_profiles"
    months = month_keys(config["study"]["start"], config["study"]["end"])
    if args.max_months:
        months = months[: args.max_months]
    audit_manifest_path = Path(args.era5_audit) if args.era5_audit else paths["manifests"] / "era5_archive_manifest.json"
    era5_audit = require_era5_audit(
        config,
        audit_manifest_path,
        root,
        months,
        require_full_study=args.max_months is None,
    )
    igra_manifest_path = (
        Path(args.igra_profile_manifest)
        if args.igra_profile_manifest
        else namespace["manifests"] / "igra_profile_manifest.json"
    )
    igra_manifest = require_igra_profile_manifest(
        config, igra_manifest_path, igra_root, require_formal=not args.smoke
    )
    if args.overwrite:
        shutil.rmtree(output, ignore_errors=True)
    elif output.exists() and any(output.rglob("*.parquet")) and not args.resume:
        raise FileExistsError(f"Output exists; pass --resume or --overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".collocation.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Another optimized collocation process may be using {output}; remove {lock_path} only after confirming it is stopped"
        ) from exc
    os.write(lock_fd, f"pid={os.getpid()}\n".encode("ascii"))
    os.close(lock_fd)
    mode = "smoke" if args.smoke else "formal"
    config_fingerprint = revision2_config_fingerprint(config)
    audit_path = namespace["manifests"] / "era5_igra_collocation_audit.csv"
    output_manifest_path = namespace["manifests"] / "era5_igra_profile_manifest.json"
    try:
        igra_dataset = pads.dataset(str(igra_root), format="parquet", partitioning="hive")
    except Exception:
        lock_path.unlink(missing_ok=True)
        raise
    audit_rows: list[dict] = []

    def write_state(complete: bool) -> None:
        pd.DataFrame(audit_rows).sort_values("yyyymm").to_csv(audit_path, index=False)
        write_json(
            output_manifest_path,
            {
                "mode": mode,
                "complete": complete,
                "dataset_version": config["study"].get("dataset_version"),
                "config_fingerprint": config_fingerprint,
                "era5_root": str(root.resolve()),
                "igra_root": str(igra_root.resolve()),
                "output": str(output.resolve()),
                "era5_audit_manifest": str(audit_manifest_path.resolve()),
                "era5_archive_fingerprint": era5_audit["archive_fingerprint"],
                "igra_profile_manifest": str(igra_manifest_path.resolve()),
                "igra_profile_fingerprint": igra_manifest["data_fingerprint"],
                "pressure_levels_hpa": config["study"]["pressure_levels_hpa"],
                "months": months,
                "completed_months": [row["yyyymm"] for row in audit_rows if row.get("status") == "complete"],
                "algorithm": "monthly contiguous ERA5 subset + NumPy vectorized bilinear interpolation",
                "interpolation": "exact time and pressure level; bilinear latitude-longitude",
                "monthly_audit": sorted(audit_rows, key=lambda row: row["yyyymm"]),
            },
        )

    try:
        for yyyymm in tqdm(months, desc="Collocate ERA5 months"):
            year, month = int(yyyymm[:4]), int(yyyymm[4:])
            partition = _month_partition(output, year, month)
            expected_month_evidence = {
                "yyyymm": yyyymm,
                "config_fingerprint": config_fingerprint,
                "era5_archive_fingerprint": era5_audit["archive_fingerprint"],
                "igra_profile_fingerprint": igra_manifest["data_fingerprint"],
            }
            if args.resume and not partition.exists():
                recovered = recover_completed_staging_month(
                    output, year, month, expected_month_evidence
                )
                if recovered is not None:
                    tqdm.write(f"[IGRA] {yyyymm}: recovered completed staging month")
            existing = _read_month_success(partition) if args.resume else None
            if existing is not None:
                matches = all(existing.get(key) == value for key, value in expected_month_evidence.items()) and existing.get("status") == "complete"
                if matches:
                    audit_rows.append(existing)
                    tqdm.write(f"[IGRA] {yyyymm}: reuse completed month")
                    write_state(False)
                    continue
                shutil.rmtree(partition, ignore_errors=True)
            frame = read_igra_month(igra_dataset, year, month)
            if frame.empty:
                audit_rows.append({"yyyymm": yyyymm, "rows": 0, "status": "no IGRA profiles"})
                write_state(False)
                continue
            month_started = time.perf_counter()
            pressure_dir = root / f"{year:04d}" / f"{month:02d}" / "pressure_levels"
            single_dir = root / f"{year:04d}" / f"{month:02d}" / "single_levels"
            result, timings = collocate_month_fast(
                frame, pressure_dir, single_dir, yyyymm, config, args.chunk_rows
            )
            audit = {
                "yyyymm": yyyymm,
                "rows": int(len(result)),
                "profiles": int(result[["station_id", "time"]].drop_duplicates().shape[0]),
                "stations": int(result["station_id"].nunique()),
                "valid_levels": int(result["level_mask"].sum()),
                "below_ground_levels": int(result["below_ground_level"].sum()),
                "elapsed_seconds": float(time.perf_counter() - month_started),
                "variable_timings": timings,
                "config_fingerprint": config_fingerprint,
                "era5_archive_fingerprint": era5_audit["archive_fingerprint"],
                "igra_profile_fingerprint": igra_manifest["data_fingerprint"],
                "status": "complete",
            }
            write_month_partition_atomic(output, result, year, month, audit)
            audit_rows.append(audit)
            write_state(False)
            tqdm.write(
                f"[IGRA] {yyyymm}: complete, rows={len(result):,}, elapsed={audit['elapsed_seconds'] / 60:.1f} min"
            )
            del result, frame
            gc.collect()
        complete = len(audit_rows) == len(months) and all(row.get("status") == "complete" for row in audit_rows)
        if complete:
            write_json(
                output / "_SUCCESS",
                {
                    "mode": mode,
                    "dataset_version": config["study"].get("dataset_version"),
                    "months": months,
                    "algorithm": "monthly_contiguous_numpy_bilinear_v2",
                    "igra_profile_fingerprint": igra_manifest["data_fingerprint"],
                    "era5_archive_fingerprint": era5_audit["archive_fingerprint"],
                },
            )
        write_state(complete)
        if not complete:
            raise RuntimeError("ERA5-IGRA collocation did not produce a complete month set; inspect the monthly audit")
        shutil.rmtree(output / ".building", ignore_errors=True)
        print(f"Collocated dataset: {output.resolve()}")
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
