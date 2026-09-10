from __future__ import annotations

import argparse
import calendar
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from _bootstrap import PROJECT_ROOT  # noqa: F401
from igra_forecast.revision2_data import (
    ensure_data_directories,
    era5_archive_fingerprint,
    load_data_config,
    month_keys,
    resolve_era5_root,
    sha256_file,
    write_json,
)


FILE_TOKENS = {"temperature": "temperature", "specific_humidity": "specific_humidity", "geopotential": "geopotential"}


def find_one(folder: Path, token: str, yyyymm: str) -> Path | None:
    matches = sorted(folder.glob(f"*{token}*{yyyymm}*.nc"))
    return matches[0] if len(matches) == 1 else None


def inspect_pressure_file(
    path: Path,
    expected_var: str,
    levels: list[int],
    hours: list[int],
    expected_grid: float,
    expected_time_count: int,
) -> dict[str, object]:
    with xr.open_dataset(path) as dataset:
        variables = list(dataset.data_vars)
        level_name = next((name for name in ["pressure_level", "level", "isobaricInhPa"] if name in dataset.coords), None)
        time_name = next((name for name in ["valid_time", "time"] if name in dataset.coords), None)
        lat_name = next((name for name in ["latitude", "lat"] if name in dataset.coords), None)
        lon_name = next((name for name in ["longitude", "lon"] if name in dataset.coords), None)
        if not all([level_name, time_name, lat_name, lon_name]):
            return {"valid": False, "error": "missing coordinate", "variables": variables}
        actual_levels = set(np.asarray(dataset[level_name].values).astype(int).tolist())
        actual_hours = set(pd.DatetimeIndex(dataset[time_name].values).hour.tolist())
        lat = np.asarray(dataset[lat_name].values, dtype=float)
        lon = np.asarray(dataset[lon_name].values, dtype=float)
        lat_step = float(np.nanmedian(np.abs(np.diff(lat))))
        lon_step = float(np.nanmedian(np.abs(np.diff(lon))))
        missing_levels = sorted(set(levels) - actual_levels)
        missing_hours = sorted(set(hours) - actual_hours)
        grid_valid = (
            len(lat) == 361
            and len(lon) == 720
            and np.isclose(lat_step, expected_grid)
            and np.isclose(lon_step, expected_grid)
        )
        time_valid = int(dataset.sizes[time_name]) == expected_time_count
        valid = expected_var in dataset.data_vars and not missing_levels and not missing_hours and grid_valid and time_valid
        return {
            "valid": valid, "variables": variables, "missing_levels": missing_levels, "missing_hours": missing_hours,
            "latitude_count": len(lat), "longitude_count": len(lon), "latitude_step": lat_step,
            "longitude_step": lon_step, "grid_valid": grid_valid,
            "time_count": int(dataset.sizes[time_name]), "expected_time_count": expected_time_count,
            "time_valid": time_valid,
        }


def inspect_single_file(
    path: Path,
    required_variables: list[str],
    hours: list[int],
    expected_grid: float,
    expected_time_count: int,
) -> dict[str, object]:
    with xr.open_dataset(path) as dataset:
        time_name = next((name for name in ["valid_time", "time"] if name in dataset.coords), None)
        lat_name = next((name for name in ["latitude", "lat"] if name in dataset.coords), None)
        lon_name = next((name for name in ["longitude", "lon"] if name in dataset.coords), None)
        if not all([time_name, lat_name, lon_name]):
            return {"valid": False, "error": "missing coordinate", "variables": list(dataset.data_vars)}
        missing_variables = sorted(set(required_variables) - set(dataset.data_vars))
        actual_hours = set(pd.DatetimeIndex(dataset[time_name].values).hour.tolist())
        missing_hours = sorted(set(hours) - actual_hours)
        lat = np.asarray(dataset[lat_name].values, dtype=float)
        lon = np.asarray(dataset[lon_name].values, dtype=float)
        lat_step = float(np.nanmedian(np.abs(np.diff(lat))))
        lon_step = float(np.nanmedian(np.abs(np.diff(lon))))
        grid_valid = (
            len(lat) == 361
            and len(lon) == 720
            and np.isclose(lat_step, expected_grid)
            and np.isclose(lon_step, expected_grid)
        )
        time_valid = int(dataset.sizes[time_name]) == expected_time_count
        return {
            "valid": not missing_variables and not missing_hours and grid_valid and time_valid,
            "variables": list(dataset.data_vars), "missing_variables": missing_variables,
            "missing_hours": missing_hours, "latitude_count": len(lat), "longitude_count": len(lon),
            "latitude_step": lat_step, "longitude_step": lon_step, "grid_valid": grid_valid,
            "time_count": int(dataset.sizes[time_name]), "expected_time_count": expected_time_count,
            "time_valid": time_valid,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a reusable global ERA5 0.5-degree archive without copying it.")
    parser.add_argument("--config", default="configs/revision2/data_pipeline.yaml")
    parser.add_argument("--era5-root", help="Optional override for era5.root in the data config.")
    parser.add_argument("--static-file", help="Optional static NetCDF containing lsm and z.")
    parser.add_argument("--skip-checksums", action="store_true", help="Faster header-only audit.")
    parser.add_argument("--max-months", type=int, help="Smoke-test limit.")
    args = parser.parse_args()
    config = load_data_config(args.config)
    paths = ensure_data_directories(config)
    root = resolve_era5_root(config, args.era5_root)
    if not root.exists():
        raise FileNotFoundError(root)
    levels = list(map(int, config["study"]["pressure_levels_hpa"]))
    archive_hours = list(map(int, config["era5"]["expected_hours_utc"]))
    expected_grid = float(config["era5"]["expected_grid_degrees"])
    records = []
    months = month_keys(config["study"]["start"], config["study"]["end"])
    if args.max_months:
        months = months[: args.max_months]
    for yyyymm in tqdm(months, desc="Audit ERA5 months"):
        year, month = yyyymm[:4], yyyymm[4:]
        pressure_dir = root / year / month / "pressure_levels"
        single_dir = root / year / month / "single_levels"
        expected_time_count = calendar.monthrange(int(year), int(month))[1] * len(archive_hours)
        source_manifest_present = (root / year / month / "manifest.json").is_file()
        success_marker_present = (root / year / month / "_SUCCESS").is_file()
        for logical_name, token in FILE_TOKENS.items():
            expected_var = config["era5"]["required_pressure_variables"][logical_name]
            path = find_one(pressure_dir, token, yyyymm)
            row = {
                "yyyymm": yyyymm, "kind": "pressure", "logical_variable": logical_name,
                "path": str(path.resolve()) if path else "", "source_manifest_present": source_manifest_present,
                "success_marker_present": success_marker_present,
            }
            if path is None:
                row.update(valid=False, file_valid=False, error="file missing or ambiguous")
            else:
                try:
                    row.update(inspect_pressure_file(path, expected_var, levels, archive_hours, expected_grid, expected_time_count))
                    row["file_valid"] = bool(row["valid"])
                    if not source_manifest_present or not success_marker_present:
                        row["valid"] = False
                        row["error"] = "source manifest or _SUCCESS marker missing"
                    row["bytes"] = path.stat().st_size
                    if not args.skip_checksums:
                        row["sha256"] = sha256_file(path)
                except Exception as exc:  # audit must preserve the failing path
                    row.update(valid=False, file_valid=False, error=f"{type(exc).__name__}: {exc}")
            records.append(row)
        single_matches = sorted(single_dir.glob(f"*{yyyymm}*.nc"))
        single_path = single_matches[0] if len(single_matches) == 1 else None
        row = {
            "yyyymm": yyyymm, "kind": "single", "logical_variable": "surface_pressure",
            "path": str(single_path.resolve()) if single_path else "", "source_manifest_present": source_manifest_present,
            "success_marker_present": success_marker_present,
        }
        if single_path is None:
            row.update(valid=False, file_valid=False, error="single-level file missing or ambiguous")
        else:
            try:
                row.update(
                    inspect_single_file(
                        single_path,
                        list(config["era5"]["required_single_variables"]),
                        archive_hours,
                        expected_grid,
                        expected_time_count,
                    )
                )
                row["file_valid"] = bool(row["valid"])
                row["variables"] = json.dumps(row["variables"])
                if not source_manifest_present or not success_marker_present:
                    row["valid"] = False
                    row["error"] = "source manifest or _SUCCESS marker missing"
                row["bytes"] = single_path.stat().st_size
                if not args.skip_checksums:
                    row["sha256"] = sha256_file(single_path)
            except Exception as exc:
                row.update(valid=False, file_valid=False, error=f"{type(exc).__name__}: {exc}")
        records.append(row)
    frame = pd.DataFrame(records)
    frame.to_csv(paths["manifests"] / "era5_archive_audit.csv", index=False)
    static_path = Path(args.static_file) if args.static_file else root.parent / "static" / "era5_global_0p5_static.nc"
    static_info: dict[str, object] = {"path": str(static_path), "valid": False, "missing_variables": config["era5"]["required_static_variables"]}
    if static_path.exists():
        with xr.open_dataset(static_path) as ds:
            missing = sorted(set(config["era5"]["required_static_variables"]) - set(ds.data_vars))
            static_info = {"path": str(static_path), "valid": not missing, "missing_variables": missing}
    invalid = frame.loc[~frame["valid"].fillna(False).astype(bool)]
    structurally_invalid = frame.loc[~frame["file_valid"].fillna(False).astype(bool)]
    invalid_months = sorted(invalid["yyyymm"].astype(str).unique().tolist())
    study_months = month_keys(config["study"]["start"], config["study"]["end"])
    complete = len(invalid) == 0 and bool(static_info["valid"])
    manifest = {
        "era5_root": str(root.resolve()), "pressure_levels_hpa": levels,
        "months_expected": len(months), "records": len(frame), "invalid_records": len(invalid),
        "structurally_invalid_records": len(structurally_invalid),
        "audited_months": months, "invalid_months": invalid_months,
        "required_record_count": len(months) * 4,
        "expected_grid_degrees": expected_grid, "expected_hours_utc": archive_hours,
        "archive_fingerprint": era5_archive_fingerprint(config, root),
        "static": static_info, "complete": complete,
        "full_study_complete": complete and months == study_months,
    }
    write_json(paths["manifests"] / "era5_archive_manifest.json", manifest)
    print(f"ERA5 structural files: {len(frame) - len(structurally_invalid)}/{len(frame)} valid")
    print(f"ERA5 files with source evidence: {len(frame) - len(invalid)}/{len(frame)} valid")
    if len(invalid) or not static_info["valid"]:
        print(invalid[["yyyymm", "logical_variable", "error"]].to_string(index=False))
        if not static_info["valid"]:
            print(f"Static file invalid or missing: {static_info}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
