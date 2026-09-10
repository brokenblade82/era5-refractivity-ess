from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

from _bootstrap import PROJECT_ROOT  # noqa: F401
from igra_forecast.revision2_data import (
    ensure_data_directories,
    load_data_config,
    month_keys,
    require_collocation_manifest,
    sha256_file,
    write_json,
)


FINITE_VALID_COLUMNS = [
    "igra_temperature_k", "igra_specific_humidity", "igra_height_m",
    "era5_temperature_k", "era5_specific_humidity", "era5_height_m", "era5_surface_pressure_pa",
    "igra_n_dry", "igra_n_wet", "igra_n", "era5_n_dry", "era5_n_wet", "era5_n",
    "residual_dry", "residual_wet", "residual_total",
]


def audit_month_frame(frame: pd.DataFrame, levels: list[int], min_valid_levels: int) -> dict[str, object]:
    """Audit one monthly partition; used by both production checks and unit tests."""
    if frame.empty:
        return {"rows": 0, "passed": False, "status": "empty"}
    frame = frame.copy()
    valid = frame["level_mask"].astype(bool)
    profile_keys = ["station_id", "time"]
    profile_size = frame.groupby(profile_keys, observed=True).size()
    valid_per_profile = frame.groupby(profile_keys, observed=True)["level_mask"].sum()
    residual_identity = frame.loc[valid, "residual_total"] - (
        frame.loc[valid, "residual_dry"] + frame.loc[valid, "residual_wet"]
    )
    igra_identity = frame.loc[valid, "igra_n"] - (
        frame.loc[valid, "igra_n_dry"] + frame.loc[valid, "igra_n_wet"]
    )
    era_identity = frame["era5_n"] - (frame["era5_n_dry"] + frame["era5_n_wet"])
    below_ground_expected = frame["era5_surface_pressure_pa"] < frame["pressure_hpa"] * 100.0
    invalid_residuals_are_nan = bool(
        frame.loc[~valid, ["residual_dry", "residual_wet", "residual_total"]].isna().all().all()
    )
    finite_valid = bool(np.isfinite(frame.loc[valid, FINITE_VALID_COLUMNS].to_numpy(dtype=float)).all())
    result: dict[str, object] = {
        "rows": int(len(frame)),
        "profiles": int(frame[profile_keys].drop_duplicates().shape[0]),
        "profiles_with_exactly_three_valid_levels": int((valid_per_profile == 3).sum()),
        "fraction_profiles_with_exactly_three_valid_levels": float((valid_per_profile == 3).mean()),
        "stations": int(frame["station_id"].nunique()),
        "valid_levels": int(frame["level_mask"].sum()),
        "below_ground_levels": int(frame["below_ground_level"].sum()),
        "all_profiles_have_six_rows": bool((profile_size == len(levels)).all()),
        "minimum_valid_levels": int(valid_per_profile.min()),
        "profiles_meet_minimum": bool((valid_per_profile >= min_valid_levels).all()),
        "level_set_exact": sorted(frame["pressure_hpa"].dropna().astype(int).unique(), reverse=True) == levels,
        "synoptic_hours_valid": set(pd.DatetimeIndex(frame["time"]).hour.unique()).issubset({0, 12}),
        "duplicate_station_time_level_rows": int(frame.duplicated(["station_id", "time", "pressure_hpa"]).sum()),
        "finite_valid_core_variables": finite_valid,
        "invalid_level_residuals_are_nan": invalid_residuals_are_nan,
        "below_ground_flag_consistent": bool((below_ground_expected.to_numpy() == frame["below_ground_level"].astype(bool).to_numpy()).all()),
        "qc_flags_present": bool(frame["qc_flags"].notna().all()),
        "max_abs_residual_identity_error": float(np.nanmax(np.abs(residual_identity))),
        "max_abs_igra_identity_error": float(np.nanmax(np.abs(igra_identity))),
        "max_abs_era5_identity_error": float(np.nanmax(np.abs(era_identity))),
    }
    result["identities_within_1e-9"] = bool(
        max(
            result["max_abs_residual_identity_error"],
            result["max_abs_igra_identity_error"],
            result["max_abs_era5_identity_error"],
        ) < 1e-9
    )
    fatal = [
        "all_profiles_have_six_rows", "profiles_meet_minimum", "level_set_exact", "synoptic_hours_valid",
        "finite_valid_core_variables", "invalid_level_residuals_are_nan", "below_ground_flag_consistent",
        "qc_flags_present", "identities_within_1e-9",
    ]
    result["passed"] = all(bool(result[name]) for name in fatal) and result["duplicate_station_time_level_rows"] == 0
    result["status"] = "complete" if result["passed"] else "failed"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the revision-2 collocated data contract and physical identities.")
    parser.add_argument("--config", default="configs/revision2/data_pipeline.yaml")
    parser.add_argument("--input", default="data/revision2/processed/era5_igra_profiles")
    parser.add_argument("--station-split", default="data/revision2/manifests/station_split.csv")
    parser.add_argument("--collocation-manifest", default="data/revision2/manifests/era5_igra_profile_manifest.json")
    args = parser.parse_args()
    config = load_data_config(args.config)
    paths = ensure_data_directories(config)
    collocation = require_collocation_manifest(
        config, args.collocation_manifest, args.input, require_formal=True
    )
    dataset = pads.dataset(args.input, format="parquet", partitioning="hive")
    required = {
        "station_id", "time", "latitude", "longitude", "station_elevation_m", "pressure_hpa", "level_mask",
        "igra_temperature_k", "igra_specific_humidity", "igra_height_m", "qc_flags", "era5_temperature_k",
        "era5_specific_humidity", "era5_height_m", "igra_n_dry", "igra_n_wet", "igra_n",
        "era5_n_dry", "era5_n_wet", "era5_n", "residual_dry", "residual_wet", "residual_total",
        "era5_surface_pressure_pa", "below_ground_level", "year", "month", "source",
    }
    missing = sorted(required - set(dataset.schema.names))
    if missing:
        raise ValueError(f"Missing required data fields: {missing}")
    levels = sorted(map(int, config["study"]["pressure_levels_hpa"]), reverse=True)
    months = month_keys(config["study"]["start"], config["study"]["end"])
    monthly_rows = []
    level_rows: list[dict[str, object]] = []
    combination_rows: list[dict[str, object]] = []
    station_role_profile_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    station_role_months: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    high_elevation_stations: set[str] = set()
    stations: set[str] = set()
    actual_levels: set[int] = set()
    actual_hours: set[int] = set()
    time_min: pd.Timestamp | None = None
    time_max: pd.Timestamp | None = None
    for yyyymm in months:
        year, month = int(yyyymm[:4]), int(yyyymm[4:])
        frame = dataset.to_table(
            columns=sorted(required - {"source", "year", "month"}),
            filter=(pads.field("year") == year) & (pads.field("month") == month),
        ).to_pandas()
        row = {"yyyymm": yyyymm, **audit_month_frame(frame, levels, int(config["study"]["min_valid_levels_per_profile"]))}
        monthly_rows.append(row)
        if not frame.empty:
            for level, group in frame.groupby("pressure_hpa", observed=True):
                level_rows.append({
                    "yyyymm": yyyymm, "pressure_hpa": int(level), "rows": int(len(group)),
                    "valid_rows": int(group["level_mask"].sum()), "valid_fraction": float(group["level_mask"].mean()),
                })
            three = frame.groupby(["station_id", "time"], observed=True).filter(lambda group: int(group["level_mask"].sum()) == 3)
            if not three.empty:
                combos = three.loc[three["level_mask"].astype(bool)].groupby(["station_id", "time"], observed=True)["pressure_hpa"].apply(
                    lambda values: "-".join(map(str, sorted(map(int, values), reverse=True)))
                ).value_counts()
                combination_rows.extend({"yyyymm": yyyymm, "level_combination_hpa": key, "profiles": int(value)} for key, value in combos.items())
            profile_times = frame[["station_id", "time"]].drop_duplicates()
            timestamps_role = pd.DatetimeIndex(pd.to_datetime(profile_times["time"], utc=True))
            roles = np.select(
                [timestamps_role < pd.Timestamp("2025-01-01", tz="UTC"), timestamps_role < pd.Timestamp("2025-04-01", tz="UTC")],
                ["development_2024", "calibration_2025q1"], default="evaluation_2025q2_q4",
            )
            for (station_id, role), count in profile_times.assign(temporal_role=roles).groupby(["station_id", "temporal_role"]).size().items():
                station_role_profile_counts[str(station_id)][str(role)] += int(count)
                station_role_months[str(station_id)][str(role)].add(yyyymm)
            high_elevation_stations.update(map(str, frame.loc[frame["station_elevation_m"] >= 1500, "station_id"].unique()))
            stations.update(map(str, frame["station_id"].unique()))
            actual_levels.update(map(int, frame["pressure_hpa"].dropna().unique()))
            timestamps = pd.DatetimeIndex(frame["time"])
            actual_hours.update(map(int, timestamps.hour.unique()))
            frame_min, frame_max = frame["time"].min(), frame["time"].max()
            time_min = frame_min if time_min is None else min(time_min, frame_min)
            time_max = frame_max if time_max is None else max(time_max, frame_max)
    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(paths["manifests"] / "revision2_monthly_data_audit.csv", index=False)
    pd.DataFrame(level_rows).to_csv(paths["manifests"] / "revision2_level_availability.csv", index=False)
    pd.DataFrame(combination_rows).to_csv(paths["manifests"] / "revision2_three_level_combinations.csv", index=False)
    coverage_rows = []
    thresholds = {key: int(value) for key, value in config["study"]["station_temporal_coverage"].items()}
    for station_id in sorted(stations):
        row: dict[str, object] = {"station_id": station_id}
        for role, threshold in thresholds.items():
            count = int(station_role_profile_counts[station_id].get(role, 0))
            row[f"profiles_{role}"] = count
            row[f"months_{role}"] = len(station_role_months[station_id].get(role, set()))
            row[f"passes_{role}"] = count >= threshold
        row["passes_all_temporal_periods"] = all(bool(row[f"passes_{role}"]) for role in thresholds)
        coverage_rows.append(row)
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(paths["manifests"] / "revision2_station_temporal_coverage_audit.csv", index=False)
    actual_levels_sorted = sorted(actual_levels, reverse=True)
    checks: dict[str, object] = {
        "dataset_version": config["study"].get("dataset_version"),
        "expected_levels": levels,
        "actual_levels": actual_levels_sorted,
        "level_set_exact": actual_levels_sorted == levels,
        "months_expected": len(months),
        "months_complete": int(monthly["passed"].sum()),
        "all_months_complete": bool(monthly["passed"].all() and len(monthly) == len(months)),
        "rows": int(monthly["rows"].sum()),
        "stations": len(stations),
        "profiles": int(monthly["profiles"].sum()),
        "profiles_with_exactly_three_valid_levels": int(monthly["profiles_with_exactly_three_valid_levels"].sum()),
        "high_elevation_stations_retained": len(high_elevation_stations),
        "all_stations_pass_temporal_coverage": bool(coverage["passes_all_temporal_periods"].all()),
        "time_min": str(time_min), "time_max": str(time_max),
        "synoptic_hours": sorted(actual_hours),
        "minimum_valid_levels": int(monthly["minimum_valid_levels"].min()),
        "max_abs_residual_identity_error": float(monthly["max_abs_residual_identity_error"].max()),
        "max_abs_igra_identity_error": float(monthly["max_abs_igra_identity_error"].max()),
        "max_abs_era5_identity_error": float(monthly["max_abs_era5_identity_error"].max()),
    }
    checks["all_identities_within_1e-9"] = bool(max(
        checks["max_abs_residual_identity_error"], checks["max_abs_igra_identity_error"], checks["max_abs_era5_identity_error"]
    ) < 1e-9)
    split_path = Path(args.station_split)
    split = pd.read_csv(split_path)
    checks["split_station_count"] = int(split["station_id"].nunique())
    checks["split_matches_data_stations"] = set(map(str, split["station_id"])) == stations
    checks["split_roles"] = split["split"].value_counts().to_dict()
    checks["split_station_unique"] = bool(not split["station_id"].duplicated().any())
    split_region_counts = split.groupby(["split", "macro_region"])["station_id"].nunique().unstack(fill_value=0)
    southern_minimums = {
        key: int(value) for key, value in config["splits"].get("minimum_southern_extratropical_stations", {}).items()
    }
    checks["southern_split_minimums_met"] = all(
        int(split_region_counts.get("southern_extratropics", pd.Series(dtype=int)).get(role, 0)) >= minimum
        for role, minimum in southern_minimums.items()
    )
    license_path = split_path.parent / "protocol_license_matrix.csv"
    checks["protocol_license_matrix_present"] = license_path.is_file()
    checks["collocation_manifest"] = str(Path(args.collocation_manifest).resolve())
    checks["collocation_manifest_sha256"] = sha256_file(args.collocation_manifest)
    fatal = [
        "level_set_exact", "all_months_complete", "all_identities_within_1e-9", "split_matches_data_stations",
        "all_stations_pass_temporal_coverage", "split_station_unique", "southern_split_minimums_met",
        "protocol_license_matrix_present",
    ]
    checks["passed"] = all(bool(checks[name]) for name in fatal)
    output = paths["manifests"] / "revision2_data_audit.json"
    write_json(output, checks)
    print(pd.Series(checks).to_string())
    if not checks["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
