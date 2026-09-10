from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from _bootstrap import PROJECT_ROOT  # noqa: F401
from igra_forecast.revision2_data import sha256_file, write_json
from igra_forecast.revision2_final import (
    add_cosmic_diagnostics,
    archive_block_bootstrap,
    grouped_cosmic_metrics,
    physical_station_bootstrap,
)


def read_config(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def require(path: str | Path) -> Path:
    result = Path(path)
    if not result.is_file():
        raise FileNotFoundError(f"Required frozen input is missing: {result}")
    return result


def read_cosmic_predictions(root: Path) -> pd.DataFrame:
    files = sorted((root / "predictions").glob("predictions_*.parquet"))
    if len(files) != 24:
        raise ValueError(f"Formal COSMIC-2 diagnostics require 24 monthly predictions, found {len(files)}")
    columns = [
        "profile_id", "time", "latitude", "longitude", "height_m", "observed_n", "source_archive",
        "land_sea", "era5_n_height", "prediction_correction", "prediction_n_height",
        "prediction_std_height", "evaluation_mask",
    ]
    return pd.concat(
        [pd.read_parquet(path, columns=columns) for path in tqdm(files, desc="Read COSMIC-2 predictions")],
        ignore_index=True,
    )


def archive_audit(frame: pd.DataFrame) -> pd.DataFrame:
    profiles = frame[["profile_id", "time", "source_archive"]].drop_duplicates().copy()
    profiles["time"] = pd.to_datetime(profiles["time"], utc=True)
    profiles["utc_date"] = profiles["time"].dt.strftime("%Y-%m-%d")
    rows = []
    for archive, selected in profiles.groupby("source_archive", observed=True):
        name = Path(str(archive)).name
        match = re.search(r"_(\d{4})_(\d{3})\.tar\.gz$", name)
        rows.append({
            "archive": str(archive), "archive_name": name,
            "preregistered_year": int(match.group(1)) if match else np.nan,
            "preregistered_day_of_year": int(match.group(2)) if match else np.nan,
            "profiles": int(len(selected)), "unique_utc_dates": int(selected["utc_date"].nunique()),
            "first_utc_date": selected["utc_date"].min(), "last_utc_date": selected["utc_date"].max(),
            "bootstrap_unit": "archive_month",
        })
    return pd.DataFrame(rows).sort_values(["preregistered_year", "preregistered_day_of_year"])


def cross_platform_table(config: dict[str, object], cosmic_root: Path) -> pd.DataFrame:
    paper = Path(config["paper_output_root"])
    internal = pd.read_csv(require(paper / "internal_generalization_summary.csv"))
    probability = pd.read_csv(require(paper / "probability_transfer_summary.csv"))
    external = pd.read_csv(require(paper / "external_validation_summary.csv"))
    rows: list[dict[str, object]] = []

    era5 = internal.query("protocol == 'space_time_holdout' and model == 'era5'").iloc[0]
    hgb = internal.query("protocol == 'space_time_holdout' and model == 'hgb'").iloc[0]
    p = probability.query("domain == 'IGRA_space_time_holdout' and model == 'hgb_prob_hetero_structured'").iloc[0]
    rows.append({
        "source": "IGRA", "coordinate": "standard_pressure_levels", "evidence_level": "primary_confirmatory_internal",
        "primary_cluster_unit": "station", "era5_rmse": era5["rmse"], "hgb_rmse": hgb["rmse"],
        "hgb_minus_era5_rmse": hgb["station_mean_rmse_difference_vs_era5"],
        "ci_lower": hgb["station_ci_lower"], "ci_upper": hgb["station_ci_upper"],
        "marginal_coverage_90": p["marginal_coverage_90"], "crps": p["crps"],
    })

    rapsodi = external.query("source == 'RAPSODI_INMG'").iloc[0]
    rapsodi_boot = pd.read_csv(require(paper / "external" / "rapsodi" / "external_cluster_bootstrap.csv"))
    rb = rapsodi_boot.query("method == 'launch_or_sampling_day_block'").iloc[0]
    rp = probability.query("domain == 'RAPSODI_INMG'").iloc[0]
    rows.append({
        "source": "RAPSODI_INMG", "coordinate": "measured_pressure", "evidence_level": "external_regional_diagnostic",
        "primary_cluster_unit": "launch_day", "era5_rmse": rapsodi["era5_rmse"], "hgb_rmse": rapsodi["model_rmse"],
        "hgb_minus_era5_rmse": rb["mean_difference"], "ci_lower": rb["ci_lower"], "ci_upper": rb["ci_upper"],
        "marginal_coverage_90": rp["marginal_coverage_90"], "crps": rp["crps"],
    })

    height = external.query("source == 'COSMIC2' and coordinate == 'MSL_geometric_height'").iloc[0]
    height_boot = pd.read_csv(require(cosmic_root / "month_bootstrap.csv")).iloc[0]
    hp = probability.query("domain == 'COSMIC2' and coordinate == 'MSL_geometric_height'").iloc[0]
    rows.append({
        "source": "COSMIC2", "coordinate": "MSL_geometric_height", "evidence_level": "primary_cross_platform_diagnostic",
        "primary_cluster_unit": "archive_month", "era5_rmse": height["era5_rmse"], "hgb_rmse": height["model_rmse"],
        "hgb_minus_era5_rmse": height_boot["mean_difference"], "ci_lower": height_boot["ci_lower"],
        "ci_upper": height_boot["ci_upper"], "marginal_coverage_90": hp["marginal_coverage_90"], "crps": hp["crps"],
    })

    dry = external.query("source == 'COSMIC2' and coordinate == 'dry_pressure_sensitivity'").iloc[0]
    dry_boot = pd.read_csv(require(paper / "external" / "cosmic2_dry_pressure_sensitivity" / "external_cluster_bootstrap.csv"))
    db = dry_boot.query("method == 'month_block'").iloc[0]
    dp = probability.query("domain == 'COSMIC2' and coordinate == 'dry_pressure_sensitivity'").iloc[0]
    rows.append({
        "source": "COSMIC2", "coordinate": "dry_pressure_sensitivity", "evidence_level": "coordinate_sensitivity",
        "primary_cluster_unit": "archive_month", "era5_rmse": dry["era5_rmse"], "hgb_rmse": dry["model_rmse"],
        "hgb_minus_era5_rmse": db["mean_difference"], "ci_lower": db["ci_lower"], "ci_upper": db["ci_upper"],
        "marginal_coverage_90": dp["marginal_coverage_90"], "crps": dp["crps"],
    })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize non-training diagnostics for the Revision 2.3 paper.")
    parser.add_argument("--config", default="configs/revision2/publication.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--bootstrap-replicates", type=int)
    args = parser.parse_args()
    config = read_config(args.config)
    output = Path(args.output_dir or Path(config["final_output_root"]) / "statistics")
    output.mkdir(parents=True, exist_ok=True)
    cosmic_root = Path(config["output_root"]) / "cosmic2_height"
    cosmic_manifest_path = require(cosmic_root / "manifest.json")
    cosmic_manifest = json.loads(cosmic_manifest_path.read_text(encoding="utf-8"))
    if not cosmic_manifest.get("complete") or cosmic_manifest.get("smoke") or cosmic_manifest.get("dry_pressure_used") is not False:
        raise ValueError("Final diagnostics require the complete formal MSL-geometric-height COSMIC-2 result")
    replicates = int(args.bootstrap_replicates or config["cosmic2_height"]["bootstrap_replicates"])
    seed = int(config["external_statistics"]["seed"])

    raw_cosmic = read_cosmic_predictions(cosmic_root)
    cosmic = add_cosmic_diagnostics(raw_cosmic)
    level = grouped_cosmic_metrics(cosmic, "height_level", ["height_km"])
    level.to_csv(output / "cosmic_height_level_metrics.csv", index=False)

    group_specs = [
        ("height_band", ["height_band"]), ("macro_region", ["macro_region"]), ("land_sea", ["land_sea"]),
        ("height_band_x_macro_region", ["height_band", "macro_region"]),
        ("height_band_x_land_sea", ["height_band", "land_sea"]),
    ]
    grouped = pd.concat([grouped_cosmic_metrics(cosmic, name, columns) for name, columns in group_specs], ignore_index=True)
    grouped.to_csv(output / "cosmic_height_band_region_metrics.csv", index=False)
    probability_columns = [
        "grouping", "height_km", "height_band", "macro_region", "land_sea", "n", "mean_predictive_sd", "crps",
        "interval_score_90", "mean_interval_width_90", "coverage_50", "coverage_60", "coverage_70",
        "coverage_80", "coverage_90", "coverage_95",
    ]
    probability = pd.concat([level, grouped], ignore_index=True).reindex(columns=probability_columns)
    probability.to_csv(output / "cosmic_height_probability_metrics.csv", index=False)

    block_specs = [("height_level", ["height_km"]), *group_specs]
    block = pd.concat(
        [archive_block_bootstrap(cosmic, name, columns, replicates, seed) for name, columns in tqdm(block_specs, desc="Archive-month bootstrap")],
        ignore_index=True,
    )
    block.to_csv(output / "cosmic_height_block_bootstrap.csv", index=False)
    archives = archive_audit(raw_cosmic)
    archives.to_csv(output / "cosmic_sampling_archive_audit.csv", index=False)

    physics_run = Path(config["confirmatory_root"]) / "space_time_holdout" / "hgb_run42"
    physics_manifest = json.loads(require(physics_run / "protocol_manifest.json").read_text(encoding="utf-8"))
    if not physics_manifest.get("complete") or physics_manifest.get("smoke"):
        raise ValueError("Physical bootstrap requires the complete frozen space-time HGB run")
    physics = pd.read_parquet(require(physics_run / "predictions.parquet"))
    physics = physics.loc[physics["evaluation_mask"]].copy()
    physics["macro_region"] = np.where(
        physics["latitude"] >= 30.0, "northern_extratropics",
        np.where(physics["latitude"] <= -30.0, "southern_extratropics", "tropics"),
    )
    level_boot = physical_station_bootstrap(physics, "pressure_level", "pressure_hpa", replicates, seed)
    region_boot = physical_station_bootstrap(physics, "macro_region", "macro_region", replicates, seed)
    level_boot.to_csv(output / "physical_budget_level_bootstrap.csv", index=False)
    region_boot.to_csv(output / "physical_budget_region_bootstrap.csv", index=False)

    cross = cross_platform_table(config, cosmic_root)
    cross.to_csv(output / "cross_platform_comparison.csv", index=False)
    claims = pd.DataFrame([
        ["Wet refractivity dominates total ERA5 discrepancy", "supported", "physical_error_budget", "Results: physical sources"],
        ["HGB has a uniform space-time advantage", "not_supported", "space_time_station_CI_crosses_zero", "Results and Discussion"],
        ["Internal predictive calibration transfers across platforms", "not_supported", "RAPSODI_and_COSMIC_undercoverage", "Results and Discussion"],
        ["HGB correction transfers to COSMIC-2", "rejected", "archive_month_CI_above_zero", "Results: cross-platform"],
        ["A globally corrected product is justified", "rejected", "confirmatory_gate_and_external_transfer", "Discussion and Conclusions"],
        ["Application diagnostics justify duct or propagation claims", "not_supported", "mixed_gradient_and_integral_results", "Supplementary only"],
    ], columns=["claim", "status", "primary_evidence", "paper_location"])
    claims.to_csv(output / "final_claim_evidence_matrix.csv", index=False)

    expected = {"archives": 24, "profiles": 120908, "rows": 2176344, "evaluable_rows": 2044878, "height_levels": 18}
    observed = {
        "archives": int(len(archives)), "profiles": int(raw_cosmic[["profile_id", "time"]].drop_duplicates().shape[0]),
        "rows": int(len(raw_cosmic)), "evaluable_rows": int(len(cosmic)), "height_levels": int(level["height_km"].nunique()),
    }
    if observed != expected:
        raise ValueError(f"Frozen COSMIC-2 counts changed: observed={observed}, expected={expected}")
    maximum_reconstruction_error = float(max(level_boot["maximum_reconstruction_error"].max(), region_boot["maximum_reconstruction_error"].max()))
    if maximum_reconstruction_error > 1e-10:
        raise ValueError(f"Physical MSE reconstruction failed: {maximum_reconstruction_error}")
    output_files = sorted(output.glob("*.csv"))
    inputs = [cosmic_manifest_path, physics_run / "protocol_manifest.json", Path(config["frozen_model_run"]) / "model.joblib"]
    write_json(output / "final_statistics_manifest.json", {
        "complete": True, "smoke": False, "training_performed": False, "model_selection_performed": False,
        "paper_title": "Physical Sources and Generalization Limits of Statistical Correction for ERA5 Atmospheric Refractivity",
        "paper_positioning": "physical error sources, strict generalization limits, and cross-platform calibration transfer",
        "frozen_counts_expected": expected, "frozen_counts_observed": observed,
        "bootstrap_replicates": replicates, "bootstrap_seed": seed,
        "primary_bootstrap_units": {"internal": "station", "rapsodi": "launch_day", "cosmic2": "archive_month"},
        "cosmic2_coordinate": "MSL_geometric_height", "cosmic2_is_independent_truth": False,
        "dry_pressure_result_role": "coordinate_sensitivity",
        "maximum_mse_reconstruction_error": maximum_reconstruction_error,
        "input_hashes": {str(path.resolve()): sha256_file(path) for path in inputs},
        "output_hashes": {path.name: sha256_file(path) for path in output_files},
        "global_product_allowed": False,
    })
    print(f"Final publication diagnostics: {output.resolve()}")


if __name__ == "__main__":
    main()
