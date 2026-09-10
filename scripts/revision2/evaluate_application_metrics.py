from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

from _bootstrap import PROJECT_ROOT  # noqa: F401
from igra_forecast.revision2_baselines import LEVELS
from igra_forecast.revision2_data import write_json


def trapezoid_weights(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("Trapezoid coordinates must be strictly increasing")
    weights = np.zeros(len(x), dtype=float)
    delta = np.diff(x)
    weights[:-1] += delta / 2.0
    weights[1:] += delta / 2.0
    return weights


def covariance_lookup(frame: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], np.ndarray]:
    result = {}
    columns = {(left, right): f"cov_{int(LEVELS[left])}_{int(LEVELS[right])}" for left in range(6) for right in range(left, 6)}
    for row in frame.itertuples(index=False):
        covariance = np.zeros((6, 6), dtype=float)
        for (left, right), column in columns.items():
            value = float(getattr(row, column))
            covariance[left, right] = covariance[right, left] = value
        result[(str(row.station_id), pd.Timestamp(row.time))] = covariance
    return result


def metric_summary(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    error = frame[f"{prefix}_error"].to_numpy(float)
    return {
        "n": int(len(error)), "bias": float(error.mean()), "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
    }


def station_bootstrap(frame: pd.DataFrame, era5_column: str, model_column: str, quantity: str, replicates: int, rng: np.random.Generator) -> dict[str, object]:
    station = frame.assign(
        _era5_sq=np.square(frame[era5_column]), _model_sq=np.square(frame[model_column]),
    ).groupby("station_id").agg(era5_mse=("_era5_sq", "mean"), model_mse=("_model_sq", "mean"))
    delta = np.sqrt(station["model_mse"].to_numpy(float)) - np.sqrt(station["era5_mse"].to_numpy(float))
    samples = np.asarray([rng.choice(delta, len(delta), replace=True).mean() for _ in range(replicates)])
    return {
        "quantity": quantity, "stations": len(delta), "mean_station_rmse_difference": float(delta.mean()),
        "ci_lower": float(np.quantile(samples, 0.025)), "ci_upper": float(np.quantile(samples, 0.975)),
        "replicates": int(replicates),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pressure/height integrals and adjacent-level N/M gradients.")
    parser.add_argument("--config", default="configs/revision2/confirmatory.yaml")
    parser.add_argument("--protocol", default="space_time_holdout", choices=["spatial_station_disjoint", "temporal_holdout", "space_time_holdout"])
    parser.add_argument("--model-run", help="Structured probability run; defaults to run 42.")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    primary = config["evaluation"]["primary_probability_variant"]
    run = Path(args.model_run or Path(config["output_root"]) / args.protocol / f"{primary}_run42")
    run_manifest = json.loads((run / "protocol_manifest.json").read_text(encoding="utf-8"))
    rows = pd.read_parquet(run / "predictions.parquet")
    rows = rows.loc[rows["evaluation_mask"]].copy()
    covariance = covariance_lookup(pd.read_parquet(run / "profile_covariance.parquet"))
    level_index = {int(level): index for index, level in enumerate(LEVELS)}
    minimum = int(config["application"]["minimum_common_levels"])
    profile_rows = []
    gradient_rows = []
    for (station, time), group in rows.groupby(["station_id", "time"], observed=True, sort=False):
        group = group.sort_values("pressure_hpa", ascending=False)
        if len(group) < minimum:
            continue
        indices = np.asarray([level_index[int(value)] for value in group["pressure_hpa"]], dtype=int)
        cov = covariance[(str(station), pd.Timestamp(time))][np.ix_(indices, indices)]
        pressure_coordinate = -np.log(group["pressure_hpa"].to_numpy(float))
        pressure_weights = trapezoid_weights(pressure_coordinate)
        era5_n = group["era5_n"].to_numpy(float)
        observed_n = era5_n + group["residual_total"].to_numpy(float)
        corrected_n = era5_n + group["prediction_total"].to_numpy(float)
        era5_z = group["era5_height_m"].to_numpy(float) / 1000.0
        observed_z = group["igra_height_m"].to_numpy(float) / 1000.0
        height_weights = trapezoid_weights(era5_z)
        observed_height_weights = trapezoid_weights(observed_z)
        pressure_truth = float(pressure_weights @ observed_n)
        pressure_era5 = float(pressure_weights @ era5_n)
        pressure_model = float(pressure_weights @ corrected_n)
        height_truth = float(observed_height_weights @ observed_n)
        height_era5 = float(height_weights @ era5_n)
        height_model = float(height_weights @ corrected_n)
        pressure_std = float(np.sqrt(pressure_weights @ cov @ pressure_weights))
        height_std = float(np.sqrt(height_weights @ cov @ height_weights))
        profile_rows.append({
            "station_id": station, "time": time, "valid_levels": len(group),
            "pressure_integral_truth": pressure_truth, "pressure_integral_era5": pressure_era5, "pressure_integral_model": pressure_model,
            "pressure_integral_era5_error": pressure_era5 - pressure_truth, "pressure_integral_model_error": pressure_model - pressure_truth,
            "pressure_integral_predictive_std": pressure_std,
            "height_integral_truth_n_units_km": height_truth, "height_integral_era5_n_units_km": height_era5,
            "height_integral_model_n_units_km": height_model,
            "height_integral_era5_error": height_era5 - height_truth, "height_integral_model_error": height_model - height_truth,
            "height_integral_predictive_std": height_std,
        })
        group_by_level = group.set_index("pressure_hpa")
        for lower, upper in zip(LEVELS[:-1], LEVELS[1:]):
            if int(lower) not in group_by_level.index or int(upper) not in group_by_level.index:
                continue
            pair = group_by_level.loc[[int(lower), int(upper)]]
            era5_pair_z = pair["era5_height_m"].to_numpy(float) / 1000.0
            observed_pair_z = pair["igra_height_m"].to_numpy(float) / 1000.0
            dz_model = float(era5_pair_z[1] - era5_pair_z[0])
            dz_truth = float(observed_pair_z[1] - observed_pair_z[0])
            if dz_model <= 0 or dz_truth <= 0:
                continue
            era5_pair_n = pair["era5_n"].to_numpy(float)
            observed_pair_n = era5_pair_n + pair["residual_total"].to_numpy(float)
            corrected_pair_n = era5_pair_n + pair["prediction_total"].to_numpy(float)
            gradient_weights = np.asarray([-1.0 / dz_model, 1.0 / dz_model])
            pair_indices = np.asarray([level_index[int(lower)], level_index[int(upper)]])
            pair_cov = covariance[(str(station), pd.Timestamp(time))][np.ix_(pair_indices, pair_indices)]
            gradient_std = float(np.sqrt(gradient_weights @ pair_cov @ gradient_weights))
            for quantity, era5_values, observed_values, corrected_values in [
                ("dN_dz", era5_pair_n, observed_pair_n, corrected_pair_n),
                ("dM_dz", era5_pair_n + 0.157 * era5_pair_z * 1000.0, observed_pair_n + 0.157 * observed_pair_z * 1000.0, corrected_pair_n + 0.157 * era5_pair_z * 1000.0),
            ]:
                truth = float((observed_values[1] - observed_values[0]) / dz_truth)
                era5_value = float((era5_values[1] - era5_values[0]) / dz_model)
                model_value = float((corrected_values[1] - corrected_values[0]) / dz_model)
                gradient_rows.append({
                    "station_id": station, "time": time, "level_pair": f"{int(lower)}-{int(upper)}", "quantity": quantity,
                    "truth": truth, "era5": era5_value, "model": model_value,
                    "era5_error": era5_value - truth, "model_error": model_value - truth,
                    "predictive_std": gradient_std,
                })
    profiles = pd.DataFrame(profile_rows)
    gradients = pd.DataFrame(gradient_rows)
    output = Path(config["paper_output_root"]) / ("smoke" if run_manifest.get("smoke") else "") / "application" / args.protocol
    output.mkdir(parents=True, exist_ok=True)
    profiles.to_parquet(output / "profile_application_metrics.parquet", index=False, compression="zstd")
    gradients.to_parquet(output / "gradient_application_metrics.parquet", index=False, compression="zstd")
    summary_rows = []
    bootstrap_rows = []
    replicates = int(config["evaluation"]["bootstrap_replicates"])
    rng = np.random.default_rng(42)
    for prefix in ["pressure_integral", "height_integral"]:
        era5 = metric_summary(profiles, f"{prefix}_era5")
        model = metric_summary(profiles, f"{prefix}_model")
        std = profiles[f"{prefix}_predictive_std"].to_numpy(float)
        error = profiles[f"{prefix}_model_error"].to_numpy(float)
        summary_rows.append({
            "quantity": prefix, **{f"era5_{key}": value for key, value in era5.items()},
            **{f"model_{key}": value for key, value in model.items()},
       "model_minus_era5_rmse": model["rmse"] - era5["rmse"],
            "marginal_coverage_90": float(np.mean(np.abs(error) <= norm.ppf(0.95) * std)),
        })
        bootstrap_rows.append(station_bootstrap(
            profiles, f"{prefix}_era5_error", f"{prefix}_model_error", prefix, replicates, rng
        ))
    for (pair, quantity), group in gradients.groupby(["level_pair", "quantity"], observed=True):
        era5 = metric_summary(group, "era5")
        model = metric_summary(group, "model")
        summary_rows.append({
            "quantity": quantity, "level_pair": pair,
            **{f"era5_{key}": value for key, value in era5.items()}, **{f"model_{key}": value for key, value in model.items()},
            "model_minus_era5_rmse": model["rmse"] - era5["rmse"],
            "marginal_coverage_90": float(np.mean(np.abs(group["model_error"]) <= norm.ppf(0.95) * group["predictive_std"])),
        })
        bootstrap_rows.append(station_bootstrap(
            group, "era5_error", "model_error", f"{quantity}_{pair}", replicates, rng
        ))
    pd.DataFrame(summary_rows).to_csv(output / "application_summary.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(output / "application_station_bootstrap.csv", index=False)
    write_json(output / "application_manifest.json", {
        "protocol": args.protocol, "model_run": str(run.resolve()), "minimum_common_levels": minimum,
        "height_uncertainty_included": False,
        "interpretation": "height-based predictive intervals are conditional on ERA5 geopotential height; no duct-height or propagation-loss claim",
        "profiles": len(profiles), "gradient_rows": len(gradients),
    })
    print(f"Application metrics: {output.resolve()}")


if __name__ == "__main__":
    main()
