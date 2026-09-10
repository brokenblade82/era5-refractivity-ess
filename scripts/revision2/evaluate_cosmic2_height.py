from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy.stats import norm
from tqdm import tqdm

from _bootstrap import PROJECT_ROOT  # noqa: F401
from collocate_external_profiles import open_era5_variable, preload_interpolation_window, spatiotemporal_points
from igra_forecast.revision2_baselines import LEVELS, rows_to_profiles
from igra_forecast.revision2_confirmatory import predict_confirmatory_probability_hgb
from igra_forecast.revision2_data import (
    G0,
    macro_region,
    refractivity_components,
    require_era5_audit,
    resolve_era5_root,
    sha256_file,
    stable_fingerprint,
    write_json,
)
from igra_forecast.revision2_publication import (
    cluster_bootstrap_mean,
    geometric_height_evaluation_mask,
    interpolate_no_extrapolation,
    normal_crps,
    paired_rmse_units,
    propagate_gaussian_to_height,
    validate_geometric_height_schema,
)


def read_config(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def static_points(field: xr.DataArray, frame: pd.DataFrame) -> np.ndarray:
    field = field.squeeze(drop=True)
    latitudes = np.asarray(field["latitude"], dtype=float)
    longitudes = np.asarray(field["longitude"], dtype=float)
    values = np.asarray(field.values, dtype=float)
    target_lat = frame["latitude"].to_numpy(float)
    target_lon = np.mod(frame["longitude"].to_numpy(float), 360.0)
    lat_step = abs(latitudes[1] - latitudes[0])
    lon_step = abs(longitudes[1] - longitudes[0])
    lat_pos = (latitudes[0] - target_lat) / lat_step if latitudes[0] > latitudes[-1] else (target_lat - latitudes[0]) / lat_step
    lat_pos = np.clip(lat_pos, 0, len(latitudes) - 1)
    lon_pos = np.mod((target_lon - longitudes[0]) / lon_step, len(longitudes))
    lat0 = np.floor(lat_pos).astype(int)
    lat1 = np.minimum(lat0 + 1, len(latitudes) - 1)
    lon0 = np.floor(lon_pos).astype(int) % len(longitudes)
    lon1 = (lon0 + 1) % len(longitudes)
    wy = lat_pos - lat0
    wx = lon_pos - np.floor(lon_pos)
    return (
        (1 - wy) * ((1 - wx) * values[lat0, lon0] + wx * values[lat0, lon1])
        + wy * ((1 - wx) * values[lat1, lon0] + wx * values[lat1, lon1])
    )


def collocate_pressure_profiles(
    target: pd.DataFrame,
    arrays: dict[str, xr.DataArray],
    pressure_levels: np.ndarray,
) -> dict[str, np.ndarray]:
    expanded = target.loc[target.index.repeat(len(pressure_levels))].reset_index(drop=True)
    expanded["pressure_hpa"] = np.tile(pressure_levels, len(target))
    shape = (len(target), len(pressure_levels))
    temperature = spatiotemporal_points(arrays["temperature"], expanded, True).reshape(shape)
    humidity = spatiotemporal_points(arrays["specific_humidity"], expanded, True).reshape(shape)
    geopotential = spatiotemporal_points(arrays["geopotential"], expanded, True).reshape(shape)
    dry, wet, total, _ = refractivity_components(
        expanded["pressure_hpa"].to_numpy(float), temperature.reshape(-1), humidity.reshape(-1)
    )
    surface_pressure = spatiotemporal_points(arrays["surface_pressure"], target, False)
    return {
        "temperature": temperature,
        "humidity": humidity,
        "height": geopotential / G0,
        "n_dry": dry.reshape(shape),
        "n_wet": wet.reshape(shape),
        "n_total": total.reshape(shape),
        "surface_pressure": surface_pressure,
    }


def preload_month_arrays(
    arrays: dict[str, xr.DataArray],
    target: pd.DataFrame,
    pressure_levels: np.ndarray,
) -> tuple[dict[str, xr.DataArray], dict[str, object]]:
    """Load the compact time/spatial window once per sampled COSMIC month.

    Direct netCDF fancy indexing turns a few thousand globally distributed
    points into many slow random reads.  The pre-registered COSMIC sample spans
    only one day per month, so loading the required global time slices is both
    faster and memory bounded.
    """
    expanded = target.loc[target.index.repeat(len(pressure_levels))].reset_index(drop=True)
    expanded["pressure_hpa"] = np.tile(pressure_levels, len(target))
    loaded: dict[str, xr.DataArray] = {}
    audit: dict[str, object] = {}
    for logical in ["temperature", "specific_humidity", "geopotential"]:
        loaded[logical], audit[logical] = preload_interpolation_window(
            arrays[logical], expanded, True, max_values=60_000_000
        )
    loaded["surface_pressure"], audit["surface_pressure"] = preload_interpolation_window(
        arrays["surface_pressure"], target, False, max_values=60_000_000
    )
    return loaded, audit


def predict_six_level_profiles(
    target: pd.DataFrame,
    collocated: dict[str, np.ndarray],
    all_levels: np.ndarray,
    model: dict[str, object],
) -> tuple[object, np.ndarray]:
    index = np.asarray([int(np.flatnonzero(all_levels == level)[0]) for level in LEVELS], dtype=int)
    n_target = len(target)
    synthetic = np.asarray([f"H{number:09d}" for number in range(n_target)])
    frame = pd.DataFrame({
        "station_id": np.repeat(synthetic, len(LEVELS)),
        "time": np.repeat(pd.to_datetime(target["time"]).to_numpy(), len(LEVELS)),
        "latitude": np.repeat(target["latitude"].to_numpy(float), len(LEVELS)),
        "longitude": np.repeat(target["longitude"].to_numpy(float), len(LEVELS)),
        "station_elevation_m": np.repeat(target["surface_height_m"].to_numpy(float), len(LEVELS)),
        "pressure_hpa": np.tile(LEVELS, n_target),
        "era5_temperature_k": collocated["temperature"][:, index].reshape(-1),
        "era5_specific_humidity": collocated["humidity"][:, index].reshape(-1),
        "era5_height_m": collocated["height"][:, index].reshape(-1),
        "era5_n_dry": collocated["n_dry"][:, index].reshape(-1),
        "era5_n_wet": collocated["n_wet"][:, index].reshape(-1),
        "era5_n": collocated["n_total"][:, index].reshape(-1),
        "era5_surface_pressure_pa": np.repeat(collocated["surface_pressure"], len(LEVELS)),
        "below_ground_level": np.repeat(collocated["surface_pressure"], len(LEVELS)) < np.tile(LEVELS, n_target) * 100.0,
        "level_mask": 1,
        "residual_dry": 0.0,
        "residual_wet": 0.0,
        "residual_total": 0.0,
    })
    arrays = rows_to_profiles(frame)
    prediction = predict_confirmatory_probability_hgb(model, arrays)
    return prediction, index


def evaluate_chunk(
    target: pd.DataFrame,
    era5_arrays: dict[str, xr.DataArray],
    static_z: xr.DataArray,
    static_lsm: xr.DataArray,
    all_levels: np.ndarray,
    model: dict[str, object],
) -> pd.DataFrame:
    target = target.copy().reset_index(drop=True)
    target["surface_height_m"] = static_points(static_z, target) / G0
    target["land_sea"] = np.where(static_points(static_lsm, target) >= 0.5, "land", "sea")
    collocated = collocate_pressure_profiles(target, era5_arrays, all_levels)
    prediction, correction_index = predict_six_level_profiles(target, collocated, all_levels, model)
    outputs: list[dict[str, object]] = []
    for row_index, row in target.iterrows():
        target_height = float(row["height_m"])
        above_surface = all_levels * 100.0 <= collocated["surface_pressure"][row_index]
        baseline_height = collocated["height"][row_index]
        baseline_n = collocated["n_total"][row_index]
        baseline_valid = above_surface & np.isfinite(baseline_height) & np.isfinite(baseline_n) & (baseline_n > 0)
        era5_n = interpolate_no_extrapolation(
            baseline_height[baseline_valid], baseline_n[baseline_valid], np.asarray([target_height]), log_values=True
        )[0]

        correction_height = collocated["height"][row_index, correction_index]
        correction_valid = (
            LEVELS * 100.0 <= collocated["surface_pressure"][row_index]
        ) & np.isfinite(correction_height)
        correction = np.nan
        variance = np.nan
        in_correction_support = False
        if correction_valid.sum() >= 2:
            selected = np.flatnonzero(correction_valid)
            try:
                mean, var, valid = propagate_gaussian_to_height(
                    correction_height[selected],
                    prediction.mean_total[row_index, selected],
                    prediction.covariance_total[row_index][np.ix_(selected, selected)],
                    np.asarray([target_height]),
                )
                correction, variance, in_correction_support = float(mean[0]), float(var[0]), bool(valid[0])
            except ValueError:
                in_correction_support = False
        below_surface = target_height <= float(row["surface_height_m"])
        evaluation_mask = bool(geometric_height_evaluation_mask(
            np.asarray([row["observed_n"]]), np.asarray([target_height]),
            np.asarray([row["surface_height_m"]]), np.asarray([era5_n]),
            np.asarray([correction]), np.asarray([variance]),
            np.asarray([row["height_mask"]]), np.asarray([in_correction_support]),
        )[0])
        model_n = era5_n + correction if evaluation_mask else np.nan
        outputs.append({
            **row.to_dict(),
            "era5_n_height": era5_n,
            "prediction_correction": correction,
            "prediction_n_height": model_n,
            "prediction_std_height": np.sqrt(variance) if np.isfinite(variance) and variance >= 0 else np.nan,
            "below_surface": below_surface,
            "within_era5_height_support": bool(np.isfinite(era5_n)),
            "within_correction_height_support": in_correction_support,
            "evaluation_mask": evaluation_mask,
        })
    return pd.DataFrame(outputs)


def point_table(valid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, column in [("era5", "era5_error"), ("hgb", "model_error")]:
        values = valid[column].to_numpy(float)
        rows.append({"model": model, "n": len(values), "bias": values.mean(), "mae": np.abs(values).mean(), "rmse": np.sqrt(np.square(values).mean())})
    return pd.DataFrame(rows)


def grouped_table(valid: pd.DataFrame, grouping: str) -> pd.DataFrame:
    rows = []
    for group, frame in valid.groupby(grouping, observed=True):
        era5 = frame["era5_error"].to_numpy(float)
        model = frame["model_error"].to_numpy(float)
        rows.append({
            "grouping": grouping, "group": group, "n": len(frame),
            "era5_bias": era5.mean(), "model_bias": model.mean(),
            "era5_mae": np.abs(era5).mean(), "model_mae": np.abs(model).mean(),
            "era5_rmse": np.sqrt(np.square(era5).mean()), "model_rmse": np.sqrt(np.square(model).mean()),
            "model_minus_era5_rmse": np.sqrt(np.square(model).mean()) - np.sqrt(np.square(era5).mean()),
        })
    return pd.DataFrame(rows)


def bootstrap_table(valid: pd.DataFrame, unit: list[str], method: str, replicates: int, seed: int) -> pd.DataFrame:
    units = paired_rmse_units(valid, unit)
    payload = cluster_bootstrap_mean(units["rmse_difference"].to_numpy(float), replicates, seed)
    return pd.DataFrame([{"comparison": "hgb_minus_era5_rmse", "method": method, **payload}])


def leave_one_period_out(valid: pd.DataFrame, period_column: str) -> pd.DataFrame:
    """Quantify how much each sampled period influences the paired RMSE result."""
    rows: list[dict[str, object]] = []
    for omitted in sorted(valid[period_column].dropna().astype(str).unique()):
        selected = valid.loc[valid[period_column].astype(str) != omitted]
        era5 = selected["era5_error"].to_numpy(float)
        model = selected["model_error"].to_numpy(float)
        rows.append({
            f"omitted_{period_column}": omitted,
            "remaining_rows": int(len(selected)),
            "remaining_profiles": int(selected[["profile_id", "time"]].drop_duplicates().shape[0]),
            "era5_rmse": float(np.sqrt(np.square(era5).mean())) if len(era5) else np.nan,
            "hgb_rmse": float(np.sqrt(np.square(model).mean())) if len(model) else np.nan,
            "hgb_minus_era5_rmse": (
                float(np.sqrt(np.square(model).mean()) - np.sqrt(np.square(era5).mean()))
                if len(model) else np.nan
            ),
        })
    return pd.DataFrame(rows)


def write_metrics(frame: pd.DataFrame, output: Path, settings: dict[str, object], seed: int) -> dict[str, object]:
    valid = frame.loc[frame["evaluation_mask"]].copy()
    valid["era5_error"] = valid["era5_n_height"] - valid["observed_n"]
    valid["model_error"] = valid["prediction_n_height"] - valid["observed_n"]
    valid["height_km"] = valid["height_m"] / 1000.0
    valid["macro_region"] = valid["latitude"].map(macro_region)
    valid["year"] = pd.to_datetime(valid["time"], utc=True).dt.year.astype(str)
    valid["month"] = pd.to_datetime(valid["time"], utc=True).dt.strftime("%Y-%m")
    valid["date"] = pd.to_datetime(valid["time"], utc=True).dt.strftime("%Y-%m-%d")
    valid["height_band"] = pd.cut(
        valid["height_km"], [0.5, 2.0, 5.0, 9.0], labels=["0.5-2", "2-5", "5-9"], include_lowest=True
    )
    point = point_table(valid)
    point.to_csv(output / "point_metrics.csv", index=False)
    grouped_table(valid, "height_band").to_csv(output / "height_band_metrics.csv", index=False)
    pd.concat([grouped_table(valid, "macro_region"), grouped_table(valid, "land_sea"), grouped_table(valid, "year")], ignore_index=True).to_csv(output / "regional_metrics.csv", index=False)
    grouped_table(valid, "month").to_csv(output / "monthly_metrics.csv", index=False)
    replicates = int(settings["bootstrap_replicates"])
    bootstrap_table(valid, ["profile_id", "time"], "external_profile_cluster", replicates, seed).to_csv(output / "profile_bootstrap.csv", index=False)
    bootstrap_table(valid, ["date"], "sampling_day_block", replicates, seed).to_csv(output / "day_bootstrap.csv", index=False)
    bootstrap_table(valid, ["month"], "month_block", replicates, seed).to_csv(output / "month_bootstrap.csv", index=False)
    leave_one_period_out(valid, "month").to_csv(output / "leave_one_month_out.csv", index=False)

    error = valid["model_error"].to_numpy(float)
    std = valid["prediction_std_height"].to_numpy(float)
    ok = np.isfinite(error) & np.isfinite(std) & (std > 0)
    nll = 0.5 * np.square(error[ok] / std[ok]) + np.log(std[ok]) + 0.5 * np.log(2 * np.pi)
    alpha = 0.10
    z90 = norm.ppf(1 - alpha / 2)
    width = 2 * z90 * std[ok]
    interval_score = width + 2 / alpha * (np.abs(error[ok]) - z90 * std[ok]) * (np.abs(error[ok]) > z90 * std[ok])
    probability = pd.DataFrame([{
        "coordinate": "MSL_geometric_height", "n": int(ok.sum()), "gaussian_nll": nll.mean(),
        "crps": normal_crps(error[ok], std[ok]).mean(),
        "central_interval_score_90": interval_score.mean(),
        "marginal_coverage_90": (np.abs(error[ok]) <= z90 * std[ok]).mean(),
        "mean_interval_width_90": width.mean(),
    }])
    probability.to_csv(output / "probabilistic_metrics.csv", index=False)
    coverage = []
    for nominal in settings["confidence_levels"]:
        z = norm.ppf((1 + float(nominal)) / 2)
        coverage.append({"nominal_coverage": nominal, "empirical_coverage": (np.abs(error[ok]) <= z * std[ok]).mean(), "n": int(ok.sum())})
    pd.DataFrame(coverage).to_csv(output / "coverage_curve.csv", index=False)
    return {"rows": len(frame), "evaluable_rows": len(valid), "profiles": int(valid["profile_id"].nunique()), "point": point.to_dict("records"), "probability": probability.iloc[0].to_dict()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen six-level HGB on COSMIC-2 geometric-height refractivity.")
    parser.add_argument("--config", default="configs/revision2/publication.yaml")
    parser.add_argument("--input")
    parser.add_argument("--prepare-manifest")
    parser.add_argument("--model-run")
    parser.add_argument("--era5-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--chunk-profiles", type=int, default=1000)
    parser.add_argument("--max-profiles", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    publication = read_config(args.config)
    settings = publication["cosmic2_height"]
    data_config = read_config(publication["data_config"])
    input_path = Path(args.input or settings["prepared_profiles"])
    prepare_manifest_path = Path(args.prepare_manifest or settings["preparation_manifest"])
    if not input_path.is_file() or not prepare_manifest_path.is_file():
        raise FileNotFoundError("Run prepare_cosmic2_height.py before geometric-height evaluation")
    prepare_manifest = json.loads(prepare_manifest_path.read_text(encoding="utf-8"))
    input_hash = sha256_file(input_path)
    if prepare_manifest.get("output_sha256") != input_hash or prepare_manifest.get("dry_pressure_used") is not False:
        raise ValueError("COSMIC-2 height input does not match its audited geometric-height manifest")
    frame = pd.read_parquet(input_path)
    validate_geometric_height_schema(frame)
    if args.max_profiles is not None:
        keep = frame[["profile_id", "time"]].drop_duplicates().head(int(args.max_profiles))
        frame = frame.merge(keep, on=["profile_id", "time"], how="inner", validate="many_to_one")
    frame["time"] = pd.to_datetime(frame["time"])
    frame["period"] = frame["time"].dt.to_period("M")

    model_run = Path(args.model_run or publication["frozen_model_run"])
    model_path = model_run / "model.joblib"
    protocol_manifest_path = model_run / "protocol_manifest.json"
    protocol_manifest = json.loads(protocol_manifest_path.read_text(encoding="utf-8"))
    if not protocol_manifest.get("complete") or protocol_manifest.get("smoke"):
        raise ValueError("Geometric-height evaluation requires a complete frozen formal model")
    model = joblib.load(model_path)
    model_hash = sha256_file(model_path)
    all_levels = np.asarray(settings["era5_pressure_levels_hpa"], dtype=int)
    smoke = args.max_profiles is not None
    output = Path(args.output_dir or Path(publication["output_root"]) / ("smoke" if smoke else "") / "cosmic2_height")
    predictions_root = output / "predictions"
    output.mkdir(parents=True, exist_ok=True)
    predictions_root.mkdir(parents=True, exist_ok=True)
    run_fingerprint = stable_fingerprint({
        "input_sha256": input_hash, "model_sha256": model_hash, "levels": all_levels.tolist(),
        "coordinate": "MSL_geometric_height", "dry_pressure_used": False,
    })

    era5_root = resolve_era5_root(data_config, args.era5_root)
    audit_path = Path(data_config["paths"]["manifests"]) / "era5_archive_manifest.json"
    periods = sorted(frame["period"].unique())
    require_era5_audit(data_config, audit_path, era5_root, [period.strftime("%Y%m") for period in periods], require_full_study=not smoke)
    archive_manifest = json.loads(audit_path.read_text(encoding="utf-8"))
    static_path = Path(archive_manifest["static"]["path"])
    with xr.open_dataset(static_path) as static_dataset:
        static_z = static_dataset["z"].squeeze(drop=True).load()
        static_lsm = static_dataset["lsm"].squeeze(drop=True).load()

    month_files: list[Path] = []
    for period in tqdm(periods, desc="COSMIC-2 height months"):
        yyyymm = period.strftime("%Y%m")
        month_file = predictions_root / f"predictions_{yyyymm}.parquet"
        month_meta = predictions_root / f"predictions_{yyyymm}.json"
        if args.resume and month_file.is_file() and month_meta.is_file():
            meta = json.loads(month_meta.read_text(encoding="utf-8"))
            if meta.get("run_fingerprint") == run_fingerprint and meta.get("sha256") == sha256_file(month_file):
                month_files.append(month_file)
                continue
        selected = frame.loc[frame["period"] == period].drop(columns="period").copy()
        keys = selected[["profile_id", "time"]].drop_duplicates().reset_index(drop=True)
        source_arrays = {logical: open_era5_variable(era5_root, period, data_config, logical) for logical in ["temperature", "specific_humidity", "geopotential", "surface_pressure"]}
        arrays, preload_audit = preload_month_arrays(source_arrays, selected, all_levels)
        chunks: list[pd.DataFrame] = []
        for start in tqdm(range(0, len(keys), int(args.chunk_profiles)), desc=yyyymm, leave=False):
            chunk_keys = keys.iloc[start:start + int(args.chunk_profiles)]
            target = selected.merge(chunk_keys, on=["profile_id", "time"], how="inner", validate="many_to_one")
            chunks.append(evaluate_chunk(target, arrays, static_z, static_lsm, all_levels, model))
        for data_array in source_arrays.values():
            data_array.close()
        result = pd.concat(chunks, ignore_index=True)
        temporary = month_file.with_suffix(".parquet.tmp")
        result.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, month_file)
        write_json(month_meta, {"run_fingerprint": run_fingerprint, "rows": len(result), "sha256": sha256_file(month_file), "preload_audit": preload_audit})
        month_files.append(month_file)

    predictions = pd.concat([pd.read_parquet(path) for path in month_files], ignore_index=True)
    audit = write_metrics(predictions, output, settings, int(protocol_manifest["seed"]))
    old_gate_path = Path(publication["confirmatory_paper_root"]) / "external" / "cosmic2" / "external_gate.csv"
    sensitivity = []
    if old_gate_path.is_file():
        old = pd.read_csv(old_gate_path).iloc[0]
        sensitivity.append({"coordinate": "dry_pressure_sensitivity", "era5_rmse": old["era5_rmse"], "model_rmse": old["model_rmse"], "model_minus_era5_rmse": old["model_rmse"] - old["era5_rmse"]})
    point = pd.read_csv(output / "point_metrics.csv").set_index("model")
    sensitivity.append({"coordinate": "MSL_geometric_height_primary", "era5_rmse": point.loc["era5", "rmse"], "model_rmse": point.loc["hgb", "rmse"], "model_minus_era5_rmse": point.loc["hgb", "rmse"] - point.loc["era5", "rmse"]})
    pd.DataFrame(sensitivity).to_csv(output / "coordinate_sensitivity.csv", index=False)
    write_json(output / "manifest.json", {
        "complete": True, "smoke": smoke, "coordinate": "MSL_geometric_height", "dry_pressure_used": False,
        "prepared_input": str(input_path.resolve()), "prepared_input_sha256": input_hash,
        "preparation_manifest": str(prepare_manifest_path.resolve()),
        "frozen_model_run": str(model_run.resolve()), "frozen_model_sha256": model_hash,
        "era5_archive_fingerprint": archive_manifest["archive_fingerprint"],
        "height_levels_m": prepare_manifest["height_levels_m"], "era5_pressure_levels_hpa": all_levels.tolist(),
        "correction_pressure_levels_hpa": list(map(int, LEVELS)), "run_fingerprint": run_fingerprint,
        "bootstrap_units": ["external profile", "sampling day", "month"], "audit": audit,
        "external_profiles_never_used_for_training_selection_or_calibration": True,
        "interpretation": publication["interpretation"]["cosmic2_primary"],
    })
    print(f"COSMIC-2 geometric-height evaluation: {output.resolve()}")


if __name__ == "__main__":
    main()
