from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

if not os.environ.get("LOKY_MAX_CPU_COUNT", "").isdigit() or int(os.environ.get("LOKY_MAX_CPU_COUNT", "0")) < 1:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)

import joblib
import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import torch
import yaml

from _bootstrap import PROJECT_ROOT  # noqa: F401
from igra_forecast.revision2_baselines import (
    ProfilePrediction,
    calibrate_standard_deviation,
    choose_rbf_anchors,
    fit_component_regressors,
    fit_probabilistic_hgb,
    fit_profile_network,
    metric_tables,
    predict_component_regressors,
    predict_probabilistic_hgb,
    predict_profile_network,
    prediction_to_rows,
    rbf_features,
    rows_to_profiles,
)
from igra_forecast.revision2_data import load_data_config, require_collocation_manifest, sha256_file, stable_fingerprint, write_json
from igra_forecast.revision2_confirmatory import (
    PROBABILITY_VARIANTS,
    covariance_frame,
    fit_confirmatory_probability_hgb,
    predict_confirmatory_probability_hgb,
    validate_covariance_masks,
)


REQUIRED_COLUMNS = [
    "station_id", "time", "latitude", "longitude", "station_elevation_m", "pressure_hpa", "level_mask", "below_ground_level",
    "igra_height_m",
    "era5_temperature_k", "era5_specific_humidity", "era5_height_m", "era5_surface_pressure_pa",
    "era5_n_dry", "era5_n_wet", "era5_n", "residual_dry", "residual_wet", "residual_total",
]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def phase_rows(
    dataset: pads.Dataset,
    split: pd.DataFrame,
    license_row: pd.Series,
    max_times: int | None,
    block: str | None,
) -> pd.DataFrame:
    start = pd.Timestamp(license_row["time_start_utc"], tz="UTC")
    end = pd.Timestamp(license_row["time_end_utc_exclusive"], tz="UTC")
    months = pd.period_range(start.tz_localize(None), (end - pd.Timedelta(seconds=1)).tz_localize(None), freq="M")
    frames = []
    for period in months:
        table = dataset.to_table(columns=REQUIRED_COLUMNS, filter=(pads.field("year") == period.year) & (pads.field("month") == period.month))
        part = table.to_pandas()
        time = pd.DatetimeIndex(pd.to_datetime(part["time"], utc=True))
        part = part.loc[(time >= start) & (time < end)]
        if not part.empty:
            frames.append(part)
    if not frames:
        raise ValueError(f"Protocol phase has no data: {license_row.to_dict()}")
    frame = pd.concat(frames, ignore_index=True)
    role = str(license_row["station_license"])
    if role.startswith("block_"):
        if block is None:
            raise ValueError("Spatial leave-block-out protocol requires --block latN_lonM")
        try:
            lat_bin = int(block.split("_")[0].removeprefix("lat"))
            lon_bin = int(block.split("_")[1].removeprefix("lon"))
        except Exception as exc:
            raise ValueError(f"Invalid block name: {block}") from exc
        actual_lat_bin = np.clip(np.floor((frame["latitude"] + 90.0) / 30.0).astype(int), 0, 5)
        actual_lon_bin = np.clip(np.floor(np.mod(frame["longitude"], 360.0) / 60.0).astype(int), 0, 5)
        in_block = (actual_lat_bin == lat_bin) & (actual_lon_bin == lon_bin)
        if role == "block_in":
            frame = frame.loc[in_block]
        else:
            split_role = {"block_out": "train", "block_out_early_val": "early_val", "block_out_calibration": "calibration"}[role]
            allowed = set(split.loc[split["split"] == split_role, "station_id"].astype(str))
            frame = frame.loc[~in_block & frame["station_id"].astype(str).isin(allowed)]
    else:
        allowed = set(split.loc[split["split"] == role, "station_id"].astype(str))
        frame = frame.loc[frame["station_id"].astype(str).isin(allowed)]
    if max_times is not None:
        selected = sorted(pd.DatetimeIndex(frame["time"].unique()))[: int(max_times)]
        frame = frame.loc[frame["time"].isin(selected)]
    if frame.empty:
        raise ValueError(f"Protocol phase became empty after station/time licensing: {license_row.to_dict()}")
    return frame.reset_index(drop=True)


def load_protocol_data(
    input_path: str | Path,
    split_path: str | Path,
    license_path: str | Path,
    protocol: str,
    max_times: int | None,
    block: str | None,
    block_evaluation_period: str = "concurrent",
) -> dict[str, pd.DataFrame]:
    dataset = pads.dataset(input_path, format="parquet", partitioning="hive")
    split = pd.read_csv(split_path, dtype={"station_id": str})
    licenses = pd.read_csv(license_path)
    selected = licenses.loc[licenses["protocol"] == protocol]
    if protocol == "spatial_leave_block_out":
        keep_evaluation = f"evaluation_{block_evaluation_period}"
        keep_calibration = f"calibration_{block_evaluation_period}"
        phase_text = selected["phase"].astype(str)
        selected = selected.loc[
            (~phase_text.str.startswith("evaluation_") & ~phase_text.str.startswith("calibration_"))
            | selected["phase"].isin([keep_evaluation, keep_calibration])
        ]
    if selected.empty:
        raise ValueError(f"Unknown protocol: {protocol}")
    result = {}
    for _, row in selected.iterrows():
        phase = str(row["phase"])
        if phase.startswith("evaluation_"):
            phase = "evaluation"
        elif phase.startswith("calibration_"):
            phase = "calibration"
        if phase in result:
            continue
        result[phase] = phase_rows(dataset, split, row, max_times, block)
    required = {"train", "early_stopping", "calibration", "evaluation"}
    if set(result) != required:
        raise ValueError(f"Protocol phases mismatch: expected {sorted(required)}, found {sorted(result)}")
    key_sets = {
        phase: set(zip(frame["station_id"].astype(str), pd.to_datetime(frame["time"], utc=True)))
        for phase, frame in result.items()
    }
    phases = sorted(key_sets)
    overlaps = {
        f"{left}__{right}": len(key_sets[left] & key_sets[right])
        for index, left in enumerate(phases) for right in phases[index + 1:]
        if key_sets[left] & key_sets[right]
    }
    if overlaps:
        raise ValueError(f"Protocol phases share station-time profiles: {overlaps}")
    return result


def load_smoke_data(input_path: str | Path, max_times: int = 8) -> dict[str, pd.DataFrame]:
    dataset = pads.dataset(input_path, format="parquet", partitioning="hive")
    frame = dataset.to_table(columns=REQUIRED_COLUMNS).to_pandas()
    times = sorted(pd.DatetimeIndex(frame["time"].unique()))[: max(8, int(max_times))]
    if len(times) < 8:
        raise ValueError("CUDA smoke training needs at least eight distinct times")
    groups = {"train": times[:4], "early_stopping": times[4:5], "calibration": times[5:6], "evaluation": times[6:8]}
    return {name: frame.loc[frame["time"].isin(values)].reset_index(drop=True) for name, values in groups.items()}


def row_prediction(arrays, mean_dry, mean_wet, std_dry=None, std_wet=None, std_total=None, covariance=None) -> ProfilePrediction:
    shape = arrays.target_total.shape
    dry = np.asarray(mean_dry, dtype=float).reshape(shape)
    wet = np.asarray(mean_wet, dtype=float).reshape(shape)
    sd_dry = None if std_dry is None else np.asarray(std_dry, dtype=float).reshape(shape)
    sd_wet = None if std_wet is None else np.asarray(std_wet, dtype=float).reshape(shape)
    sd_total = None if std_total is None else np.asarray(std_total, dtype=float).reshape(shape)
    if covariance is None and sd_dry is not None and sd_wet is not None:
        covariance = np.zeros((shape[0], shape[1], shape[1]), dtype=np.float32)
        idx = np.arange(shape[1])
        covariance[:, idx, idx] = np.square(sd_dry) + np.square(sd_wet)
        sd_total = np.sqrt(np.square(sd_dry) + np.square(sd_wet))
    prediction = ProfilePrediction(dry, wet, dry + wet, sd_dry, sd_wet, sd_total, covariance, arrays.mask.copy())
    prediction.validate()
    return prediction


def seasonal_mean_prediction(train: pd.DataFrame, target: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train = train.copy()
    target = target.copy()
    train["month_of_year"] = pd.DatetimeIndex(train["time"]).month
    train["utc_hour"] = pd.DatetimeIndex(train["time"]).hour
    target["month_of_year"] = pd.DatetimeIndex(target["time"]).month
    target["utc_hour"] = pd.DatetimeIndex(target["time"]).hour
    valid = train["level_mask"].astype(bool) & ~train["below_ground_level"].astype(bool)
    keys = ["pressure_hpa", "month_of_year", "utc_hour"]
    lookup = train.loc[valid].groupby(keys)[["residual_dry", "residual_wet"]].mean()
    fallback = train.loc[valid].groupby("pressure_hpa")[["residual_dry", "residual_wet"]].mean()
    index = pd.MultiIndex.from_frame(target[keys])
    values = lookup.reindex(index).reset_index(drop=True)
    missing = values["residual_dry"].isna()
    if missing.any():
        values.loc[missing, ["residual_dry", "residual_wet"]] = fallback.reindex(target.loc[missing, "pressure_hpa"]).to_numpy()
    return values["residual_dry"].to_numpy(float), values["residual_wet"].to_numpy(float)


def calibrate_neural_prediction(calibration_arrays, calibration_prediction, prediction) -> list[float]:
    scales = []
    for component in ("dry", "wet"):
        target = getattr(calibration_arrays, f"target_{component}")
        mean = getattr(calibration_prediction, f"mean_{component}")
        std = getattr(calibration_prediction, f"std_{component}")
        scale = calibrate_standard_deviation((target - mean)[calibration_arrays.mask], std[calibration_arrays.mask], 0.90)
        scales.append(scale)
        setattr(prediction, f"std_{component}", getattr(prediction, f"std_{component}") * scale)
    idx = np.arange(prediction.mean_total.shape[1])
    prediction.covariance_total[:, idx, idx] = np.square(prediction.std_dry) + np.square(prediction.std_wet)
    prediction.std_total = np.sqrt(np.square(prediction.std_dry) + np.square(prediction.std_wet))
    return scales


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one unified revision-2 strong baseline under a licensed protocol.")
    parser.add_argument("--config", default="configs/revision2/baselines.yaml")
    parser.add_argument(
        "--model", required=True,
        choices=["era5", "seasonal_mean", "ridge", "hgb", "probabilistic_hgb", *PROBABILITY_VARIANTS, "hgb_mlp", "rbf_mlp"],
    )
    parser.add_argument("--protocol", default="spatial_station_disjoint", choices=["spatial_station_disjoint", "temporal_holdout", "space_time_holdout", "spatial_leave_block_out"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block")
    parser.add_argument("--block-evaluation-period", choices=["concurrent", "future"], default="concurrent")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-times", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mean-model", help="Optional frozen dry/wet HGB joblib shared by all probability variants.")
    args = parser.parse_args()
    config = load_yaml(args.config)
    data_config = load_data_config(config["data_config"])
    input_path = Path("data/revision2/smoke/processed/era5_igra_profiles" if args.smoke else config["input"])
    upstream_evidence: dict[str, Any] = {}
    if args.smoke:
        phases = load_smoke_data(input_path, args.max_times or 8)
    else:
        collocation_manifest = Path(config["input"]).parent.parent / "manifests" / "era5_igra_profile_manifest.json"
        collocation_evidence = require_collocation_manifest(data_config, collocation_manifest, input_path, require_formal=True)
        upstream_evidence = {
            "collocation_manifest": str(collocation_manifest.resolve()),
            "collocation_config_fingerprint": collocation_evidence.get("config_fingerprint"),
            "igra_profile_fingerprint": collocation_evidence.get("igra_profile_fingerprint"),
            "era5_archive_fingerprint": collocation_evidence.get("era5_archive_fingerprint"),
            "station_split_sha256": sha256_file(config["station_split"]),
            "protocol_license_matrix_sha256": sha256_file(config["protocol_license_matrix"]),
        }
        phases = load_protocol_data(
            input_path, config["station_split"], config["protocol_license_matrix"], args.protocol,
            args.max_times, args.block, args.block_evaluation_period,
        )
    arrays = {name: rows_to_profiles(frame) for name, frame in phases.items()}
    model_label = args.model
    suffix = f"_{args.block}_{args.block_evaluation_period}" if args.block else ""
    output = Path(args.output_dir or Path(config["output_root"]) / f"{args.protocol}{suffix}" / f"{model_label}_run{args.seed}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    fitted: Any = None
    parameter_count = 0
    inference_seconds = 0.0
    eval_rows = arrays["evaluation"].row_frame

    if args.model == "era5":
        inference_started = time.perf_counter()
        prediction = row_prediction(arrays["evaluation"], np.zeros(len(eval_rows)), np.zeros(len(eval_rows)))
        inference_seconds = time.perf_counter() - inference_started
    elif args.model == "seasonal_mean":
        inference_started = time.perf_counter()
        dry, wet = seasonal_mean_prediction(arrays["train"].row_frame, eval_rows)
        prediction = row_prediction(arrays["evaluation"], dry, wet)
        inference_seconds = time.perf_counter() - inference_started
    elif args.model in {"ridge", "hgb"}:
        settings = config["models"][args.model]
        fitted = fit_component_regressors(args.model, arrays["train"].row_frame, settings)
        inference_started = time.perf_counter()
        dry, wet = predict_component_regressors(fitted, eval_rows)
        prediction = row_prediction(arrays["evaluation"], dry, wet)
        inference_seconds = time.perf_counter() - inference_started
        joblib.dump(fitted, output / "model.joblib")
    elif args.model == "probabilistic_hgb":
        fitted = fit_probabilistic_hgb(arrays["train"].row_frame, arrays["calibration"].row_frame, config["models"]["hgb"], config["models"]["probabilistic_hgb"], args.seed)
        inference_started = time.perf_counter()
        dry, sd_dry, wet, sd_wet = predict_probabilistic_hgb(fitted, eval_rows)
        prediction = row_prediction(arrays["evaluation"], dry, wet, sd_dry, sd_wet)
        inference_seconds = time.perf_counter() - inference_started
        joblib.dump(fitted, output / "model.joblib")
    elif args.model in PROBABILITY_VARIANTS:
        mean_models = joblib.load(args.mean_model) if args.mean_model else None
        fitted = fit_confirmatory_probability_hgb(
            arrays["train"].row_frame,
            arrays["calibration"].row_frame,
            config["models"]["hgb"],
            config["models"]["probabilistic_hgb"],
            args.model,
            args.seed,
            mean_models=mean_models,
        )
        inference_started = time.perf_counter()
        prediction = predict_confirmatory_probability_hgb(fitted, arrays["evaluation"])
        inference_seconds = time.perf_counter() - inference_started
        joblib.dump(fitted, output / "model.joblib")
        fitted["station_fold_table"].to_csv(output / "crossfit_station_audit.csv", index=False)
        write_json(output / "probability_license_audit.json", fitted["license_audit"])
    else:
        device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
        neural = config["models"]["neural"]
        if args.model == "hgb_mlp":
            prior_models = fit_component_regressors("hgb", arrays["train"].row_frame, config["models"]["hgb"])
            prior = {}
            for phase in arrays:
                dry, wet = predict_component_regressors(prior_models, arrays[phase].row_frame)
                prior[phase] = np.stack([dry, wet], axis=-1).reshape(*arrays[phase].target_total.shape, 2)
            extras = {phase: None for phase in arrays}
            fitted = {"prior": prior_models}
        else:
            prior = {phase: np.zeros((*arrays[phase].target_total.shape, 2), dtype=np.float32) for phase in arrays}
            anchors = choose_rbf_anchors(arrays["train"].keys, int(neural.get("rbf_anchors", 64)), args.seed)
            extras = {phase: rbf_features(arrays[phase].keys["latitude"], arrays[phase].keys["longitude"], anchors, float(neural.get("rbf_scale_km", 1500.0))) for phase in arrays}
            fitted = {"anchors": anchors}
        network, metadata, history = fit_profile_network(
            arrays["train"], arrays["early_stopping"], prior["train"], prior["early_stopping"], neural,
            device, output / "checkpoint", args.seed, extras["train"], extras["early_stopping"], args.epochs,
            args.resume,
        )
        parameter_count = sum(parameter.numel() for parameter in network.parameters())
        calibration_prediction = predict_profile_network(network, arrays["calibration"], prior["calibration"], metadata, neural, device, extras["calibration"])
        inference_started = time.perf_counter()
        prediction = predict_profile_network(network, arrays["evaluation"], prior["evaluation"], metadata, neural, device, extras["evaluation"])
        inference_seconds = time.perf_counter() - inference_started
        fitted["calibration_scales"] = calibrate_neural_prediction(arrays["calibration"], calibration_prediction, prediction)
        joblib.dump(fitted, output / "auxiliary_models.joblib")

    elapsed = time.perf_counter() - started
    predictions = prediction_to_rows(arrays["evaluation"], prediction, model_label, args.seed)
    predictions.to_parquet(output / "predictions.parquet", index=False, compression="zstd")
    covariance_audit = validate_covariance_masks(prediction)
    covariance_table = covariance_frame(arrays["evaluation"], prediction)
    if not covariance_table.empty:
        covariance_table.to_parquet(output / "profile_covariance.parquet", index=False, compression="zstd")
    evaluation_settings = config["evaluation"]
    tables = metric_tables(
        predictions, list(map(float, evaluation_settings["confidence_levels"])),
        int(evaluation_settings["bootstrap_replicates"]), args.seed,
        int(evaluation_settings.get("energy_score_draws", 64)),
        arrays=arrays["evaluation"], prediction=prediction,
    )
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    inference_rows = int(predictions["evaluation_mask"].sum())
    peak_gpu_mb = float(torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0
    pd.DataFrame([{
        "model": model_label, "protocol": args.protocol, "seed": args.seed, "parameter_count": parameter_count,
        "total_seconds": elapsed, "training_and_setup_seconds": max(elapsed - inference_seconds, 0.0),
        "inference_seconds": inference_seconds, "evaluated_level_rows": inference_rows,
        "evaluated_level_rows_per_second": inference_rows / max(inference_seconds, 1e-9), "peak_gpu_memory_mb": peak_gpu_mb,
    }]).to_csv(output / "compute_cost.csv", index=False)
    manifest = {
        "experiment_family": config["experiment_family"], "dataset_version": data_config["study"].get("dataset_version"),
        "model": model_label, "protocol": args.protocol, "block": args.block,
        "block_evaluation_period": args.block_evaluation_period if args.block else None,
        "seed": args.seed, "smoke": args.smoke,
        "input": str(input_path.resolve()), "phase_profiles": {key: len(value.keys) for key, value in arrays.items()},
        "config_fingerprint": stable_fingerprint(config), "upstream_evidence": upstream_evidence,
        "mean_model_source": str(Path(args.mean_model).resolve()) if args.mean_model else None,
        "covariance_audit": covariance_audit,
        "output": str(output.resolve()), "complete": True,
    }
    write_json(output / "protocol_manifest.json", manifest)
    print(f"Results: {output.resolve()}")


if __name__ == "__main__":
    main()
