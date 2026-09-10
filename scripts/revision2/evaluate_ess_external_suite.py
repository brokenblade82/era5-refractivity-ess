from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from _bootstrap import PROJECT_ROOT  # noqa: F401
import evaluate_cosmic2_height as cosmic
from evaluate_external_profiles import land_sea_classification, prepare_external_rows
from igra_forecast.revision2_baselines import LEVELS, make_row_features, rows_to_profiles
from igra_forecast.revision2_data import (
    load_data_config,
    require_era5_audit,
    resolve_era5_root,
    sha256_file,
    stable_fingerprint,
    write_json,
)
from igra_forecast.revision2_ess import (
    ESS_MODELS,
    add_common_groups,
    apportion_total_refractivity,
    build_seasonal_lookup,
    constant_feature_support,
    ess_fingerprint,
    frozen_run,
    load_protocol_training_rows,
    load_yaml,
    predict_row_model,
    predict_structured_hgb,
    training_wet_edges,
    validate_seasonal_reconstruction,
)
from igra_forecast.revision2_publication import (
    geometric_height_evaluation_mask,
    interpolate_no_extrapolation,
    propagate_gaussian_to_height,
    validate_geometric_height_schema,
)


def _assert_hash(path: str | Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"Frozen {label} hash mismatch: expected={expected}, actual={actual}, path={path}")
    return actual


def _load_frozen_context(config: dict[str, Any]) -> dict[str, Any]:
    if any(bool(config["frozen_boundaries"].get(key)) for key in (
        "allow_training", "allow_model_selection_from_external_data", "allow_external_recalibration", "allow_global_product"
    )):
        raise ValueError("ESS frozen-boundary flags must all remain false")
    frozen: dict[str, Any] = {}
    for name in ("ridge", "hgb", "hgb_probability"):
        settings = config["frozen_runs"][name]
        item = frozen_run(name, settings["path"])
        if item.model_sha256 != settings["model_sha256"]:
            raise ValueError(f"Frozen model hash mismatch for {name}")
        frozen[name] = item
    seasonal_settings = config["frozen_runs"]["seasonal_mean"]
    seasonal_run = frozen_run("seasonal_mean", seasonal_settings["path"], require_model=False)
    seasonal_prediction = seasonal_run.path / "predictions.parquet"
    _assert_hash(seasonal_prediction, seasonal_settings["frozen_prediction_sha256"], "seasonal prediction")
    frozen["seasonal_mean"] = seasonal_run
    for label, item in config["frozen_inputs"].items():
        if label in {"rapsodi", "cosmic2_height"}:
            _assert_hash(item["path"], item["sha256"], label)
            manifest_key = "collocation_manifest" if label == "rapsodi" else "preparation_manifest"
            _assert_hash(item[manifest_key], item[f"{manifest_key}_sha256"], f"{label} manifest")
        else:
            _assert_hash(item["path"], item["sha256"], label)
    train = load_protocol_training_rows(config["input"], config["station_split"])
    training_features = make_row_features(train)
    seasonal_model = build_seasonal_lookup(train)
    seasonal_audit = validate_seasonal_reconstruction(
        seasonal_model,
        seasonal_prediction,
        float(config["statistics"]["seasonal_reproduction_tolerance"]),
    )
    models = {
        "seasonal_mean": seasonal_model,
        "ridge": joblib.load(frozen["ridge"].model_path),
        "hgb": joblib.load(frozen["hgb"].model_path),
        "hgb_probability": joblib.load(frozen["hgb_probability"].model_path),
    }
    return {
        "runs": frozen,
        "models": models,
        "wet_edges": training_wet_edges(train),
        "seasonal_audit": seasonal_audit,
        "feature_support": {
            "mean": training_features.mean(axis=0),
            "std": training_features.std(axis=0),
            "min": training_features.min(axis=0),
            "max": training_features.max(axis=0),
        },
        "fingerprint": ess_fingerprint(config, frozen),
    }


def _ridge_support(frame: pd.DataFrame, context: dict[str, Any]) -> np.ndarray:
    features = make_row_features(frame)
    support = context["feature_support"]
    return constant_feature_support(features, support["mean"], support["std"])


def _point_metrics(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected = frame.loc[frame["evaluation_mask"]].copy()
    observed = selected["observed_n"].to_numpy(float)
    era5_column = "era5_n_height" if source == "cosmic2" else "era5_n"
    for model, prediction_column in [
        ("era5", era5_column),
        *[(name, f"{name}_prediction_n") for name in ESS_MODELS],
    ]:
        error = selected[prediction_column].to_numpy(float) - observed
        support_fraction = (
            float(selected["ridge_feature_support"].mean())
            if model == "ridge" and "ridge_feature_support" in selected else 1.0
        )
        rows.append({
            "source": source, "model": model, "n": int(len(error)),
            "bias": float(error.mean()), "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "training_feature_support_fraction": support_fraction,
            "comparison_valid": bool(support_fraction == 1.0),
            "status": "valid" if support_fraction == 1.0 else "out_of_training_constant_feature_support",
        })
    table = pd.DataFrame(rows)
    era5_rmse = float(table.loc[table["model"] == "era5", "rmse"].iloc[0])
    table["model_minus_era5_rmse"] = table["rmse"] - era5_rmse
    return table


def _grouped_metrics(frame: pd.DataFrame, source: str, groups: list[str]) -> pd.DataFrame:
    tables = []
    for group_column in groups:
        for group, part in frame.loc[frame["evaluation_mask"]].groupby(group_column, observed=True, dropna=False):
            values = _point_metrics(part.assign(evaluation_mask=True), source)
            values.insert(1, "grouping", group_column)
            values.insert(2, "group", str(group))
            tables.append(values)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _rapsodi(config: dict[str, Any], context: dict[str, Any], output: Path) -> None:
    settings = config["frozen_inputs"]["rapsodi"]
    frame, platform_audit = prepare_external_rows(
        pd.read_parquet(settings["path"]), "rapsodi", Path(settings["platform_audit"])
    )
    arrays = rows_to_profiles(frame)
    rows = arrays.row_frame.copy()
    probability = predict_structured_hgb(context["models"]["hgb_probability"], rows)
    shape = probability.mean_total.shape
    hgb_dry, hgb_wet = predict_row_model("hgb", rows, context["models"]["hgb"])
    if not np.allclose(hgb_dry.reshape(shape), probability.mean_dry, atol=1e-10) or not np.allclose(
        hgb_wet.reshape(shape), probability.mean_wet, atol=1e-10
    ):
        raise ValueError("Frozen deterministic HGB and structured-probability HGB means differ")
    for name in ESS_MODELS:
        dry, wet = predict_row_model(name, rows, context["models"][name])
        rows[f"{name}_correction_dry"] = dry
        rows[f"{name}_correction_wet"] = wet
        rows[f"{name}_correction_total"] = dry + wet
        rows[f"{name}_prediction_n"] = rows["era5_n"].to_numpy(float) + dry + wet
    rows["hgb_prediction_std"] = probability.std_total.reshape(-1)
    rows["ridge_feature_support"] = _ridge_support(rows, context)
    rows["evaluation_mask"] = arrays.mask.reshape(-1)
    rows["observed_n"] = rows["era5_n"] + rows["residual_total"]
    static_manifest = json.loads(Path("data/revision2/manifests/era5_archive_manifest.json").read_text(encoding="utf-8"))
    locations = rows[["station_id", "time", "latitude", "longitude"]].drop_duplicates()
    locations["land_sea"] = land_sea_classification(locations, Path(static_manifest["static"]["path"]))
    rows = rows.merge(locations, on=["station_id", "time", "latitude", "longitude"], how="left", validate="many_to_one")
    rows = add_common_groups(rows, context["wet_edges"])
    output.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output / "predictions.parquet", index=False, compression="zstd")
    _point_metrics(rows, "rapsodi").to_csv(output / "model_metrics.csv", index=False)
    _grouped_metrics(rows, "rapsodi", ["pressure_hpa", "macro_region", "land_sea", "wet_regime", "date"]).to_csv(
        output / "grouped_metrics.csv", index=False
    )
    write_json(output / "manifest.json", {
        "complete": True, "source": "rapsodi", "ess_fingerprint": context["fingerprint"],
        "input": str(Path(settings["path"]).resolve()), "input_sha256": settings["sha256"],
        "platform_audit": platform_audit, "models": list(ESS_MODELS),
        "hgb_probability_only": True, "external_data_used_for_training_selection_or_calibration": False,
        "rows": int(len(rows)), "evaluable_rows": int(rows["evaluation_mask"].sum()),
        "output_sha256": sha256_file(output / "predictions.parquet"),
    })


def _six_level_frame(target: pd.DataFrame, collocated: dict[str, np.ndarray], all_levels: np.ndarray) -> pd.DataFrame:
    correction_index = np.asarray([int(np.flatnonzero(all_levels == level)[0]) for level in LEVELS], dtype=int)
    n_target = len(target)
    synthetic = np.asarray([f"H{number:09d}" for number in range(n_target)])
    return pd.DataFrame({
        "station_id": np.repeat(synthetic, len(LEVELS)),
        "time": np.repeat(pd.to_datetime(target["time"]).to_numpy(), len(LEVELS)),
        "latitude": np.repeat(target["latitude"].to_numpy(float), len(LEVELS)),
        "longitude": np.repeat(target["longitude"].to_numpy(float), len(LEVELS)),
        "station_elevation_m": np.repeat(target["surface_height_m"].to_numpy(float), len(LEVELS)),
        "pressure_hpa": np.tile(LEVELS, n_target),
        "era5_temperature_k": collocated["temperature"][:, correction_index].reshape(-1),
        "era5_specific_humidity": collocated["humidity"][:, correction_index].reshape(-1),
        "era5_height_m": collocated["height"][:, correction_index].reshape(-1),
        "era5_n_dry": collocated["n_dry"][:, correction_index].reshape(-1),
        "era5_n_wet": collocated["n_wet"][:, correction_index].reshape(-1),
        "era5_n": collocated["n_total"][:, correction_index].reshape(-1),
        "era5_surface_pressure_pa": np.repeat(collocated["surface_pressure"], len(LEVELS)),
        "below_ground_level": np.repeat(collocated["surface_pressure"], len(LEVELS)) < np.tile(LEVELS, n_target) * 100.0,
        "level_mask": 1, "residual_dry": 0.0, "residual_wet": 0.0, "residual_total": 0.0,
    })


def _cosmic_chunk(
    target: pd.DataFrame,
    era5_arrays: dict[str, xr.DataArray],
    static_z: xr.DataArray,
    static_lsm: xr.DataArray,
    all_levels: np.ndarray,
    context: dict[str, Any],
) -> pd.DataFrame:
    target = target.copy().reset_index(drop=True)
    target["surface_height_m"] = cosmic.static_points(static_z, target) / cosmic.G0
    target["land_sea"] = np.where(cosmic.static_points(static_lsm, target) >= 0.5, "land", "sea")
    collocated = cosmic.collocate_pressure_profiles(target, era5_arrays, all_levels)
    six = _six_level_frame(target, collocated, all_levels)
    arrays = rows_to_profiles(six)
    ridge_row_support = _ridge_support(six, context).reshape(arrays.target_total.shape)
    ridge_profile_support = ridge_row_support.all(axis=1)
    probability = predict_structured_hgb(context["models"]["hgb_probability"], six)
    model_profiles: dict[str, dict[str, np.ndarray]] = {}
    for name in ESS_MODELS:
        dry, wet = predict_row_model(name, six, context["models"][name])
        model_profiles[name] = {
            "dry": dry.reshape(arrays.target_total.shape),
            "wet": wet.reshape(arrays.target_total.shape),
        }
    if not np.allclose(model_profiles["hgb"]["dry"], probability.mean_dry, atol=1e-10) or not np.allclose(
        model_profiles["hgb"]["wet"], probability.mean_wet, atol=1e-10
    ):
        raise ValueError("Frozen deterministic and probability HGB means differ in COSMIC inference")
    correction_index = np.asarray([int(np.flatnonzero(all_levels == level)[0]) for level in LEVELS], dtype=int)
    outputs: list[dict[str, Any]] = []
    for row_index, row in target.iterrows():
        height = float(row["height_m"])
        above_surface = all_levels * 100.0 <= collocated["surface_pressure"][row_index]
        baseline_height = collocated["height"][row_index]
        baseline_total = collocated["n_total"][row_index]
        valid = above_surface & np.isfinite(baseline_height) & np.isfinite(baseline_total) & (baseline_total > 0)
        era5_total = interpolate_no_extrapolation(
            baseline_height[valid], baseline_total[valid], np.asarray([height]), log_values=True
        )[0]
        wet_fraction = np.divide(
            collocated["n_wet"][row_index], baseline_total,
            out=np.full_like(baseline_total, np.nan, dtype=float), where=baseline_total > 0,
        )
        fraction = interpolate_no_extrapolation(
            baseline_height[valid], wet_fraction[valid], np.asarray([height]), log_values=False
        )[0]
        era5_dry_array, era5_wet_array = apportion_total_refractivity(
            np.asarray([era5_total]), np.asarray([fraction])
        )
        era5_dry, era5_wet = float(era5_dry_array[0]), float(era5_wet_array[0])
        correction_height = collocated["height"][row_index, correction_index]
        correction_valid = (LEVELS * 100.0 <= collocated["surface_pressure"][row_index]) & np.isfinite(correction_height)
        selected = np.flatnonzero(correction_valid)
        corrections: dict[str, tuple[float, float, float]] = {}
        supported = False
        if selected.size >= 2:
            for name in ESS_MODELS:
                dry = interpolate_no_extrapolation(
                    correction_height[selected], model_profiles[name]["dry"][row_index, selected], np.asarray([height])
                )[0]
                wet = interpolate_no_extrapolation(
                    correction_height[selected], model_profiles[name]["wet"][row_index, selected], np.asarray([height])
                )[0]
                corrections[name] = (float(dry), float(wet), float(dry + wet))
            mean, variance, support = propagate_gaussian_to_height(
                correction_height[selected], probability.mean_total[row_index, selected],
                probability.covariance_total[row_index][np.ix_(selected, selected)], np.asarray([height]),
            )
            supported = bool(support[0])
            hgb_variance = float(variance[0]) if supported else np.nan
            if supported and abs(float(mean[0]) - corrections["hgb"][2]) > 1e-8:
                raise ValueError("HGB deterministic and covariance-propagated height means differ")
        else:
            hgb_variance = np.nan
            corrections = {name: (np.nan, np.nan, np.nan) for name in ESS_MODELS}
        mask = bool(geometric_height_evaluation_mask(
            np.asarray([row["observed_n"]]), np.asarray([height]), np.asarray([row["surface_height_m"]]),
            np.asarray([era5_total]), np.asarray([corrections["hgb"][2]]), np.asarray([hgb_variance]),
            np.asarray([row["height_mask"]]), np.asarray([supported]),
        )[0])
        record = {
            **row.to_dict(), "era5_n_height": era5_total, "era5_n_dry_height": era5_dry,
            "era5_n_wet_height": era5_wet, "hgb_prediction_std": math.sqrt(hgb_variance) if np.isfinite(hgb_variance) and hgb_variance >= 0 else np.nan,
            "below_surface": height <= float(row["surface_height_m"]), "within_era5_height_support": bool(np.isfinite(era5_total)),
            "within_correction_height_support": supported, "evaluation_mask": mask,
            "ridge_feature_support": bool(ridge_profile_support[row_index]),
        }
        for name, (dry, wet, total) in corrections.items():
            record[f"{name}_correction_dry"] = dry
            record[f"{name}_correction_wet"] = wet
            record[f"{name}_correction_total"] = total
            record[f"{name}_prediction_n"] = era5_total + total if mask else np.nan
        outputs.append(record)
    return pd.DataFrame(outputs)


def _cosmic(config: dict[str, Any], context: dict[str, Any], output: Path, input_override: str | None, manifest_override: str | None, chunk_profiles: int, resume: bool, max_profiles: int | None) -> None:
    frozen_input = config["frozen_inputs"]["cosmic2_height"]
    input_path = Path(input_override or frozen_input["path"])
    manifest_path = Path(manifest_override or frozen_input["preparation_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if Path(manifest["output"]).resolve() != input_path.resolve() or manifest.get("dry_pressure_used") is not False:
        raise ValueError("COSMIC height input is not bound to its geometric-height preparation manifest")
    if manifest.get("output_sha256") != sha256_file(input_path):
        raise ValueError("COSMIC height parquet hash differs from its preparation manifest")
    frame = pd.read_parquet(input_path)
    validate_geometric_height_schema(frame)
    if max_profiles is not None:
        keys = frame[["profile_id", "time"]].drop_duplicates().head(int(max_profiles))
        frame = frame.merge(keys, on=["profile_id", "time"], how="inner", validate="many_to_one")
    frame["time"] = pd.to_datetime(frame["time"])
    frame["period"] = frame["time"].dt.to_period("M")
    data_config = load_data_config(config["data_config"])
    era5_root = resolve_era5_root(data_config, None)
    periods = sorted(frame["period"].unique())
    audit_path = Path(data_config["paths"]["manifests"]) / "era5_archive_manifest.json"
    require_era5_audit(data_config, audit_path, era5_root, [period.strftime("%Y%m") for period in periods], require_full_study=max_profiles is None)
    archive_manifest = json.loads(audit_path.read_text(encoding="utf-8"))
    all_levels = np.asarray(config["cosmic2_height"]["era5_pressure_levels_hpa"], dtype=int)
    output.mkdir(parents=True, exist_ok=True)
    predictions_root = output / "predictions"
    predictions_root.mkdir(parents=True, exist_ok=True)
    run_fingerprint = stable_fingerprint({
        "ess_fingerprint": context["fingerprint"], "input_sha256": sha256_file(input_path),
        "schema": int(config["cosmic2_height"]["output_schema_version"]), "models": list(ESS_MODELS),
    })
    with xr.open_dataset(Path(archive_manifest["static"]["path"])) as static_dataset:
        static_z = static_dataset["z"].squeeze(drop=True).load()
        static_lsm = static_dataset["lsm"].squeeze(drop=True).load()
    month_files: list[Path] = []
    for period in tqdm(periods, desc="ESS COSMIC-2 model suite months"):
        yyyymm = period.strftime("%Y%m")
        month_file = predictions_root / f"predictions_{yyyymm}.parquet"
        month_meta = predictions_root / f"predictions_{yyyymm}.json"
        if resume and month_file.is_file() and month_meta.is_file():
            meta = json.loads(month_meta.read_text(encoding="utf-8"))
            if meta.get("run_fingerprint") == run_fingerprint and meta.get("sha256") == sha256_file(month_file):
                month_files.append(month_file)
                continue
        selected = frame.loc[frame["period"] == period].drop(columns="period").copy()
        profile_keys = selected[["profile_id", "time"]].drop_duplicates().reset_index(drop=True)
        source_arrays = {logical: cosmic.open_era5_variable(era5_root, period, data_config, logical) for logical in ["temperature", "specific_humidity", "geopotential", "surface_pressure"]}
        arrays, preload_audit = cosmic.preload_month_arrays(source_arrays, selected, all_levels)
        chunks: list[pd.DataFrame] = []
        for start in tqdm(range(0, len(profile_keys), int(chunk_profiles)), desc=yyyymm, leave=False):
            chunk_keys = profile_keys.iloc[start:start + int(chunk_profiles)]
            target = selected.merge(chunk_keys, on=["profile_id", "time"], how="inner", validate="many_to_one")
            chunks.append(_cosmic_chunk(target, arrays, static_z, static_lsm, all_levels, context))
        for array in source_arrays.values():
            array.close()
        result = add_common_groups(pd.concat(chunks, ignore_index=True), context["wet_edges"])
        temporary = month_file.with_suffix(".parquet.tmp")
        result.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, month_file)
        write_json(month_meta, {"run_fingerprint": run_fingerprint, "rows": len(result), "sha256": sha256_file(month_file), "preload_audit": preload_audit})
        month_files.append(month_file)
    predictions = pd.concat([pd.read_parquet(path) for path in month_files], ignore_index=True)
    _point_metrics(predictions, "cosmic2").to_csv(output / "model_metrics.csv", index=False)
    _grouped_metrics(predictions, "cosmic2", ["height_m", "macro_region", "land_sea", "wet_regime", "year", "month"]).to_csv(output / "grouped_metrics.csv", index=False)
    write_json(output / "manifest.json", {
        "complete": True, "source": "cosmic2", "ess_fingerprint": context["fingerprint"],
        "input": str(input_path.resolve()), "input_sha256": sha256_file(input_path),
        "preparation_manifest": str(manifest_path.resolve()), "models": list(ESS_MODELS),
        "hgb_probability_only": True, "external_data_used_for_training_selection_or_calibration": False,
        "coordinate": "MSL_geometric_height", "dry_pressure_used": False,
        "rows": int(len(predictions)), "evaluable_rows": int(predictions["evaluation_mask"].sum()),
        "profiles": int(predictions.loc[predictions["evaluation_mask"], "profile_id"].nunique()),
        "archives": int(predictions.loc[predictions["evaluation_mask"], "source_archive"].nunique()),
        "month_files": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in month_files],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the frozen ESS simple-model suite to one external observing system.")
    parser.add_argument("--config", default="configs/revision2/ess.yaml")
    parser.add_argument("--source", required=True, choices=["rapsodi", "cosmic2"])
    parser.add_argument("--input")
    parser.add_argument("--prepare-manifest")
    parser.add_argument("--output-dir")
    parser.add_argument("--chunk-profiles", type=int, default=1000)
    parser.add_argument("--max-profiles", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    context = _load_frozen_context(config)
    root = Path(config["outputs"]["external_root"])
    output = Path(args.output_dir or root / ("rapsodi_model_suite" if args.source == "rapsodi" else "cosmic2_height_model_suite"))
    if args.source == "rapsodi":
        _rapsodi(config, context, output)
    else:
        _cosmic(config, context, output, args.input, args.prepare_manifest, args.chunk_profiles, args.resume, args.max_profiles)
    manifest_path = root / "model_suite_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("ess_fingerprint") != context["fingerprint"]:
            raise ValueError("Existing ESS model-suite manifest belongs to a different frozen evidence set")
        # Migrate the first single-source draft manifest without preserving an
        # ambiguous last-completed-source label.
        manifest.pop("complete_source", None)
    else:
        manifest = {
            "ess_fingerprint": context["fingerprint"],
            "seasonal_reconstruction": context["seasonal_audit"],
            "models": list(ESS_MODELS),
            "frozen_model_hashes": {name: run.model_sha256 for name, run in context["runs"].items()},
            "external_data_used_for_training_selection_or_calibration": False,
            "sources": {},
        }
    manifest.setdefault("sources", {})[args.source] = {
        "complete": True,
        "output": str(output.resolve()),
        "manifest": str((output / "manifest.json").resolve()),
        "ridge_comparison_policy": (
            "raw predictions retained, but metrics are excluded from cross-model claims when frozen-feature support is violated"
        ),
    }
    write_json(manifest_path, manifest)
    print(f"ESS external suite: {output.resolve()}")


if __name__ == "__main__":
    main()
