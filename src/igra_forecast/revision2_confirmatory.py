from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2
from tqdm import tqdm

from .revision2_baselines import (
    LEVELS,
    ProfileArrays,
    ProfilePrediction,
    calibrate_standard_deviation,
    fit_component_regressors,
    hgb_factory,
    make_row_features,
    row_training_mask,
    station_crossfit_folds,
)


PROBABILITY_VARIANTS = (
    "hgb_prob_global",
    "hgb_prob_level",
    "hgb_prob_hetero_diag",
    "hgb_prob_hetero_structured",
)


def fixed_hgb_mean_models(train: pd.DataFrame, settings: dict[str, Any]) -> dict[str, Any]:
    """Fit the single frozen dry/wet HGB mean used by every probability variant."""
    fixed = {**settings, "random_state": int(settings.get("random_state", 42))}
    return fit_component_regressors("hgb", train, fixed)


def _profile_matrix(frame: pd.DataFrame, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    work = frame[["station_id", "time", "pressure_hpa"]].copy()
    work["value"] = np.asarray(values, dtype=float)
    work["pressure_hpa"] = work["pressure_hpa"].astype(int)
    matrix = work.pivot(index=["station_id", "time"], columns="pressure_hpa", values="value")
    matrix = matrix.reindex(columns=list(map(int, LEVELS)))
    return matrix.to_numpy(float), matrix.index.to_numpy()


def nearest_correlation_matrix(correlation: np.ndarray, eigen_floor: float = 1e-5) -> np.ndarray:
    """Project a symmetric pairwise correlation estimate to a positive-definite correlation matrix."""
    value = np.asarray(correlation, dtype=float)
    value = np.where(np.isfinite(value), value, 0.0)
    value = (value + value.T) / 2.0
    np.fill_diagonal(value, 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    projected = (eigenvectors * np.maximum(eigenvalues, eigen_floor)) @ eigenvectors.T
    scale = np.sqrt(np.maximum(np.diag(projected), eigen_floor))
    projected = projected / np.outer(scale, scale)
    projected = (projected + projected.T) / 2.0
    np.fill_diagonal(projected, 1.0)
    return projected


def estimate_pairwise_correlation(
    train_valid: pd.DataFrame,
    standardized_total_error: np.ndarray,
    minimum_pair_count: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    matrix, _ = _profile_matrix(train_valid, standardized_total_error)
    n_level = matrix.shape[1]
    correlation = np.eye(n_level, dtype=float)
    counts = np.zeros((n_level, n_level), dtype=int)
    for left in range(n_level):
        for right in range(left, n_level):
            valid = np.isfinite(matrix[:, left]) & np.isfinite(matrix[:, right])
            counts[left, right] = counts[right, left] = int(valid.sum())
            if left == right:
                continue
            if valid.sum() < minimum_pair_count:
                coefficient = 0.0
            else:
                coefficient = float(np.corrcoef(matrix[valid, left], matrix[valid, right])[0, 1])
                if not np.isfinite(coefficient):
                    coefficient = 0.0
            correlation[left, right] = correlation[right, left] = coefficient
    return nearest_correlation_matrix(correlation), counts


def _raw_scale_by_variant(
    variant: str,
    train_valid: pd.DataFrame,
    x_valid: np.ndarray,
    error: np.ndarray,
    hgb_settings: dict[str, Any],
    probability_settings: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    floor = float(probability_settings.get("scale_floor", 0.05))
    if variant == "hgb_prob_global":
        return {"kind": "global", "value": float(max(np.sqrt(np.mean(np.square(error))), floor))}
    if variant == "hgb_prob_level":
        level_values: dict[int, float] = {}
        pressure = train_valid["pressure_hpa"].to_numpy(int)
        fallback = float(max(np.sqrt(np.mean(np.square(error))), floor))
        for level in LEVELS:
            selected = pressure == int(level)
            level_values[int(level)] = float(max(np.sqrt(np.mean(np.square(error[selected]))), floor)) if selected.any() else fallback
        return {"kind": "level", "values": level_values, "fallback": fallback}
    model = hgb_factory({**hgb_settings, "random_state": seed + 1000})
    target = np.log(np.maximum(np.abs(error), floor))
    model.fit(x_valid, target)
    return {
        "kind": "heteroscedastic",
        "model": model,
        "log_scale_bounds": [
            float(probability_settings.get("log_scale_min", -5.0)),
            float(probability_settings.get("log_scale_max", 4.0)),
        ],
    }


def predict_raw_scale(item: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    if item["kind"] == "global":
        return np.full(len(frame), float(item["value"]), dtype=float)
    if item["kind"] == "level":
        return frame["pressure_hpa"].astype(int).map(item["values"]).fillna(float(item["fallback"])).to_numpy(float)
    bounds = item["log_scale_bounds"]
    return np.exp(np.clip(item["model"].predict(make_row_features(frame)), bounds[0], bounds[1]))


def _level_calibration_scales(
    calibration: pd.DataFrame,
    error: np.ndarray,
    raw_std: np.ndarray,
    coverage: float,
) -> dict[int, float]:
    pressure = calibration["pressure_hpa"].to_numpy(int)
    valid = row_training_mask(calibration)
    fallback = calibrate_standard_deviation(error[valid], raw_std[valid], coverage)
    result: dict[int, float] = {}
    for level in LEVELS:
        selected = valid & (pressure == int(level))
        result[int(level)] = calibrate_standard_deviation(error[selected], raw_std[selected], coverage) if selected.any() else fallback
    return result


def apply_level_scales(frame: pd.DataFrame, raw_std: np.ndarray, scales: dict[int, float]) -> np.ndarray:
    multiplier = frame["pressure_hpa"].astype(int).map(scales).to_numpy(float)
    return np.maximum(np.asarray(raw_std, dtype=float) * multiplier, 1e-6)


def fit_confirmatory_probability_hgb(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    hgb_settings: dict[str, Any],
    probability_settings: dict[str, Any],
    variant: str,
    seed: int,
    mean_models: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if variant not in PROBABILITY_VARIANTS:
        raise ValueError(f"Unsupported confirmatory probability variant: {variant}")
    valid = row_training_mask(train)
    train_valid = train.loc[valid].reset_index(drop=True)
    x_valid = make_row_features(train_valid)
    n_folds = int(probability_settings.get("crossfit_folds", 5))
    folds = station_crossfit_folds(train_valid["station_id"], n_folds, seed)
    components = ("dry", "wet")
    oof_predictions: dict[str, np.ndarray] = {name: np.full(len(train_valid), np.nan) for name in components}
    for fold in tqdm(sorted(np.unique(folds)), desc=f"{variant} station cross-fit"):
        selected_train = folds != fold
        selected_holdout = folds == fold
        for component in components:
            model = hgb_factory({**hgb_settings, "random_state": int(hgb_settings.get("random_state", 42)) + fold})
            model.fit(x_valid[selected_train], train_valid.loc[selected_train, f"residual_{component}"].to_numpy(float))
            oof_predictions[component][selected_holdout] = model.predict(x_valid[selected_holdout])
    if not all(np.isfinite(value).all() for value in oof_predictions.values()):
        raise ValueError("Station cross-fit left non-finite HGB predictions")

    mean_models = mean_models or fixed_hgb_mean_models(train, hgb_settings)
    errors = {
        "dry": train_valid["residual_dry"].to_numpy(float) - oof_predictions["dry"],
        "wet": train_valid["residual_wet"].to_numpy(float) - oof_predictions["wet"],
    }
    errors["total"] = errors["dry"] + errors["wet"]
    scale_models: dict[str, dict[str, Any]] = {}
    for component in ("dry", "wet", "total"):
        scale_models[component] = _raw_scale_by_variant(
            variant, train_valid, x_valid, errors[component], hgb_settings, probability_settings, seed
        )

    cal_valid = row_training_mask(calibration)
    x_cal = make_row_features(calibration)
    mean_cal_dry = mean_models["dry"].predict(x_cal)
    mean_cal_wet = mean_models["wet"].predict(x_cal)
    cal_errors = {
        "dry": calibration["residual_dry"].to_numpy(float) - mean_cal_dry,
        "wet": calibration["residual_wet"].to_numpy(float) - mean_cal_wet,
    }
    cal_errors["total"] = cal_errors["dry"] + cal_errors["wet"]
    coverage = float(probability_settings.get("calibration_coverage", 0.90))
    calibration_scales: dict[str, dict[int, float]] = {}
    for component in ("dry", "wet", "total"):
        raw_calibration = predict_raw_scale(scale_models[component], calibration)
        if variant == "hgb_prob_global":
            selected = cal_valid & np.isfinite(cal_errors[component])
            scalar = calibrate_standard_deviation(cal_errors[component][selected], raw_calibration[selected], coverage)
            calibration_scales[component] = {int(level): scalar for level in LEVELS}
        else:
            calibration_scales[component] = _level_calibration_scales(
                calibration, cal_errors[component], raw_calibration, coverage
            )

    train_total_raw = predict_raw_scale(scale_models["total"], train_valid)
    standardized = errors["total"] / np.maximum(train_total_raw, 1e-6)
    if variant == "hgb_prob_hetero_structured":
        correlation, pair_counts = estimate_pairwise_correlation(
            train_valid, standardized, int(probability_settings.get("minimum_pair_count", 30))
        )
    else:
        correlation = np.eye(len(LEVELS), dtype=float)
        pair_counts = np.zeros_like(correlation, dtype=int)
    station_fold = train_valid[["station_id"]].copy()
    station_fold["crossfit_fold"] = folds
    station_fold = station_fold.drop_duplicates().sort_values("station_id").reset_index(drop=True)
    train_keys = set(zip(train["station_id"].astype(str), pd.to_datetime(train["time"], utc=True)))
    calibration_keys = set(zip(calibration["station_id"].astype(str), pd.to_datetime(calibration["time"], utc=True)))
    return {
        "variant": variant,
        "seed": int(seed),
        "folds": folds,
        "fold_station_ids": train_valid["station_id"].astype(str).to_numpy(),
        "mean_models": mean_models,
        "scale_models": scale_models,
        "calibration_scales": calibration_scales,
        "correlation_total": correlation,
        "correlation_pair_counts": pair_counts,
        "mean_random_state": int(hgb_settings.get("random_state", 42)),
        "calibration_coverage": coverage,
        "station_fold_table": station_fold,
        "license_audit": {
            "training_stations": int(train["station_id"].nunique()),
            "calibration_stations": int(calibration["station_id"].nunique()),
            "training_calibration_station_time_overlap": int(len(train_keys & calibration_keys)),
            "scale_models_use_calibration_rows": False,
            "mean_models_use_calibration_rows": False,
        },
    }


def predict_confirmatory_probability_hgb(
    model: dict[str, Any],
    arrays: ProfileArrays,
) -> ProfilePrediction:
    frame = arrays.row_frame
    x = make_row_features(frame)
    mean_dry = np.asarray(model["mean_models"]["dry"].predict(x), dtype=float)
    mean_wet = np.asarray(model["mean_models"]["wet"].predict(x), dtype=float)
    std = {
        component: apply_level_scales(
            frame,
            predict_raw_scale(model["scale_models"][component], frame),
            model["calibration_scales"][component],
        )
        for component in ("dry", "wet", "total")
    }
    profile_shape = arrays.target_total.shape
    mean_dry = mean_dry.reshape(profile_shape)
    mean_wet = mean_wet.reshape(profile_shape)
    std_dry = std["dry"].reshape(profile_shape)
    std_wet = std["wet"].reshape(profile_shape)
    std_total = std["total"].reshape(profile_shape)
    correlation = np.asarray(model["correlation_total"], dtype=float)
    covariance = std_total[:, :, None] * correlation[None, :, :] * std_total[:, None, :]
    prediction = ProfilePrediction(
        mean_dry=mean_dry,
        mean_wet=mean_wet,
        mean_total=mean_dry + mean_wet,
        std_dry=std_dry,
        std_wet=std_wet,
        std_total=std_total,
        covariance_total=covariance,
        level_mask=arrays.mask.copy(),
    )
    prediction.validate()
    return prediction


def validate_covariance_masks(prediction: ProfilePrediction, eigen_floor: float = 1e-8) -> dict[str, float]:
    if prediction.covariance_total is None:
        return {"profiles_checked": 0, "minimum_eigenvalue": np.nan}
    minimum = np.inf
    checked = 0
    masks = prediction.level_mask
    patterns = {tuple(row) for row in masks.tolist()}
    for pattern in patterns:
        selected = np.flatnonzero(pattern)
        if not selected.size:
            continue
        indices = np.flatnonzero(np.all(masks == np.asarray(pattern, dtype=bool), axis=1))
        covariance = prediction.covariance_total[indices][:, selected][:, :, selected]
        # NumPy evaluates eigvalsh over the leading batch dimension.  This is
        # materially faster than one decomposition per profile in formal runs.
        eigenvalues = np.linalg.eigvalsh(covariance)
        minimum = min(minimum, float(np.min(eigenvalues)))
        checked += len(indices)
    if checked and minimum <= eigen_floor:
        raise ValueError(f"Predictive covariance is not sufficiently positive definite: min eigenvalue={minimum}")
    return {"profiles_checked": checked, "minimum_eigenvalue": float(minimum) if checked else np.nan}


def covariance_frame(arrays: ProfileArrays, prediction: ProfilePrediction) -> pd.DataFrame:
    if prediction.covariance_total is None:
        return pd.DataFrame()
    result = arrays.keys.copy()
    result["mask_bits"] = [int(sum((1 << index) for index, valid in enumerate(mask) if valid)) for mask in prediction.level_mask]
    for left in range(len(LEVELS)):
        for right in range(left, len(LEVELS)):
            result[f"cov_{int(LEVELS[left])}_{int(LEVELS[right])}"] = prediction.covariance_total[:, left, right]
    return result


def joint_probability_metrics(
    arrays: ProfileArrays,
    prediction: ProfilePrediction,
    confidence_levels: list[float],
    draws: int,
    seed: int,
    chunk_size: int = 2048,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if prediction.covariance_total is None:
        return pd.DataFrame(), pd.DataFrame()
    rng = np.random.default_rng(seed + 2000)
    nll_values: list[float] = []
    energy_values: list[float] = []
    mahalanobis: list[float] = []
    dimensions: list[int] = []
    errors = prediction.mean_total - arrays.target_total
    masks = prediction.level_mask
    patterns: dict[tuple[bool, ...], np.ndarray] = {}
    for pattern in {tuple(row) for row in masks.tolist()}:
        patterns[pattern] = np.flatnonzero(np.all(masks == np.asarray(pattern, dtype=bool), axis=1))
    for pattern, indices in tqdm(patterns.items(), desc="Joint profile probability", leave=False):
        selected_levels = np.flatnonzero(pattern)
        if not selected_levels.size:
            continue
        for start in range(0, len(indices), chunk_size):
            batch = indices[start : start + chunk_size]
            covariance = prediction.covariance_total[batch][:, selected_levels][:, :, selected_levels]
            error = errors[batch][:, selected_levels]
            sign, logdet = np.linalg.slogdet(covariance)
            if np.any(sign <= 0) or not np.isfinite(logdet).all():
                raise ValueError("Non-positive-definite covariance encountered in joint NLL")
            solved = np.linalg.solve(covariance, error[..., None])[..., 0]
            mahal = np.sum(error * solved, axis=1)
            dimension = len(selected_levels)
            nll_values.extend((0.5 * (dimension * np.log(2 * np.pi) + logdet + mahal)).tolist())
            mahalanobis.extend(mahal.tolist())
            dimensions.extend([dimension] * len(batch))
            chol = np.linalg.cholesky(covariance)
            normal_first = rng.standard_normal((len(batch), draws, dimension))
            normal_second = rng.standard_normal((len(batch), draws, dimension))
            first = np.einsum("bij,bdj->bdi", chol, normal_first)
            second = np.einsum("bij,bdj->bdi", chol, normal_second)
            energy = np.linalg.norm(first + error[:, None, :], axis=2).mean(axis=1)
            energy -= 0.5 * np.linalg.norm(first - second, axis=2).mean(axis=1)
            energy_values.extend(energy.tolist())
    joint = pd.DataFrame([{
        "component": "total_profile",
        "profiles": len(nll_values),
        "multivariate_gaussian_nll": float(np.mean(nll_values)),
        "multivariate_gaussian_nll_per_valid_level": float(np.sum(nll_values) / np.sum(dimensions)),
        "profile_energy_score": float(np.mean(energy_values)),
    }])
    mahalanobis_array = np.asarray(mahalanobis)
    dimensions_array = np.asarray(dimensions)
    coverage_rows = []
    for confidence in confidence_levels:
        thresholds = chi2.ppf(float(confidence), dimensions_array)
        coverage_rows.append({
            "component": "total_profile",
            "nominal_coverage": float(confidence),
            "empirical_joint_coverage": float(np.mean(mahalanobis_array <= thresholds)),
            "profiles": len(mahalanobis_array),
        })
    return joint, pd.DataFrame(coverage_rows)


def bootstrap_inference_tables(predictions: pd.DataFrame, replicates: int, seed: int) -> dict[str, pd.DataFrame]:
    valid = predictions.loc[predictions["evaluation_mask"]].copy()
    valid["month"] = pd.to_datetime(valid["time"], utc=True).dt.strftime("%Y-%m")
    valid["model_sq"] = np.square(valid["prediction_total"] - valid["residual_total"])
    valid["era5_sq"] = np.square(valid["residual_total"])
    station_month = valid.groupby(["station_id", "month"], observed=True).agg(model_mse=("model_sq", "mean"), era5_mse=("era5_sq", "mean"))
    station_month["rmse_delta"] = np.sqrt(station_month["model_mse"]) - np.sqrt(station_month["era5_mse"])

    def summarize(values: np.ndarray, samples: np.ndarray, method: str, clusters: int) -> pd.DataFrame:
        return pd.DataFrame([{
            "comparison": "model_minus_era5_rmse",
            "method": method,
            "clusters": int(clusters),
            "mean_difference": float(np.mean(values)),
            "ci_lower": float(np.quantile(samples, 0.025)),
            "ci_upper": float(np.quantile(samples, 0.975)),
            "replicates": int(replicates),
            "seed": int(seed),
        }])

    rng = np.random.default_rng(seed)
    station_direct = valid.groupby("station_id", observed=True).agg(model_mse=("model_sq", "mean"), era5_mse=("era5_sq", "mean"))
    station_values = (np.sqrt(station_direct["model_mse"]) - np.sqrt(station_direct["era5_mse"])).to_numpy(float)
    station_samples = np.asarray([rng.choice(station_values, len(station_values), replace=True).mean() for _ in range(replicates)])
    month_direct = valid.groupby("month", observed=True).agg(model_mse=("model_sq", "mean"), era5_mse=("era5_sq", "mean"))
    month_values = (np.sqrt(month_direct["model_mse"]) - np.sqrt(month_direct["era5_mse"])).to_numpy(float)
    month_samples = np.asarray([rng.choice(month_values, len(month_values), replace=True).mean() for _ in range(replicates)])
    by_station = {station: group.droplevel("station_id")["rmse_delta"].to_numpy(float) for station, group in station_month.groupby(level="station_id")}
    stations = np.asarray(list(by_station), dtype=object)
    hierarchical = np.empty(replicates, dtype=float)
    for index in tqdm(range(replicates), desc="Hierarchical station-month bootstrap", leave=False):
        sampled_stations = rng.choice(stations, len(stations), replace=True)
        station_means = []
        for station in sampled_stations:
            values = by_station[station]
            station_means.append(float(rng.choice(values, len(values), replace=True).mean()))
        hierarchical[index] = float(np.mean(station_means))
    return {
        "station_bootstrap": summarize(station_values, station_samples, "station_cluster", len(station_values)),
        "month_bootstrap": summarize(month_values, month_samples, "month_block", len(month_values)),
        "hierarchical_bootstrap": summarize(station_values, hierarchical, "station_then_month", len(station_values)),
    }
