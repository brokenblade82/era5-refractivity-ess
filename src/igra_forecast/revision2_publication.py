from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


HEIGHT_LEVELS_M = np.arange(500.0, 9000.0 + 0.1, 500.0)


def cosmic_official_qc_passes(attributes: dict[str, object]) -> tuple[bool, str]:
    try:
        bad = int(float(attributes.get("bad")))
    except (TypeError, ValueError):
        return False, "missing_bad_flag"
    if bad != 0:
        return False, "bad_flag_nonzero"
    return True, "retained"


def validate_geometric_height_schema(frame: pd.DataFrame) -> None:
    required = {"profile_id", "time", "height_m", "height_mask", "latitude", "longitude", "observed_n"}
    missing = required - set(frame.columns)
    if frame.empty or missing:
        raise ValueError(f"Geometric-height input is empty or incomplete; missing={sorted(missing)}")
    forbidden = {"Pres", "pres", "dry_pressure_hpa"} & set(frame.columns)
    if forbidden:
        raise ValueError(f"Dry-pressure columns are forbidden in the primary geometric-height contract: {sorted(forbidden)}")


def geometric_height_evaluation_mask(
    observed_n: np.ndarray,
    target_height_m: np.ndarray,
    surface_height_m: np.ndarray,
    era5_n: np.ndarray,
    correction: np.ndarray,
    variance: np.ndarray,
    input_mask: np.ndarray,
    correction_support: np.ndarray,
) -> np.ndarray:
    observed_n = np.asarray(observed_n, dtype=float)
    target_height_m = np.asarray(target_height_m, dtype=float)
    surface_height_m = np.asarray(surface_height_m, dtype=float)
    era5_n = np.asarray(era5_n, dtype=float)
    correction = np.asarray(correction, dtype=float)
    variance = np.asarray(variance, dtype=float)
    return (
        np.asarray(input_mask, dtype=bool)
        & np.asarray(correction_support, dtype=bool)
        & (target_height_m > surface_height_m)
        & np.isfinite(observed_n)
        & np.isfinite(era5_n)
        & np.isfinite(correction)
        & np.isfinite(variance)
        & (variance >= 0)
    )


def _finite_sorted_coordinates(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    order = np.argsort(x[valid])
    x_sorted = x[valid][order]
    y_sorted = y[valid][order]
    x_unique, unique_index = np.unique(x_sorted, return_index=True)
    return x_unique, y_sorted[unique_index]


def interpolate_no_extrapolation(
    source_coordinate: np.ndarray,
    source_values: np.ndarray,
    target_coordinate: np.ndarray,
    *,
    log_values: bool = False,
) -> np.ndarray:
    """One-dimensional interpolation with an explicit no-extrapolation contract."""
    x, y = _finite_sorted_coordinates(source_coordinate, source_values)
    target = np.asarray(target_coordinate, dtype=float)
    result = np.full(target.shape, np.nan, dtype=float)
    if len(x) < 2:
        return result
    if log_values:
        positive = y > 0
        x, y = x[positive], y[positive]
        if len(x) < 2:
            return result
        y = np.log(y)
    valid_target = np.isfinite(target) & (target >= x[0]) & (target <= x[-1])
    result[valid_target] = np.interp(target[valid_target], x, y)
    if log_values:
        result[valid_target] = np.exp(result[valid_target])
    return result


def interpolate_longitude_no_extrapolation(
    source_coordinate: np.ndarray,
    longitude_degrees: np.ndarray,
    target_coordinate: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_coordinate, dtype=float)
    longitude = np.asarray(longitude_degrees, dtype=float)
    valid = np.isfinite(source) & np.isfinite(longitude)
    if valid.sum() < 2:
        return np.full(np.asarray(target_coordinate).shape, np.nan)
    order = np.argsort(source[valid])
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(longitude[valid][order])))
    interpolated = interpolate_no_extrapolation(source[valid][order], unwrapped, target_coordinate)
    return (interpolated + 180.0) % 360.0 - 180.0


def interpolation_weights(
    source_coordinate: np.ndarray,
    target_coordinate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return linear weights and target validity without extrapolation.

    Source coordinates must be finite and strictly unique after sorting.  The
    returned columns follow the original source order, which lets callers apply
    the weights directly to means and covariance matrices.
    """
    source = np.asarray(source_coordinate, dtype=float).reshape(-1)
    target = np.asarray(target_coordinate, dtype=float).reshape(-1)
    if not np.isfinite(source).all():
        raise ValueError("Interpolation source coordinates must be finite")
    order = np.argsort(source)
    sorted_source = source[order]
    if len(np.unique(sorted_source)) != len(sorted_source):
        raise ValueError("Interpolation source coordinates must be unique")
    weights_sorted = np.zeros((len(target), len(source)), dtype=float)
    valid = np.isfinite(target) & (target >= sorted_source[0]) & (target <= sorted_source[-1])
    for row in np.flatnonzero(valid):
        value = target[row]
        right = int(np.searchsorted(sorted_source, value, side="left"))
        if right == 0:
            weights_sorted[row, 0] = 1.0
        elif right == len(sorted_source):
            weights_sorted[row, -1] = 1.0
        elif sorted_source[right] == value:
            weights_sorted[row, right] = 1.0
        else:
            left = right - 1
            fraction = (value - sorted_source[left]) / (sorted_source[right] - sorted_source[left])
            weights_sorted[row, left] = 1.0 - fraction
            weights_sorted[row, right] = fraction
    inverse = np.argsort(order)
    return weights_sorted[:, inverse], valid


def propagate_gaussian_to_height(
    source_coordinate: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    target_coordinate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights, valid = interpolation_weights(source_coordinate, target_coordinate)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (len(mean), len(mean)) or weights.shape[1] != len(mean):
        raise ValueError("Mean/covariance dimensions do not match interpolation sources")
    target_mean = weights @ mean
    target_variance = np.einsum("ti,ij,tj->t", weights, covariance, weights)
    target_mean[~valid] = np.nan
    target_variance[~valid] = np.nan
    target_variance[valid] = np.maximum(target_variance[valid], 0.0)
    return target_mean, target_variance, valid


def normal_crps(error: np.ndarray, std: np.ndarray) -> np.ndarray:
    from scipy.special import ndtr

    error = np.asarray(error, dtype=float)
    std = np.maximum(np.asarray(std, dtype=float), 1e-12)
    z = error / std
    return std * (
        z * (2 * ndtr(z) - 1)
        + 2 * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
        - 1 / np.sqrt(np.pi)
    )


def paired_rmse_units(
    frame: pd.DataFrame,
    unit_columns: Iterable[str],
    *,
    model_error: str = "model_error",
    era5_error: str = "era5_error",
) -> pd.DataFrame:
    columns = list(unit_columns)
    work = frame.loc[
        np.isfinite(frame[model_error]) & np.isfinite(frame[era5_error]),
        [*columns, model_error, era5_error],
    ].copy()
    work["model_sq"] = np.square(work[model_error])
    work["era5_sq"] = np.square(work[era5_error])
    grouped = work.groupby(columns, observed=True).agg(
        n=(model_error, "size"), model_mse=("model_sq", "mean"), era5_mse=("era5_sq", "mean")
    ).reset_index()
    grouped["model_rmse"] = np.sqrt(grouped["model_mse"])
    grouped["era5_rmse"] = np.sqrt(grouped["era5_mse"])
    grouped["rmse_difference"] = grouped["model_rmse"] - grouped["era5_rmse"]
    return grouped


def cluster_bootstrap_mean(
    values: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"clusters": 0, "mean_difference": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
                "replicates": int(replicates), "seed": int(seed)}
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(int(replicates), dtype=float)
    for index in range(int(replicates)):
        bootstrap[index] = rng.choice(values, size=len(values), replace=True).mean()
    return {
        "clusters": int(len(values)),
        "mean_difference": float(values.mean()),
        "ci_lower": float(np.quantile(bootstrap, 0.025)),
        "ci_upper": float(np.quantile(bootstrap, 0.975)),
        "replicates": int(replicates),
        "seed": int(seed),
    }


@dataclass(frozen=True)
class ErrorBudget:
    samples: int
    dry_mse: float
    wet_mse: float
    cross_term: float
    total_mse: float
    reconstructed_total_mse: float
    total_bias_squared: float
    total_centered_variance: float


def component_error_budget(dry_error: np.ndarray, wet_error: np.ndarray) -> ErrorBudget:
    dry = np.asarray(dry_error, dtype=float)
    wet = np.asarray(wet_error, dtype=float)
    valid = np.isfinite(dry) & np.isfinite(wet)
    dry, wet = dry[valid], wet[valid]
    if not len(dry):
        return ErrorBudget(0, *(np.nan for _ in range(7)))
    total = dry + wet
    dry_mse = float(np.mean(np.square(dry)))
    wet_mse = float(np.mean(np.square(wet)))
    cross = float(2.0 * np.mean(dry * wet))
    total_mse = float(np.mean(np.square(total)))
    return ErrorBudget(
        samples=int(len(total)),
        dry_mse=dry_mse,
        wet_mse=wet_mse,
        cross_term=cross,
        total_mse=total_mse,
        reconstructed_total_mse=dry_mse + wet_mse + cross,
        total_bias_squared=float(np.square(total.mean())),
        total_centered_variance=float(np.var(total)),
    )
