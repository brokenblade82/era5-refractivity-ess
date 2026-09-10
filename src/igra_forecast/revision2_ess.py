from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import pyarrow.dataset as pads
from scipy.stats import norm

from .revision2_baselines import (
    LEVELS,
    make_row_features,
    predict_component_regressors,
    rows_to_profiles,
)
from .revision2_confirmatory import predict_confirmatory_probability_hgb
from .revision2_data import macro_region, sha256_file, stable_fingerprint


ESS_SCHEMA_VERSION = 1
ESS_MODELS = ("seasonal_mean", "ridge", "hgb")
CORE_ROW_COLUMNS = [
    "station_id", "time", "latitude", "longitude", "station_elevation_m",
    "pressure_hpa", "level_mask", "below_ground_level", "igra_height_m",
    "era5_temperature_k", "era5_specific_humidity", "era5_height_m",
    "era5_surface_pressure_pa", "era5_n_dry", "era5_n_wet", "era5_n",
    "residual_dry", "residual_wet", "residual_total",
]


@dataclass(frozen=True)
class FrozenRun:
    name: str
    path: Path
    model_path: Path | None
    manifest: dict[str, Any]
    model_sha256: str | None


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def frozen_run(name: str, path: str | Path, require_model: bool = True) -> FrozenRun:
    run = Path(path)
    manifest_path = run / "protocol_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Frozen run lacks protocol_manifest.json: {run}")
    manifest = read_json(manifest_path)
    if not manifest.get("complete") or manifest.get("smoke"):
        raise ValueError(f"ESS requires a complete non-smoke frozen run: {run}")
    if manifest.get("protocol") != "space_time_holdout":
        raise ValueError(f"ESS external inference is licensed only from space_time_holdout: {run}")
    model_path = run / "model.joblib"
    if require_model and not model_path.is_file():
        raise FileNotFoundError(f"Frozen run lacks model.joblib: {run}")
    return FrozenRun(
        name=name,
        path=run,
        model_path=model_path if model_path.is_file() else None,
        manifest=manifest,
        model_sha256=sha256_file(model_path) if model_path.is_file() else None,
    )


def build_seasonal_lookup(train: pd.DataFrame) -> dict[str, Any]:
    work = train.copy()
    utc = pd.DatetimeIndex(pd.to_datetime(work["time"], utc=True))
    work["month_of_year"] = utc.month
    work["utc_hour"] = utc.hour
    valid = (
        work["level_mask"].astype(bool)
        & ~work["below_ground_level"].astype(bool)
        & np.isfinite(work["residual_dry"])
        & np.isfinite(work["residual_wet"])
    )
    keys = ["pressure_hpa", "month_of_year", "utc_hour"]
    lookup = work.loc[valid].groupby(keys, observed=True)[["residual_dry", "residual_wet"]].mean()
    fallback = work.loc[valid].groupby("pressure_hpa", observed=True)[["residual_dry", "residual_wet"]].mean()
    if lookup.empty or fallback.empty:
        raise ValueError("Seasonal-mean reconstruction received no valid training rows")
    return {"lookup": lookup, "fallback": fallback}


def predict_seasonal_lookup(model: dict[str, Any], target: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    work = target.copy()
    utc = pd.DatetimeIndex(pd.to_datetime(work["time"], utc=True))
    work["month_of_year"] = utc.month
    work["utc_hour"] = utc.hour
    keys = ["pressure_hpa", "month_of_year", "utc_hour"]
    index = pd.MultiIndex.from_frame(work[keys])
    values = model["lookup"].reindex(index).reset_index(drop=True)
    missing = values["residual_dry"].isna() | values["residual_wet"].isna()
    if missing.any():
        fallback = model["fallback"].reindex(work.loc[missing, "pressure_hpa"].astype(int)).to_numpy()
        values.loc[missing, ["residual_dry", "residual_wet"]] = fallback
    if values[["residual_dry", "residual_wet"]].isna().any().any():
        raise ValueError("Seasonal-mean lookup cannot cover one or more target pressure levels")
    return values["residual_dry"].to_numpy(float), values["residual_wet"].to_numpy(float)


def load_protocol_training_rows(
    input_root: str | Path,
    station_split: str | Path,
    start: str = "2024-01-01",
    end: str = "2025-01-01",
) -> pd.DataFrame:
    dataset = pads.dataset(input_root, format="parquet", partitioning="hive")
    split = pd.read_csv(station_split, dtype={"station_id": str})
    allowed = set(split.loc[split["split"] == "train", "station_id"].astype(str))
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    frames: list[pd.DataFrame] = []
    for period in pd.period_range(start_ts.tz_localize(None), (end_ts - pd.Timedelta(seconds=1)).tz_localize(None), freq="M"):
        table = dataset.to_table(
            columns=CORE_ROW_COLUMNS,
            filter=(pads.field("year") == period.year) & (pads.field("month") == period.month),
        )
        part = table.to_pandas()
        time = pd.DatetimeIndex(pd.to_datetime(part["time"], utc=True))
        part = part.loc[(time >= start_ts) & (time < end_ts) & part["station_id"].astype(str).isin(allowed)]
        if not part.empty:
            frames.append(part)
    if not frames:
        raise ValueError("No licensed 2024 training rows were found for seasonal-mean reconstruction")
    return pd.concat(frames, ignore_index=True)


def validate_seasonal_reconstruction(
    lookup: dict[str, Any], frozen_prediction_path: str | Path, atol: float = 1e-10
) -> dict[str, Any]:
    frozen = pd.read_parquet(frozen_prediction_path)
    dry, wet = predict_seasonal_lookup(lookup, frozen)
    dry_error = float(np.max(np.abs(dry - frozen["prediction_dry"].to_numpy(float))))
    wet_error = float(np.max(np.abs(wet - frozen["prediction_wet"].to_numpy(float))))
    if max(dry_error, wet_error) > atol:
        raise ValueError(
            "Reconstructed seasonal mean does not reproduce the frozen prediction: "
            f"dry={dry_error}, wet={wet_error}, tolerance={atol}"
        )
    return {"rows": int(len(frozen)), "max_abs_dry": dry_error, "max_abs_wet": wet_error, "tolerance": atol}


def predict_row_model(
    name: str,
    frame: pd.DataFrame,
    model: Any,
) -> tuple[np.ndarray, np.ndarray]:
    if name == "seasonal_mean":
        return predict_seasonal_lookup(model, frame)
    if name in {"ridge", "hgb"}:
        return predict_component_regressors(model, frame)
    raise ValueError(f"Unsupported ESS deterministic model: {name}")


def predict_structured_hgb(model: dict[str, Any], frame: pd.DataFrame):
    return predict_confirmatory_probability_hgb(model, rows_to_profiles(frame))


def add_common_groups(frame: pd.DataFrame, wet_edges: Iterable[float] | None = None) -> pd.DataFrame:
    result = frame.copy()
    result["time"] = pd.to_datetime(result["time"], utc=True)
    result["year"] = result["time"].dt.year.astype(str)
    result["month"] = result["time"].dt.strftime("%Y-%m")
    result["date"] = result["time"].dt.strftime("%Y-%m-%d")
    result["season"] = result["time"].dt.month.map(
        {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
         6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
    )
    result["macro_region"] = result["latitude"].map(macro_region)
    if wet_edges is not None:
        edges = np.asarray(list(wet_edges), dtype=float)
        edges[0], edges[-1] = -np.inf, np.inf
        wet_column = "era5_n_wet_height" if "era5_n_wet_height" in result else "era5_n_wet"
        result["wet_regime"] = pd.cut(
            result[wet_column], edges, labels=["Q1", "Q2", "Q3", "Q4"], include_lowest=True
        )
    return result


def training_wet_edges(train: pd.DataFrame) -> np.ndarray:
    valid = (
        train["level_mask"].astype(bool)
        & ~train["below_ground_level"].astype(bool)
        & np.isfinite(train["era5_n_wet"])
    )
    edges = train.loc[valid, "era5_n_wet"].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_numpy(float)
    if not np.all(np.diff(edges) > 0):
        raise ValueError("Training wet-refractivity quartile edges are not strictly increasing")
    return edges


def constant_feature_support(
    features: np.ndarray,
    training_mean: np.ndarray,
    training_std: np.ndarray,
    constant_tolerance: float = 1e-8,
    deviation_tolerance: float = 1e-6,
) -> np.ndarray:
    """Audit whether targets preserve features that were constant in training.

    This does not judge general extrapolation.  It specifically catches an
    ill-posed standardized linear prediction when a training feature has
    effectively zero variance but takes a different value at evaluation time.
    """
    values = np.asarray(features, dtype=float)
    mean = np.asarray(training_mean, dtype=float)
    std = np.asarray(training_std, dtype=float)
    constant = std < float(constant_tolerance)
    if not constant.any():
        return np.ones(len(values), dtype=bool)
    deviation = np.abs(values[:, constant] - mean[constant])
    return np.all(deviation <= float(deviation_tolerance), axis=1)


def apportion_total_refractivity(total: np.ndarray, wet_fraction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Preserve the frozen log-interpolated total while adding a wet-regime covariate."""
    total = np.asarray(total, dtype=float)
    fraction = np.asarray(wet_fraction, dtype=float)
    valid = np.isfinite(total) & np.isfinite(fraction)
    fraction = np.where(valid, np.clip(fraction, 0.0, 1.0), np.nan)
    wet = np.where(valid, total * fraction, np.nan)
    dry = np.where(valid, total - wet, np.nan)
    return dry, wet


def correction_decomposition(residual: np.ndarray, correction: np.ndarray) -> dict[str, float]:
    residual = np.asarray(residual, dtype=float)
    correction = np.asarray(correction, dtype=float)
    valid = np.isfinite(residual) & np.isfinite(correction)
    residual = residual[valid]
    correction = correction[valid]
    if not len(residual):
        return {key: math.nan for key in (
            "n", "era5_mse", "corrected_mse", "delta_mse", "correction_penalty",
            "alignment_term", "algebraic_delta_mse", "closure_error", "bias_squared_delta",
            "centered_variance_delta", "rmse_difference",
        )}
    era5_error = -residual
    corrected_error = correction - residual
    era5_mse = float(np.mean(np.square(era5_error)))
    corrected_mse = float(np.mean(np.square(corrected_error)))
    penalty = float(np.mean(np.square(correction)))
    alignment = float(-2.0 * np.mean(residual * correction))
    direct = corrected_mse - era5_mse
    algebraic = penalty + alignment
    bias_delta = float(np.square(corrected_error.mean()) - np.square(era5_error.mean()))
    variance_delta = float(np.var(corrected_error) - np.var(era5_error))
    return {
        "n": int(len(residual)),
        "era5_mse": era5_mse,
        "corrected_mse": corrected_mse,
        "delta_mse": direct,
        "correction_penalty": penalty,
        "alignment_term": alignment,
        "algebraic_delta_mse": algebraic,
        "closure_error": direct - algebraic,
        "bias_squared_delta": bias_delta,
        "centered_variance_delta": variance_delta,
        "rmse_difference": math.sqrt(corrected_mse) - math.sqrt(era5_mse),
    }


def probability_diagnostics(error: np.ndarray, std: np.ndarray) -> dict[str, float]:
    error = np.asarray(error, dtype=float)
    std = np.asarray(std, dtype=float)
    valid = np.isfinite(error) & np.isfinite(std) & (std > 0)
    error, std = error[valid], std[valid]
    if not len(error):
        return {"n": 0}
    z = error / std
    pit = norm.cdf(z)
    z90 = float(norm.ppf(0.95))
    required_scale = float(np.quantile(np.abs(z), 0.90) / z90)
    abs_error = np.abs(error)
    correlation = (
        float(np.corrcoef(std, abs_error)[0, 1])
        if len(error) > 1 and float(np.std(std)) > 0 and float(np.std(abs_error)) > 0
        else math.nan
    )
    return {
        "n": int(len(z)),
        "z_mean": float(z.mean()),
        "z_std": float(z.std()),
        "z_median_abs": float(np.median(np.abs(z))),
        "tail_abs_gt_1p645": float((np.abs(z) > 1.645).mean()),
        "tail_abs_gt_1p96": float((np.abs(z) > 1.96).mean()),
        "tail_abs_gt_3": float((np.abs(z) > 3.0).mean()),
        "pit_mean": float(pit.mean()),
        "pit_variance": float(pit.var()),
        "coverage_90": float((np.abs(z) <= z90).mean()),
        "required_scale_for_90": required_scale,
        "mean_predictive_sd": float(std.mean()),
        "mean_absolute_error": float(abs_error.mean()),
        "sd_abs_error_correlation": correlation,
    }


def pit_histogram(error: np.ndarray, std: np.ndarray, bins: int = 20) -> pd.DataFrame:
    error = np.asarray(error, dtype=float)
    std = np.asarray(std, dtype=float)
    valid = np.isfinite(error) & np.isfinite(std) & (std > 0)
    pit = norm.cdf(error[valid] / std[valid])
    counts, edges = np.histogram(pit, bins=np.linspace(0.0, 1.0, bins + 1))
    return pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "count": counts})


def cosmic_sampling_checks(
    full_delta: float,
    leave_one_delta: np.ndarray,
    leave_one_departure: np.ndarray,
    height_band_ci: np.ndarray,
    archive_row_fraction: np.ndarray,
    year_delta: np.ndarray,
    thresholds: dict[str, Any],
) -> dict[str, bool]:
    """Apply the preregistered COSMIC archive-expansion checks deterministically."""
    leave_one = np.asarray(leave_one_delta, dtype=float)
    leave_one = leave_one[np.isfinite(leave_one)]
    departure = np.asarray(leave_one_departure, dtype=float)
    departure = departure[np.isfinite(departure)]
    intervals = np.asarray(height_band_ci, dtype=float).reshape(-1, 2)
    fractions = np.asarray(archive_row_fraction, dtype=float)
    years = np.asarray(year_delta, dtype=float)
    limit = max(
        float(thresholds["leave_one_archive_absolute_threshold_n_units"]),
        float(thresholds["leave_one_archive_relative_threshold"]) * abs(float(full_delta)),
    )
    finite_years = years[np.isfinite(years)]
    return {
        "leave_one_archive_sign_change": bool(
            leave_one.size == 0 or (np.sign(leave_one) != np.sign(float(full_delta))).any()
        ),
        "leave_one_archive_excess_departure": bool((np.abs(departure) > limit).any()),
        "primary_height_band_ci_crosses_zero": bool(
            intervals.size and ((intervals[:, 0] <= 0) & (intervals[:, 1] >= 0)).any()
        ),
        "archive_row_fraction_exceeds_limit": bool(
            fractions.size and np.nanmax(fractions) > float(thresholds["maximum_archive_row_fraction"])
        ),
        "year_sign_change": bool(len(set(np.sign(finite_years).tolist())) > 1),
        "year_difference_exceeds_limit": bool(
            finite_years.size > 1
            and float(np.max(finite_years) - np.min(finite_years))
            > float(thresholds["maximum_year_difference_n_units"])
        ),
    }


def cluster_bootstrap_decomposition(
    frame: pd.DataFrame,
    residual_column: str,
    correction_column: str,
    cluster_columns: list[str],
    replicates: int,
    seed: int,
) -> dict[str, float]:
    work = frame[cluster_columns + [residual_column, correction_column]].dropna().copy()
    residual = work[residual_column].to_numpy(float)
    correction = work[correction_column].to_numpy(float)
    work["n"] = 1
    work["r2"] = np.square(residual)
    work["m2"] = np.square(correction - residual)
    work["c2"] = np.square(correction)
    work["rc"] = residual * correction
    grouped = work.groupby(cluster_columns, observed=True)[["n", "r2", "m2", "c2", "rc"]].sum()
    values = grouped.to_numpy(float)
    if not len(values):
        return {"clusters": 0, "replicates": int(replicates)}
    rng = np.random.default_rng(seed)
    sampled = np.empty(int(replicates), dtype=float)
    for index in range(int(replicates)):
        selected = values[rng.integers(0, len(values), size=len(values))].sum(axis=0)
        n, r2, m2, _, _ = selected
        sampled[index] = math.sqrt(m2 / n) - math.sqrt(r2 / n)
    point = correction_decomposition(work[residual_column], work[correction_column])
    return {
        "clusters": int(len(values)),
        "mean_difference": float(point["rmse_difference"]),
        "ci_lower": float(np.quantile(sampled, 0.025)),
        "ci_upper": float(np.quantile(sampled, 0.975)),
        "replicates": int(replicates),
        "seed": int(seed),
    }


def ess_fingerprint(config: dict[str, Any], frozen: dict[str, FrozenRun]) -> str:
    return stable_fingerprint({
        "schema_version": ESS_SCHEMA_VERSION,
        "config": config,
        "models": {name: item.model_sha256 for name, item in frozen.items()},
    })
