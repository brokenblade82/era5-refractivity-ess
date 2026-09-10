from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy.stats import norm

from .revision2_publication import cluster_bootstrap_mean, component_error_budget, normal_crps


NOMINAL_COVERAGES = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
ERROR_BUDGET_TERMS = (
    "dry_mse", "wet_mse", "cross_term", "total_mse",
    "total_bias_squared", "total_centered_variance",
)


def add_cosmic_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[frame["evaluation_mask"]].copy()
    result["era5_error"] = result["era5_n_height"] - result["observed_n"]
    result["hgb_error"] = result["prediction_n_height"] - result["observed_n"]
    result["height_km"] = result["height_m"] / 1000.0
    result["height_band"] = pd.cut(
        result["height_km"], [0.5, 2.0, 5.0, 9.0],
        labels=["0.5-2", "2-5", "5-9"], include_lowest=True,
    )
    result["macro_region"] = np.where(
        result["latitude"] >= 30.0, "northern_extratropics",
        np.where(result["latitude"] <= -30.0, "southern_extratropics", "tropics"),
    )
    result["archive_month"] = pd.to_datetime(result["time"], utc=True).dt.strftime("%Y-%m")
    return result


def deterministic_probability_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    era5 = frame["era5_error"].to_numpy(float)
    hgb = frame["hgb_error"].to_numpy(float)
    std = frame["prediction_std_height"].to_numpy(float)
    valid = np.isfinite(era5) & np.isfinite(hgb) & np.isfinite(std) & (std > 0)
    era5, hgb, std = era5[valid], hgb[valid], std[valid]
    if len(hgb) == 0:
        return {"n": 0}
    alpha, z90 = 0.10, norm.ppf(0.95)
    interval_width = 2.0 * z90 * std
    interval_score = interval_width + (2.0 / alpha) * (
        (np.abs(hgb) - z90 * std) * (np.abs(hgb) > z90 * std)
    )
    row: dict[str, float | int] = {
        "n": int(len(hgb)),
        "era5_bias": float(np.mean(era5)),
        "hgb_bias": float(np.mean(hgb)),
        "era5_mae": float(np.mean(np.abs(era5))),
        "hgb_mae": float(np.mean(np.abs(hgb))),
        "era5_rmse": float(np.sqrt(np.mean(np.square(era5)))),
        "hgb_rmse": float(np.sqrt(np.mean(np.square(hgb)))),
        "hgb_minus_era5_rmse": float(np.sqrt(np.mean(np.square(hgb))) - np.sqrt(np.mean(np.square(era5)))),
        "mean_correction": float(np.mean(frame.loc[valid, "prediction_correction"])),
        "mean_predictive_sd": float(np.mean(std)),
        "crps": float(np.mean(normal_crps(hgb, std))),
        "interval_score_90": float(np.mean(interval_score)),
        "mean_interval_width_90": float(np.mean(interval_width)),
    }
    for nominal in NOMINAL_COVERAGES:
        z = norm.ppf((1.0 + nominal) / 2.0)
        row[f"coverage_{int(nominal * 100)}"] = float(np.mean(np.abs(hgb) <= z * std))
    return row


def grouped_cosmic_metrics(frame: pd.DataFrame, grouping: str, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouper: str | list[str] = columns[0] if len(columns) == 1 else columns
    for key, selected in frame.groupby(grouper, observed=True, dropna=False):
        keys = (key,) if len(columns) == 1 else tuple(key)
        row: dict[str, object] = {"grouping": grouping}
        row.update(dict(zip(columns, keys, strict=True)))
        row.update(deterministic_probability_metrics(selected))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_rmse_by_unit(frame: pd.DataFrame, unit: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, selected in frame.groupby(unit, observed=True):
        era5 = selected["era5_error"].to_numpy(float)
        hgb = selected["hgb_error"].to_numpy(float)
        rows.append({
            unit: key,
            "n": int(len(selected)),
            "rmse_difference": float(np.sqrt(np.mean(np.square(hgb))) - np.sqrt(np.mean(np.square(era5)))),
        })
    return pd.DataFrame(rows)


def archive_block_bootstrap(
    frame: pd.DataFrame,
    grouping: str,
    columns: list[str],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouper: str | list[str] = columns[0] if len(columns) == 1 else columns
    for key, selected in frame.groupby(grouper, observed=True, dropna=False):
        keys = (key,) if len(columns) == 1 else tuple(key)
        units = paired_rmse_by_unit(selected, "archive_month")
        inference = cluster_bootstrap_mean(units["rmse_difference"].to_numpy(float), replicates, seed)
        row: dict[str, object] = {"grouping": grouping, "bootstrap_unit": "archive_month"}
        row.update(dict(zip(columns, keys, strict=True)))
        row.update(inference)
        rows.append(row)
    return pd.DataFrame(rows)


def physical_budget_differences(frame: pd.DataFrame) -> dict[str, float]:
    era5 = component_error_budget(-frame["residual_dry"], -frame["residual_wet"])
    hgb = component_error_budget(
        frame["prediction_dry"] - frame["residual_dry"],
        frame["prediction_wet"] - frame["residual_wet"],
    )
    result = {f"era5_{key}": value for key, value in asdict(era5).items()}
    result.update({f"hgb_{key}": value for key, value in asdict(hgb).items()})
    result.update({f"hgb_minus_era5_{term}": getattr(hgb, term) - getattr(era5, term) for term in ERROR_BUDGET_TERMS})
    result["era5_reconstruction_error"] = era5.reconstructed_total_mse - era5.total_mse
    result["hgb_reconstruction_error"] = hgb.reconstructed_total_mse - hgb.total_mse
    return result


def physical_station_bootstrap(
    frame: pd.DataFrame,
    grouping: str,
    column: str,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group, selected in frame.groupby(column, observed=True, dropna=False):
        point = physical_budget_differences(selected)
        station_rows = []
        for station, station_frame in selected.groupby("station_id", observed=True):
            values = physical_budget_differences(station_frame)
            station_rows.append({"station_id": station, **{term: values[f"hgb_minus_era5_{term}"] for term in ERROR_BUDGET_TERMS}})
        station = pd.DataFrame(station_rows)
        for term in ERROR_BUDGET_TERMS:
            inference = cluster_bootstrap_mean(station[term].to_numpy(float), replicates, seed)
            rows.append({
                "grouping": grouping, "group": group, "term": term,
                "samples": int(len(selected)), "stations": int(len(station)),
                "pooled_difference": point[f"hgb_minus_era5_{term}"],
                "bootstrap_unit": "station", **inference,
                "maximum_reconstruction_error": max(
                    abs(point["era5_reconstruction_error"]), abs(point["hgb_reconstruction_error"])
                ),
            })
    return pd.DataFrame(rows)
