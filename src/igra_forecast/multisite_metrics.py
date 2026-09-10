from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_multisite_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(["layer_hpa", "horizon_hours"], sort=True):
        layer, horizon = keys
        records.extend(metric_rows(group, "layer_horizon", int(layer), int(horizon)))
    for layer, group in predictions.groupby("layer_hpa", sort=True):
        records.extend(metric_rows(group, "layer_average", int(layer), "all"))
    for horizon, group in predictions.groupby("horizon_hours", sort=True):
        records.extend(metric_rows(group, "horizon_average", "all", int(horizon)))
    for station, group in predictions.groupby("station_id", sort=True):
        for row in metric_rows(group, "station_average", "all", "all"):
            row["station_id"] = station
            records.append(row)
    records.extend(metric_rows(predictions, "overall", "all", "all"))
    return pd.DataFrame.from_records(records)


def metric_rows(group: pd.DataFrame, scope: str, layer: int | str, horizon: int | str) -> list[dict[str, Any]]:
    y_true = group["y_true"].to_numpy(dtype=float)
    y_pred = group["y_pred"].to_numpy(dtype=float)
    err = y_pred - y_true
    denom = np.maximum(np.abs(y_true), 1e-8)
    values = {
        "mse": float(np.mean(err**2)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "mape": float(np.mean(np.abs(err) / denom) * 100.0),
        "bias": float(np.mean(err)),
    }
    return [
        {
            "scope": scope,
            "layer_hpa": layer,
            "horizon_hours": horizon,
            "metric": metric,
            "value": value,
            "n": int(len(group)),
        }
        for metric, value in values.items()
    ]
