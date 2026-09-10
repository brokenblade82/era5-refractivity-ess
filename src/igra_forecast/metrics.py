from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_metrics(predictions: pd.DataFrame, metric_names: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    group_cols = ["target", "horizon"]
    for (target, horizon), group in predictions.groupby(group_cols, sort=True):
        y_true = group["y_true"].to_numpy(dtype=float)
        y_pred = group["y_pred"].to_numpy(dtype=float)
        values = _metric_values(y_true, y_pred, metric_names)
        for metric, value in values.items():
            records.append(
                {
                    "scope": "target_horizon",
                    "target": target,
                    "horizon": int(horizon),
                    "metric": metric,
                    "value": float(value),
                    "n": int(len(group)),
                }
            )

    for target, group in predictions.groupby("target", sort=True):
        values = _metric_values(group["y_true"].to_numpy(float), group["y_pred"].to_numpy(float), metric_names)
        for metric, value in values.items():
            records.append(
                {
                    "scope": "target_average",
                    "target": target,
                    "horizon": "all",
                    "metric": metric,
                    "value": float(value),
                    "n": int(len(group)),
                }
            )

    for horizon, group in predictions.groupby("horizon", sort=True):
        values = _metric_values(group["y_true"].to_numpy(float), group["y_pred"].to_numpy(float), metric_names)
        for metric, value in values.items():
            records.append(
                {
                    "scope": "horizon_average",
                    "target": "all",
                    "horizon": int(horizon),
                    "metric": metric,
                    "value": float(value),
                    "n": int(len(group)),
                }
            )

    values = _metric_values(predictions["y_true"].to_numpy(float), predictions["y_pred"].to_numpy(float), metric_names)
    for metric, value in values.items():
        records.append(
            {
                "scope": "overall",
                "target": "all",
                "horizon": "all",
                "metric": metric,
                "value": float(value),
                "n": int(len(predictions)),
            }
        )
    return records


def metrics_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)


def _metric_values(y_true: np.ndarray, y_pred: np.ndarray, metric_names: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    err = y_pred - y_true
    for name in metric_names:
        key = name.lower()
        if key == "mse":
            out["mse"] = float(np.mean(err**2))
        elif key == "rmse":
            out["rmse"] = float(np.sqrt(np.mean(err**2)))
        elif key == "mae":
            out["mae"] = float(np.mean(np.abs(err)))
        elif key == "mape":
            denom = np.maximum(np.abs(y_true), 1e-8)
            out["mape"] = float(np.mean(np.abs(err) / denom) * 100.0)
        else:
            raise ValueError(f"Unsupported metric: {name}")
    return out
