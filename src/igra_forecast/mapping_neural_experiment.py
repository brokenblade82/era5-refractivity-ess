from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from igra_forecast.mapping_neural_data import MappingScalers


def evaluate_network(network, loader, scalers: MappingScalers, device: torch.device) -> dict[str, np.ndarray]:
    collected: dict[str, list[np.ndarray]] = defaultdict(list)
    network.eval()
    with torch.inference_mode():
        for batch in loader:
            device_batch = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
            output = network(device_batch)
            for key in ["residual", "std", "gate"]:
                collected[key].append(output[key].detach().cpu().numpy())
            for key in ["target", "target_index", "min_distance_km", "context_count"]:
                if key in batch:
                    collected[key].append(batch[key].numpy())
    result = {key: np.concatenate(values) for key, values in collected.items()}
    for key in ["residual", "std", "target"]:
        if key in result:
            result[key] = scalers.denormalize_residual(result[key]).astype(np.float32)
    return result


def profiles_to_prediction_frame(
    targets: pd.DataFrame,
    outputs: dict[str, np.ndarray],
    layers: list[int],
    split: str,
    model: str,
    calibration: np.ndarray | None = None,
) -> pd.DataFrame:
    indices = outputs["target_index"].astype(int)
    selected = targets.iloc[indices].reset_index(drop=True)
    residual_pred = outputs["residual"]
    std = outputs["std"]
    if calibration is not None:
        std = std * np.asarray(calibration, dtype=np.float32)[None, :]
    rows: list[dict[str, Any]] = []
    for profile_index, row in selected.iterrows():
        for layer_index, layer in enumerate(layers):
            n_era5 = float(row[f"n_era5_{layer}"])
            residual_true = float(row[f"residual_n_{layer}"])
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "station_id": str(row["station_id"]),
                    "time": pd.Timestamp(row["time"]),
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "layer_hpa": int(layer),
                    "n_era5": n_era5,
                    "n_igra": n_era5 + residual_true,
                    "residual_true": residual_true,
                    "residual_prediction": float(residual_pred[profile_index, layer_index]),
                    "prediction": n_era5 + float(residual_pred[profile_index, layer_index]),
                    "prediction_std": float(std[profile_index, layer_index]),
                    "gate": float(outputs["gate"][profile_index]),
                    "nearest_context_distance_km": float(outputs["min_distance_km"][profile_index]),
                    "context_station_count": float(outputs["context_count"][profile_index]),
                }
            )
    return pd.DataFrame(rows)


def build_metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    station_rows: list[dict[str, Any]] = []
    distance_rows: list[dict[str, Any]] = []
    uncertainty_rows: list[dict[str, Any]] = []
    group_columns = ["model", "split"]
    for keys, group in predictions.groupby(group_columns, sort=True):
        model, split = keys
        metric_rows.append(_metric_row(group, model, split, "overall", "all"))
        for layer, layer_group in group.groupby("layer_hpa", sort=True):
            metric_rows.append(_metric_row(layer_group, model, split, "layer", int(layer)))
        for station, station_group in group.groupby("station_id", sort=True):
            row = _metric_row(station_group, model, split, "station", station)
            row["station_id"] = station
            station_rows.append(row)
        distance_column = "evaluation_distance_km" if "evaluation_distance_km" in group else "nearest_context_distance_km"
        if distance_column in group:
            bands = pd.cut(
                group[distance_column],
                bins=[-np.inf, 250.0, 500.0, 1000.0, np.inf],
                labels=["0-250", "250-500", "500-1000", ">1000"],
            )
            for band, band_group in group.assign(distance_band=bands).groupby("distance_band", observed=True):
                distance_rows.append(_metric_row(band_group, model, split, "distance", str(band)))
        if {"prediction_std", "n_igra", "prediction"}.issubset(group.columns):
            valid = group[np.isfinite(group["prediction_std"]) & (group["prediction_std"] > 0)].copy()
            if not valid.empty:
                error = np.abs(valid["n_igra"].to_numpy() - valid["prediction"].to_numpy())
                std = valid["prediction_std"].to_numpy()
                covered = error <= 1.6448536269514722 * std
                uncertainty_rows.append(
                    {
                        "model": model,
                        "split": split,
                        "n": len(valid),
                        "coverage_90": float(covered.mean()),
                        "mean_interval_width_90": float((2.0 * 1.6448536269514722 * std).mean()),
                        "error_uncertainty_correlation": float(np.corrcoef(error, std)[0, 1]) if len(valid) > 2 else np.nan,
                    }
                )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(station_rows),
        pd.DataFrame(distance_rows),
        pd.DataFrame(uncertainty_rows),
    )


def build_uncertainty_calibration_curve(predictions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate probabilistic predictions beyond a single nominal interval."""
    nominal_to_z = {
        0.50: 0.6744897502,
        0.60: 0.8416212336,
        0.70: 1.0364333895,
        0.80: 1.2815515655,
        0.90: 1.6448536270,
        0.95: 1.9599639845,
    }
    rows: list[dict[str, Any]] = []
    for (model, split), group in predictions.groupby(["model", "split"], sort=True):
        valid = group[
            np.isfinite(group["prediction_std"])
            & (group["prediction_std"] > 0)
            & np.isfinite(group["prediction"])
            & np.isfinite(group["n_igra"])
        ].copy()
        if valid.empty:
            continue
        distance_col = "evaluation_distance_km" if "evaluation_distance_km" in valid else "nearest_context_distance_km"
        valid["distance_band"] = pd.cut(
            valid[distance_col],
            bins=[-np.inf, 250.0, 500.0, 1000.0, np.inf],
            labels=["0-250", "250-500", "500-1000", ">1000"],
        )
        groupings = [("overall", "all", valid)]
        groupings.extend(("layer", str(level), frame) for level, frame in valid.groupby("layer_hpa", sort=True))
        groupings.extend(("distance", str(band), frame) for band, frame in valid.groupby("distance_band", observed=True))
        for group_type, group_name, frame in groupings:
            y = frame["n_igra"].to_numpy(dtype=float)
            mu = frame["prediction"].to_numpy(dtype=float)
            sigma = frame["prediction_std"].to_numpy(dtype=float)
            error = y - mu
            nll = 0.5 * (np.log(2.0 * np.pi * sigma**2) + error**2 / sigma**2)
            for nominal, z in nominal_to_z.items():
                lower, upper = mu - z * sigma, mu + z * sigma
                width = upper - lower
                alpha = 1.0 - nominal
                score = width + (2.0 / alpha) * np.maximum(lower - y, 0.0) + (2.0 / alpha) * np.maximum(y - upper, 0.0)
                rows.append(
                    {
                        "model": model,
                        "split": split,
                        "group_type": group_type,
                        "group_name": group_name,
                        "nominal_coverage": nominal,
                        "empirical_coverage": float(((y >= lower) & (y <= upper)).mean()),
                        "mean_interval_width": float(width.mean()),
                        "mean_interval_score": float(score.mean()),
                        "gaussian_nll": float(nll.mean()),
                        "n": int(len(frame)),
                    }
                )
    return pd.DataFrame(rows)


def paired_station_bootstrap(
    predictions: pd.DataFrame,
    reference: str,
    candidate: str,
    split: str = "test",
    replicates: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    subset = predictions[predictions["split"].eq(split)]
    _, stations, _, _ = build_metric_tables(subset)
    station_rmse = stations.pivot(index="station_id", columns="model", values="rmse").dropna(subset=[reference, candidate])
    delta = station_rmse[candidate].to_numpy() - station_rmse[reference].to_numpy()
    if len(delta) == 0:
        return {"reference": reference, "candidate": candidate, "n_stations": 0}
    rng = np.random.default_rng(seed)
    samples = rng.choice(delta, size=(int(replicates), len(delta)), replace=True).mean(axis=1)
    return {
        "reference": reference,
        "candidate": candidate,
        "split": split,
        "n_stations": len(delta),
        "mean_rmse_delta": float(delta.mean()),
        "ci_lower": float(np.quantile(samples, 0.025)),
        "ci_upper": float(np.quantile(samples, 0.975)),
        "candidate_better_fraction": float((delta < 0).mean()),
    }


def paired_distance_bootstrap(
    predictions: pd.DataFrame,
    reference: str,
    candidate: str,
    split: str = "test",
    replicates: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    subset = predictions[predictions["split"].eq(split)].copy()
    if "evaluation_distance_km" not in subset:
        raise ValueError("evaluation_distance_km is required for common distance-band comparison.")
    subset["distance_band"] = pd.cut(
        subset["evaluation_distance_km"],
        bins=[-np.inf, 250.0, 500.0, 1000.0, np.inf],
        labels=["0-250", "250-500", "500-1000", ">1000"],
    )
    rows = []
    for band, frame in subset.groupby("distance_band", observed=True):
        station_rows = []
        for (station, model), group in frame.groupby(["station_id", "model"]):
            station_rows.append({"station_id": station, "model": model, "rmse": float(np.sqrt(np.mean((group["prediction"] - group["n_igra"]) ** 2)))})
        station_metrics = pd.DataFrame(station_rows)
        pivot = station_metrics.pivot(index="station_id", columns="model", values="rmse")
        if reference not in pivot or candidate not in pivot:
            continue
        paired = pivot[[reference, candidate]].dropna()
        delta = (paired[candidate] - paired[reference]).to_numpy(dtype=float)
        if not len(delta):
            continue
        rng = np.random.default_rng(seed)
        boot = rng.choice(delta, size=(int(replicates), len(delta)), replace=True).mean(axis=1)
        rows.append({
            "split": split,
            "distance_band": str(band),
            "reference": reference,
            "candidate": candidate,
            "n_stations": len(delta),
            "mean_station_rmse_delta": float(delta.mean()),
            "ci_lower": float(np.quantile(boot, 0.025)),
            "ci_upper": float(np.quantile(boot, 0.975)),
            "candidate_better_fraction": float((delta < 0).mean()),
        })
    return pd.DataFrame(rows)


def _metric_row(group: pd.DataFrame, model: str, split: str, group_type: str, group_name: Any) -> dict[str, Any]:
    true = group["n_igra"].to_numpy(dtype=float)
    pred = group["prediction"].to_numpy(dtype=float)
    return {
        "model": model,
        "split": split,
        "group_type": group_type,
        "group_name": group_name,
        "layer_hpa": int(group_name) if group_type == "layer" else "all",
        "n": len(group),
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "mae": float(mean_absolute_error(true, pred)),
   "bias": float(np.mean(pred - true)),
        "r2": float(r2_score(true, pred)) if len(group) > 1 else np.nan,
    }
