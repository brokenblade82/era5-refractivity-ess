from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from igra_forecast.mapping_neural_data import spherical_distance_and_bearing
from igra_forecast.refractivity_mapping import ResidualMappingModel, default_feature_columns


def mean_residual_prediction(train: pd.DataFrame, targets: pd.DataFrame) -> np.ndarray:
    work = train.copy()
    work["month"] = pd.to_datetime(work["time"]).dt.month
    work["hour"] = pd.to_datetime(work["time"]).dt.hour
    lookup = work.groupby(["layer_hpa", "month", "hour"])["residual_n"].mean().to_dict()
    layer_lookup = work.groupby("layer_hpa")["residual_n"].mean().to_dict()
    global_mean = float(work["residual_n"].mean())
    times = pd.to_datetime(targets["time"])
    values = [
        lookup.get((int(layer), int(time.month), int(time.hour)), layer_lookup.get(int(layer), global_mean))
        for layer, time in zip(targets["layer_hpa"], times)
    ]
    return np.asarray(values, dtype=np.float32)


def ridge_residual_prediction(train: pd.DataFrame, targets: pd.DataFrame, alpha: float = 10.0) -> np.ndarray:
    features = default_feature_columns(train)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha), solver="lsqr")),
        ]
    )
    model.fit(train[features].to_numpy(dtype=np.float32), train["residual_n"].to_numpy(dtype=np.float32))
    return model.predict(targets[features].to_numpy(dtype=np.float32)).astype(np.float32)


def train_ridge_residual_model(train: pd.DataFrame, alpha: float = 1.0) -> ResidualMappingModel:
    """Fit a standardized Ridge residual prior with the HGB-compatible API."""
    features = default_feature_columns(train)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha), solver="lsqr")),
        ]
    )
    model.fit(train[features].to_numpy(dtype=np.float32), train["residual_n"].to_numpy(dtype=np.float32))
    levels = sorted(pd.to_numeric(train["layer_hpa"], errors="coerce").dropna().astype(int).unique().tolist())
    return ResidualMappingModel(model=model, feature_columns=features, layers=levels)


@dataclass
class IDWNeighborCache:
    indices: np.ndarray
    distances_km: np.ndarray
    nearest_km: np.ndarray
    counts: np.ndarray
    train: pd.DataFrame
    targets: pd.DataFrame


def build_idw_neighbor_cache(
    train: pd.DataFrame,
    targets: pd.DataFrame,
    max_k: int = 16,
    progress_desc: str = "Build IDW neighbor cache",
) -> IDWNeighborCache:
    train = train.reset_index(drop=True).copy()
    targets = targets.reset_index(drop=True).copy()
    indices = np.full((len(targets), int(max_k)), -1, dtype=np.int32)
    distances = np.full((len(targets), int(max_k)), np.inf, dtype=np.float32)
    train["_train_index"] = np.arange(len(train), dtype=np.int32)
    targets["_target_index"] = np.arange(len(targets), dtype=np.int32)
    train_groups = {key: group for key, group in train.groupby(["time", "layer_hpa"], sort=False)}
    target_groups = targets.groupby(["time", "layer_hpa"], sort=False)
    for key, target_group in tqdm(
        target_groups,
        total=target_groups.ngroups,
        desc=progress_desc,
        unit="group",
        dynamic_ncols=True,
    ):
        context = train_groups.get(key)
        if context is None or context.empty:
            continue
        distance = _distance_matrix(
            target_group["lat"].to_numpy(dtype=float),
            target_group["lon"].to_numpy(dtype=float),
            context["lat"].to_numpy(dtype=float),
            context["lon"].to_numpy(dtype=float),
        )
        distance[
            target_group["station_id"].astype(str).to_numpy()[:, None]
            == context["station_id"].astype(str).to_numpy()[None, :]
        ] = np.inf
        context_indices = context["_train_index"].to_numpy(dtype=np.int32)
        for local_index, target_index in enumerate(target_group["_target_index"].to_numpy(dtype=np.int32)):
            order = np.argsort(distance[local_index])
            order = order[np.isfinite(distance[local_index, order])][: int(max_k)]
            take = len(order)
            if take:
                indices[target_index, :take] = context_indices[order]
                distances[target_index, :take] = distance[local_index, order]
    valid = indices >= 0
    counts = valid.sum(axis=1).astype(np.float32)
    nearest = np.where(valid.any(axis=1), distances[:, 0], np.inf).astype(np.float32)
    return IDWNeighborCache(indices, distances, nearest, counts, train.drop(columns="_train_index"), targets.drop(columns="_target_index"))


def predict_from_idw_cache(
    cache: IDWNeighborCache,
    k: int,
    power: float,
    context_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = cache.train["residual_n"].to_numpy(dtype=np.float32) if context_values is None else np.asarray(context_values, dtype=np.float32)
    if len(values) != len(cache.train):
        raise ValueError("context_values length must match the cached training frame.")
    fallback_train = cache.train.copy()
    fallback_train["residual_n"] = values
    prediction = mean_residual_prediction(fallback_train, cache.targets)
    take = min(int(k), cache.indices.shape[1])
    selected_indices = cache.indices[:, :take]
    selected_distances = cache.distances_km[:, :take]
    valid = selected_indices >= 0
    safe_indices = np.maximum(selected_indices, 0)
    weights = np.where(valid, 1.0 / np.maximum(selected_distances, 1e-3) ** float(power), 0.0)
    denominator = weights.sum(axis=1)
    numerator = (weights * values[safe_indices]).sum(axis=1)
    has_context = denominator > 0
    prediction[has_context] = (numerator[has_context] / denominator[has_context]).astype(np.float32)
    counts = valid.sum(axis=1).astype(np.float32)
    nearest = np.where(has_context, selected_distances[:, 0], np.inf).astype(np.float32)
    return prediction.astype(np.float32), nearest, counts


def idw_residual_prediction(
    train: pd.DataFrame,
    targets: pd.DataFrame,
    k: int = 8,
    power: float = 2.0,
    context_values: np.ndarray | None = None,
    progress_desc: str = "Evaluate IDW",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache = build_idw_neighbor_cache(train, targets, max_k=k, progress_desc=progress_desc)
    return predict_from_idw_cache(cache, k=k, power=power, context_values=context_values)


def _distance_matrix(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1 = np.deg2rad(np.asarray(lat1, dtype=np.float64))[:, None]
    lon1 = np.deg2rad(np.asarray(lon1, dtype=np.float64))[:, None]
    lat2 = np.deg2rad(np.asarray(lat2, dtype=np.float64))[None, :]
    lon2 = np.deg2rad(np.asarray(lon2, dtype=np.float64))[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return (6371.0 * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))).astype(np.float32)


class RBFMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass
class RBFMLPResult:
    prediction: np.ndarray
    anchors: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    model_state: dict[str, Any]


def train_predict_rbf_mlp(
    train: pd.DataFrame,
    targets: pd.DataFrame,
    anchors_count: int = 64,
    range_km: float = 1500.0,
    epochs: int = 20,
    batch_size: int = 4096,
    seed: int = 42,
    device: str | torch.device | None = None,
    progress_desc: str = "RBF-MLP",
) -> RBFMLPResult:
    rng = np.random.default_rng(seed)
    station_coords = train[["station_id", "lat", "lon"]].drop_duplicates("station_id")[["lat", "lon"]].to_numpy(dtype=np.float32)
    anchors = farthest_point_anchors(station_coords, min(int(anchors_count), len(station_coords)), rng)
    train_x = rbf_features(train, anchors, range_km, progress_desc=f"{progress_desc}: train features")
    target_x = rbf_features(targets, anchors, range_km, progress_desc=f"{progress_desc}: target features")
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    train_x = ((train_x - mean) / std).astype(np.float32)
    target_x = ((target_x - mean) / std).astype(np.float32)
    y = train["residual_n"].to_numpy(dtype=np.float32)[:, None]
    torch.manual_seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = RBFMLP(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(y)), batch_size=batch_size, shuffle=True)
    progress = tqdm(
        total=int(epochs) * len(loader),
        desc=f"{progress_desc}: training",
        unit="batch",
        dynamic_ncols=True,
    )
    running_loss = 0.0
    completed_batches = 0
    for epoch in range(int(epochs)):
        model.train()
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.smooth_l1_loss(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach())
            completed_batches += 1
            progress.update(1)
            if completed_batches == 1 or completed_batches % 10 == 0:
                progress.set_postfix(
                    epoch=f"{epoch + 1}/{epochs}",
                    loss=f"{running_loss / completed_batches:.4f}",
                    device=str(device),
                )
    progress.close()
    model.eval()
    output = []
    with torch.inference_mode():
        starts = range(0, len(target_x), batch_size)
        for start in tqdm(starts, total=len(starts), desc=f"{progress_desc}: inference", unit="batch", dynamic_ncols=True):
            output.append(model(torch.from_numpy(target_x[start : start + batch_size]).to(device)).cpu().numpy())
    return RBFMLPResult(
        prediction=np.concatenate(output)[:, 0].astype(np.float32),
        anchors=anchors,
        feature_mean=mean,
        feature_std=std,
        model_state={key: value.detach().cpu() for key, value in model.state_dict().items()},
    )


def rbf_features(
    frame: pd.DataFrame,
    anchors: np.ndarray,
    range_km: float,
    progress_desc: str = "Build RBF features",
    chunk_size: int = 100_000,
) -> np.ndarray:
    coords = frame[["lat", "lon"]].to_numpy(dtype=np.float32)
    chunks = []
    starts = range(0, len(coords), int(chunk_size))
    for start in tqdm(starts, total=len(starts), desc=progress_desc, unit="chunk", dynamic_ncols=True):
        part = coords[start : start + int(chunk_size)]
        distance = _distance_matrix(part[:, 0], part[:, 1], anchors[:, 0], anchors[:, 1])
        chunks.append(np.exp(-((distance / float(range_km)) ** 2)).astype(np.float32))
    rbf = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, len(anchors)), dtype=np.float32)
    time = pd.to_datetime(frame["time"])
    calendar = np.column_stack(
        [
            np.sin(2 * np.pi * time.dt.dayofyear.to_numpy() / 366.0),
            np.cos(2 * np.pi * time.dt.dayofyear.to_numpy() / 366.0),
            np.sin(2 * np.pi * time.dt.hour.to_numpy() / 24.0),
            np.cos(2 * np.pi * time.dt.hour.to_numpy() / 24.0),
        ]
    )
    physical = frame[["n_era5", "era5_height_km", "layer_norm"]].to_numpy(dtype=np.float32)
    return np.concatenate([rbf.astype(np.float32), calendar.astype(np.float32), physical], axis=1)


def farthest_point_anchors(coords: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    selected = [int(rng.integers(0, len(coords)))]
    min_distance = np.full(len(coords), np.inf)
    while len(selected) < count:
        last = coords[selected[-1]]
        distance, _ = spherical_distance_and_bearing(last[0], last[1], coords[:, 0], coords[:, 1])
        min_distance = np.minimum(min_distance, distance)
        selected.append(int(np.argmax(min_distance)))
    return coords[np.asarray(selected)]
