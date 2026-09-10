from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from igra_forecast.multisite_data import MultiSiteData, MultiSiteSplit, multisite_predictions_to_frame


class MultiSiteModel(ABC):
    def __init__(self, name: str, params: dict[str, Any] | None = None) -> None:
        self.name = name
        self.params = params or {}

    @abstractmethod
    def fit(self, data: MultiSiteData) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, data: MultiSiteData, split: str = "test") -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError


def build_multisite_model(cfg: dict[str, Any]) -> MultiSiteModel:
    name = cfg["model"]["name"]
    params = dict(cfg.get("model", {}).get("params", {}) or {})
    params.update(dict(cfg.get("multisite_models", {}).get(name, {}) or {}))
    if name == "persistence":
        return PersistenceMultiSite(name, params)
    if name == "recent_mean":
        return RecentMeanMultiSite(name, params)
    if name == "seasonal_hour_climatology":
        return SeasonalHourClimatology(name, params)
    if name == "geo_knn_climatology":
        return GeoKnnClimatology(name, params)
    if name in {"ridge_lag", "hgb_lag"}:
        return LagTabularModel(name, params)
    if name in {"gru", "tcn", "patchtst", "itransformer", "timemixer", "glr_prior_mixer", "glr_prior_mixer_v2"}:
        deep_params = {
            "common": dict(cfg.get("deep_models", {}).get("common", {}) or {}),
            name: dict(cfg.get("deep_models", {}).get(name, {}) or {}),
        }
        deep_params["common"].update(params.get("common", {}))
        deep_params[name].update({k: v for k, v in params.items() if k != "common"})
        return DeepMultiSiteModel(name, deep_params)
    raise ValueError(f"Unknown multisite model: {name}")


class PersistenceMultiSite(MultiSiteModel):
    def fit(self, data: MultiSiteData) -> None:
        self.n_index = data.feature_names.index("Refractivity_N")

    def predict(self, data: MultiSiteData, split: str = "test") -> pd.DataFrame:
        split_data = data.splits[split]
        last_n = split_data.x[:, -1, :, self.n_index]
        pred = np.repeat(last_n[:, None, :], len(data.horizons), axis=1)
        return multisite_predictions_to_frame(data, split, pred.astype(np.float32))

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump({"name": self.name, "params": self.params}, path / "model.pkl")


class RecentMeanMultiSite(MultiSiteModel):
    def fit(self, data: MultiSiteData) -> None:
        self.n_index = data.feature_names.index("Refractivity_N")
        self.steps = int(self.params.get("steps", 6))

    def predict(self, data: MultiSiteData, split: str = "test") -> pd.DataFrame:
        split_data = data.splits[split]
        mean_n = split_data.x[:, -self.steps :, :, self.n_index].mean(axis=1)
        pred = np.repeat(mean_n[:, None, :], len(data.horizons), axis=1)
        return multisite_predictions_to_frame(data, split, pred.astype(np.float32))

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump({"name": self.name, "params": self.params}, path / "model.pkl")


class SeasonalHourClimatology(MultiSiteModel):
    def fit(self, data: MultiSiteData) -> None:
        self.layers = data.layers
        self.horizons = data.horizons
        self.global_mean = data.train.y.mean(axis=(0, 1))
        rows = []
        for idx in range(len(data.train.y)):
            for h_idx, _ in enumerate(data.horizons):
                target_time = pd.Timestamp(data.train.target_times[idx, h_idx])
                for l_idx, layer in enumerate(data.layers):
                    rows.append(
                        {
                            "month": target_time.month,
                            "hour": target_time.hour,
                            "layer": layer,
                            "value": float(data.train.y[idx, h_idx, l_idx]),
                        }
                    )
        frame = pd.DataFrame(rows)
        self.lookup = frame.groupby(["month", "hour", "layer"])["value"].mean().to_dict()

    def predict(self, data: MultiSiteData, split: str = "test") -> pd.DataFrame:
        split_data = data.splits[split]
        pred = np.empty_like(split_data.y)
        for i in range(len(split_data.y)):
            for h_idx, _ in enumerate(data.horizons):
                target_time = pd.Timestamp(split_data.target_times[i, h_idx])
                for l_idx, layer in enumerate(data.layers):
                    pred[i, h_idx, l_idx] = self.lookup.get((target_time.month, target_time.hour, layer), self.global_mean[l_idx])
        return multisite_predictions_to_frame(data, split, pred.astype(np.float32))

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump({"name": self.name, "params": self.params, "lookup": self.lookup}, path / "model.pkl")


class GeoKnnClimatology(SeasonalHourClimatology):
    def fit(self, data: MultiSiteData) -> None:
        self.layers = data.layers
        self.horizons = data.horizons
        self.k = int(self.params.get("k", 5))
        self.global_mean = data.train.y.mean(axis=(0, 1))
        station_coords = {}
        for station_id, coord in zip(data.train.station_ids, data.train.coords):
            station_coords[str(station_id)] = coord
        self.train_station_ids = np.array(sorted(station_coords))
        self.train_coords = np.stack([station_coords[sid] for sid in self.train_station_ids]).astype(np.float32)
        rows = []
        for idx, station_id in enumerate(data.train.station_ids):
            for h_idx, _ in enumerate(data.horizons):
                target_time = pd.Timestamp(data.train.target_times[idx, h_idx])
                for l_idx, layer in enumerate(data.layers):
                    rows.append(
                        {
                            "station_id": str(station_id),
                            "month": target_time.month,
                            "hour": target_time.hour,
                            "layer": layer,
                            "value": float(data.train.y[idx, h_idx, l_idx]),
                        }
                    )
        self.station_lookup = pd.DataFrame(rows).groupby(["station_id", "month", "hour", "layer"])["value"].mean().to_dict()

    def predict(self, data: MultiSiteData, split: str = "test") -> pd.DataFrame:
        split_data = data.splits[split]
        pred = np.empty_like(split_data.y)
        for i in range(len(split_data.y)):
            distances = haversine_distance(split_data.coords[i], self.train_coords)
            neighbor_ids = self.train_station_ids[np.argsort(distances)[: self.k]]
            for h_idx, _ in enumerate(data.horizons):
                target_time = pd.Timestamp(split_data.target_times[i, h_idx])
                for l_idx, layer in enumerate(data.layers):
                    values = [
                        self.station_lookup[(sid, target_time.month, target_time.hour, layer)]
                        for sid in neighbor_ids
                        if (sid, target_time.month, target_time.hour, layer) in self.station_lookup
                    ]
                    pred[i, h_idx, l_idx] = float(np.mean(values)) if values else self.global_mean[l_idx]
        return multisite_predictions_to_frame(data, split, pred.astype(np.float32))

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "name": self.name,
                "params": self.params,
                "station_lookup": self.station_lookup,
                "train_station_ids": self.train_station_ids,
                "train_coords": self.train_coords,
                "global_mean": self.global_mean,
                "k": self.k,
            },
            path / "model.pkl",
        )


class LagTabularModel(MultiSiteModel):
    def fit(self, data: MultiSiteData) -> None:
        x_train = tabular_features(data.train)
        y_train = data.train.y.reshape(len(data.train.y), -1)
        if self.name == "ridge_lag":
            self.model = Ridge(alpha=float(self.params.get("alpha", 10.0)))
        else:
            base = HistGradientBoostingRegressor(
                max_iter=int(self.params.get("max_iter", 300)),
                learning_rate=float(self.params.get("learning_rate", 0.04)),
                max_leaf_nodes=int(self.params.get("max_leaf_nodes", 31)),
                l2_regularization=float(self.params.get("l2_regularization", 0.01)),
            )
            self.model = MultiOutputRegressor(base)
        self.model.fit(x_train, y_train)

    def predict(self, data: MultiSiteData, split: str = "test") -> pd.DataFrame:
        split_data = data.splits[split]
        pred = self.model.predict(tabular_features(split_data)).reshape(len(split_data.y), len(data.horizons), len(data.layers))
        return multisite_predictions_to_frame(data, split, pred.astype(np.float32))

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump({"name": self.name, "params": self.params, "model": self.model}, path / "model.pkl")


class DeepMultiSiteModel(MultiSiteModel):
    def fit(self, data: MultiSiteData) -> None:
        common = self.params["common"]
        self.device = select_device(common.get("device", "auto"))
        self.module = build_module(self.name, data, self.params[self.name]).to(self.device)
        self.history = train_module(self.module, data, common, self.device)

    def predict(self, data: MultiSiteData, split: str = "test") -> pd.DataFrame:
        split_data = data.splits[split]
        loader = make_loader(split_data, int(self.params["common"].get("batch_size", 128)), shuffle=False)
        self.module.eval()
        preds = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"预测 {split}", unit="batch"):
                xb, _, coords, calendar = [item.to(self.device) for item in batch]
                preds.append(self.module(xb, coords, calendar).detach().cpu().numpy())
        pred = np.concatenate(preds, axis=0)
        return multisite_predictions_to_frame(data, split, pred.astype(np.float32))

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"name": self.name, "params": self.params, "state_dict": self.module.state_dict()}, path / "model.pt")
        if hasattr(self, "history"):
            self.history.to_csv(path / "training_history.csv", index=False)


def tabular_features(split: MultiSiteSplit) -> np.ndarray:
    coords = normalize_coords(split.coords)
    return np.concatenate([split.x.reshape(len(split.x), -1), coords, split.calendar], axis=1)


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    lat = coords[:, 0:1] / 90.0
    lon_rad = np.deg2rad(coords[:, 1:2])
    return np.concatenate([lat, np.sin(lon_rad), np.cos(lon_rad)], axis=1).astype(np.float32)


def haversine_distance(coord: np.ndarray, coords: np.ndarray) -> np.ndarray:
    lat1 = np.deg2rad(coord[0])
    lon1 = np.deg2rad(coord[1])
    lat2 = np.deg2rad(coords[:, 0])
    lon2 = np.deg2rad(coords[:, 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * np.arcsin(np.sqrt(a))


def select_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def make_loader(split: MultiSiteSplit, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(split.x).float(),
        torch.from_numpy(split.y).float(),
        torch.from_numpy(split.coords).float(),
        torch.from_numpy(split.calendar).float(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_module(module: nn.Module, data: MultiSiteData, common: dict[str, Any], device: torch.device) -> pd.DataFrame:
    epochs = int(common.get("epochs", 30))
    batch_size = int(common.get("batch_size", 128))
    patience = int(common.get("patience", 8))
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=float(common.get("learning_rate", 1e-3)),
        weight_decay=float(common.get("weight_decay", 1e-4)),
    )
    train_loader = make_loader(data.train, batch_size, shuffle=True)
    val_loader = make_loader(data.val, batch_size, shuffle=False)
    best_state = None
    best_val = float("inf")
    wait = 0
    rows = []
    for epoch in tqdm(range(1, epochs + 1), desc="训练多站深度模型", unit="epoch"):
        module.train()
        losses = []
        for batch in train_loader:
            xb, yb, coords, calendar = [item.to(device) for item in batch]
            optimizer.zero_grad(set_to_none=True)
            pred = module(xb, coords, calendar)
            loss = torch.mean((pred - yb) ** 2)
            loss.backward()
            nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_loss = evaluate_module(module, val_loader, device)
        train_loss = float(np.mean(losses))
        rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state:
        module.load_state_dict(best_state)
    return pd.DataFrame(rows)


def evaluate_module(module: nn.Module, loader: DataLoader, device: torch.device) -> float:
    module.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            xb, yb, coords, calendar = [item.to(device) for item in batch]
            losses.append(float(torch.mean((module(xb, coords, calendar) - yb) ** 2).detach().cpu()))
    return float(np.mean(losses))


def build_module(name: str, data: MultiSiteData, params: dict[str, Any]) -> nn.Module:
    n_layers = len(data.layers)
    n_features = len(data.feature_names)
    horizons = len(data.horizons)
    if name == "gru":
        return SequenceModule("gru", data.input_length, n_layers, n_features, horizons, params)
    if name == "tcn":
        return TcnModule(data.input_length, n_layers, n_features, horizons, params)
    if name == "patchtst":
        return PatchTstModule(data.input_length, n_layers, n_features, horizons, params)
    if name == "itransformer":
        return ITransformerModule(data.input_length, n_layers, n_features, horizons, params)
    if name == "timemixer":
        return TimeMixerModule(data.input_length, n_layers, n_features, horizons, params)
    if name == "glr_prior_mixer":
        n_index = data.feature_names.index("Refractivity_N")
        return GlrPriorMixer(data.input_length, n_layers, n_features, horizons, n_index, params)
    if name == "glr_prior_mixer_v2":
        n_index = data.feature_names.index("Refractivity_N")
        return GlrPriorMixerV2(data.input_length, n_layers, n_features, horizons, n_index, params)
    raise ValueError(f"Unsupported deep multisite model: {name}")


class SequenceModule(nn.Module):
    def __init__(self, cell: str, input_length: int, n_layers: int, n_features: int, horizons: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        hidden = int(cfg.get("hidden_size", 128))
        layers = int(cfg.get("num_layers", 2))
        rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn_cls(n_layers * n_features, hidden, layers, batch_first=True, dropout=float(cfg.get("dropout", 0.1)))
        self.head = nn.Sequential(nn.LayerNorm(hidden + 7), nn.Linear(hidden + 7, horizons * n_layers))
        self.horizons = horizons
        self.n_layers = n_layers

    def forward(self, x: torch.Tensor, coords: torch.Tensor, calendar: torch.Tensor) -> torch.Tensor:
        z = x.flatten(2)
        out, _ = self.rnn(z)
        meta = torch.cat([coord_features(coords), calendar], dim=1)
        y = self.head(torch.cat([out[:, -1], meta], dim=1))
        return y.view(x.size(0), self.horizons, self.n_layers)


class TcnModule(nn.Module):
    def __init__(self, input_length: int, n_layers: int, n_features: int, horizons: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        hidden = int(cfg.get("hidden_size", 128))
        blocks = []
        in_ch = n_layers * n_features
        for idx in range(int(cfg.get("num_layers", 4))):
            blocks.append(nn.Sequential(nn.Conv1d(in_ch, hidden, 3, padding=2**idx, dilation=2**idx), nn.GELU(), nn.Dropout(float(cfg.get("dropout", 0.1)))))
            in_ch = hidden
        self.net = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden + 7, horizons * n_layers)
        self.horizons = horizons
        self.n_layers = n_layers

    def forward(self, x: torch.Tensor, coords: torch.Tensor, calendar: torch.Tensor) -> torch.Tensor:
        z = x.flatten(2).transpose(1, 2)
        out = self.net(z)
        pooled = out[:, :, -1]
        y = self.head(torch.cat([pooled, coord_features(coords), calendar], dim=1))
        return y.view(x.size(0), self.horizons, self.n_layers)


class PatchTstModule(nn.Module):
    def __init__(self, input_length: int, n_layers: int, n_features: int, horizons: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        d_model = int(cfg.get("d_model", cfg.get("hidden_size", 128)))
        patch_len = int(cfg.get("patch_len", 6))
        stride = int(cfg.get("stride", 3))
        heads = int(cfg.get("num_heads", 4))
        depth = int(cfg.get("num_layers", 2))
        dropout = float(cfg.get("dropout", 0.1))
        self.patch_len = patch_len
        self.stride = stride
        self.patch = nn.Linear(patch_len * n_layers * n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model + 7)
        self.head = nn.Linear(d_model + 7, horizons * n_layers)
        self.horizons = horizons
        self.n_layers = n_layers

    def forward(self, x: torch.Tensor, coords: torch.Tensor, calendar: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 1, 4, 2, 3).flatten(2)
        z = self.encoder(self.patch(patches)).mean(dim=1)
        y = self.head(torch.cat([z, coord_features(coords), calendar], dim=1))
        return y.view(x.size(0), self.horizons, self.n_layers)


class ITransformerModule(nn.Module):
    def __init__(self, input_length: int, n_layers: int, n_features: int, horizons: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        d_model = int(cfg.get("d_model", cfg.get("hidden_size", 128)))
        heads = int(cfg.get("num_heads", 4))
        depth = int(cfg.get("num_layers", 2))
        dropout = float(cfg.get("dropout", 0.1))
        self.n_tokens = n_layers * n_features
        self.token_proj = nn.Linear(input_length, d_model)
        self.token_embedding = nn.Parameter(torch.randn(1, self.n_tokens, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Linear(d_model + 7, horizons * n_layers)
        self.horizons = horizons
        self.n_layers = n_layers

    def forward(self, x: torch.Tensor, coords: torch.Tensor, calendar: torch.Tensor) -> torch.Tensor:
        tokens = x.permute(0, 2, 3, 1).flatten(1, 2)
        z = self.encoder(self.token_proj(tokens) + self.token_embedding).mean(dim=1)
        y = self.head(torch.cat([z, coord_features(coords), calendar], dim=1))
        return y.view(x.size(0), self.horizons, self.n_layers)


class TimeMixerModule(nn.Module):
    def __init__(self, input_length: int, n_layers: int, n_features: int, horizons: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        hidden = int(cfg.get("hidden_size", 256))
        dropout = float(cfg.get("dropout", 0.1))
        self.scales = [int(s) for s in cfg.get("scales", [1, 2, 6])]
        self.branches = nn.ModuleList()
        for scale in self.scales:
            length = input_length // scale
            self.branches.append(
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(length * n_layers * n_features, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.mix = nn.Sequential(nn.Linear(hidden * len(self.scales) + 7, hidden), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Linear(hidden, horizons * n_layers)
        self.horizons = horizons
        self.n_layers = n_layers

    def forward(self, x: torch.Tensor, coords: torch.Tensor, calendar: torch.Tensor) -> torch.Tensor:
        features = []
        for scale, branch in zip(self.scales, self.branches):
            if scale == 1:
                xs = x
            else:
                keep = (x.size(1) // scale) * scale
                xs = x[:, -keep:].reshape(x.size(0), keep // scale, scale, x.size(2), x.size(3)).mean(dim=2)
            features.append(branch(xs))
        z = self.mix(torch.cat([*features, coord_features(coords), calendar], dim=1))
        y = self.head(z)
        return y.view(x.size(0), self.horizons, self.n_layers)


class GlrPriorMixer(nn.Module):
    def __init__(self, input_length: int, n_layers: int, n_features: int, horizons: int, n_index: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.n_index = n_index
        self.horizons = horizons
        self.n_layers = n_layers
        hidden = int(cfg.get("hidden_size", 128))
        self.encoder = nn.GRU(n_layers * n_features, hidden, batch_first=True)
        self.residual = nn.Linear(hidden + 7, horizons * n_layers)
        self.gate = nn.Linear(hidden + 7, horizons * n_layers * 4)

    def forward(self, x: torch.Tensor, coords: torch.Tensor, calendar: torch.Tensor) -> torch.Tensor:
        n_seq = x[:, :, :, self.n_index]
        persistence = n_seq[:, -1]
        mean_24h = n_seq[:, -2:].mean(dim=1)
        mean_7d = n_seq[:, -14:].mean(dim=1)
        trend = persistence + (n_seq[:, -1] - n_seq[:, -2])
        priors = torch.stack([persistence, mean_24h, mean_7d, trend], dim=1)
        priors = priors[:, None, :, :].repeat(1, self.horizons, 1, 1)
        out, _ = self.encoder(x.flatten(2))
        meta = torch.cat([out[:, -1], coord_features(coords), calendar], dim=1)
        residual = self.residual(meta).view(x.size(0), self.horizons, self.n_layers)
        weights = torch.softmax(self.gate(meta).view(x.size(0), self.horizons, self.n_layers, 4), dim=-1)
        prior_pred = (weights * priors.permute(0, 1, 3, 2)).sum(dim=-1)
        return prior_pred + 0.2 * residual


class GlrPriorMixerV2(nn.Module):
    def __init__(self, input_length: int, n_layers: int, n_features: int, horizons: int, n_index: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.n_index = n_index
        self.horizons = horizons
        self.n_layers = n_layers
        self.use_lag_prior = bool(cfg.get("use_lag_prior", True))
        self.use_climatology = bool(cfg.get("use_climatology", True))
        self.use_geo = bool(cfg.get("use_geo", True))
        self.use_layer_gate = bool(cfg.get("use_layer_gate", True))
        hidden = int(cfg.get("hidden_size", 160))
        self.encoder = nn.GRU(n_layers * n_features, hidden, batch_first=True)
        meta_dim = hidden + (7 if self.use_geo else 4)
        self.lag_prior = nn.Linear(input_length * n_layers * n_features, horizons * n_layers)
        self.clim_prior = nn.Sequential(nn.Linear((7 if self.use_geo else 4), hidden), nn.GELU(), nn.Linear(hidden, horizons * n_layers))
        self.residual = nn.Sequential(nn.LayerNorm(meta_dim), nn.Linear(meta_dim, hidden), nn.GELU(), nn.Linear(hidden, horizons * n_layers))
        gate_units = horizons * n_layers if self.use_layer_gate else horizons
        self.gate = nn.Linear(meta_dim, gate_units * 5)
        self.residual_scale = float(cfg.get("residual_scale", 0.15))

    def forward(self, x: torch.Tensor, coords: torch.Tensor, calendar: torch.Tensor) -> torch.Tensor:
        n_seq = x[:, :, :, self.n_index]
        persistence = n_seq[:, -1]
        recent_mean = n_seq[:, -6:].mean(dim=1)
        trend = persistence + torch.clamp(n_seq[:, -1] - n_seq[:, -2], -2.5, 2.5)
        persistence = persistence[:, None].repeat(1, self.horizons, 1)
        recent_mean = recent_mean[:, None].repeat(1, self.horizons, 1)
        trend = trend[:, None].repeat(1, self.horizons, 1)
        lag_prior = self.lag_prior(x.flatten(1)).view(x.size(0), self.horizons, self.n_layers)
        if not self.use_lag_prior:
            lag_prior = torch.zeros_like(lag_prior)
        geo_calendar = torch.cat([coord_features(coords), calendar], dim=1) if self.use_geo else calendar
        climatology = self.clim_prior(geo_calendar).view(x.size(0), self.horizons, self.n_layers)
        if not self.use_climatology:
            climatology = torch.zeros_like(climatology)
        out, _ = self.encoder(x.flatten(2))
        meta = torch.cat([out[:, -1], geo_calendar], dim=1)
        residual = self.residual(meta).view(x.size(0), self.horizons, self.n_layers) * self.residual_scale
        candidates = torch.stack([persistence, recent_mean, trend, lag_prior, climatology], dim=-1)
        gate = self.gate(meta)
        if self.use_layer_gate:
            weights = torch.softmax(gate.view(x.size(0), self.horizons, self.n_layers, 5), dim=-1)
        else:
            weights = torch.softmax(gate.view(x.size(0), self.horizons, 1, 5), dim=-1).repeat(1, 1, self.n_layers, 1)
        return (weights * candidates).sum(dim=-1) + residual


def coord_features(coords: torch.Tensor) -> torch.Tensor:
    lat = coords[:, 0:1] / 90.0
    lon = torch.deg2rad(coords[:, 1:2])
    return torch.cat([lat, torch.sin(lon), torch.cos(lon)], dim=1)
