from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import BallTree
from torch.utils.data import Dataset
from tqdm.auto import tqdm


EARTH_RADIUS_KM = 6371.0
DEFAULT_MAPPING_LAYERS = (925, 850, 700)


@dataclass
class MappingScalers:
    layers: list[int]
    n_mean: list[float]
    n_std: list[float]
    height_mean: list[float]
    height_std: list[float]
    residual_scale: list[float]

    @classmethod
    def fit(cls, profiles: pd.DataFrame, layers: Iterable[int] = DEFAULT_MAPPING_LAYERS) -> "MappingScalers":
        layer_list = [int(v) for v in layers]
        n = profiles[[f"n_era5_{v}" for v in layer_list]].to_numpy(dtype=np.float64)
        h = profiles[[f"era5_height_km_{v}" for v in layer_list]].to_numpy(dtype=np.float64)
        residual = profiles[[f"residual_n_{v}" for v in layer_list]].to_numpy(dtype=np.float64)
        n_std = np.nanstd(n, axis=0)
        h_std = np.nanstd(h, axis=0)
        residual_scale = np.nanstd(residual, axis=0)
        n_std = np.where(n_std > 1e-6, n_std, 1.0)
        h_std = np.where(h_std > 1e-6, h_std, 1.0)
        residual_scale = np.where(residual_scale > 1e-6, residual_scale, 1.0)
        return cls(
            layers=layer_list,
            n_mean=np.nanmean(n, axis=0).tolist(),
            n_std=n_std.tolist(),
            height_mean=np.nanmean(h, axis=0).tolist(),
            height_std=h_std.tolist(),
            residual_scale=residual_scale.tolist(),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MappingScalers":
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def normalize_n(self, values: np.ndarray) -> np.ndarray:
        return (values - np.asarray(self.n_mean)) / np.asarray(self.n_std)

    def normalize_height(self, values: np.ndarray) -> np.ndarray:
        return (values - np.asarray(self.height_mean)) / np.asarray(self.height_std)

    def normalize_residual(self, values: np.ndarray) -> np.ndarray:
        return values / np.asarray(self.residual_scale)

    def denormalize_residual(self, values: np.ndarray) -> np.ndarray:
        return values * np.asarray(self.residual_scale)


def build_profile_table(
    frame: pd.DataFrame,
    prior_residual: np.ndarray | None = None,
    layers: Iterable[int] = DEFAULT_MAPPING_LAYERS,
) -> pd.DataFrame:
    """Convert layer rows to complete station/time (or grid/time) three-level profiles."""
    layers = [int(v) for v in layers]
    required = {"time", "lat", "lon", "layer_hpa", "n_era5", "era5_height_km"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Profile input is missing columns: {missing}")
    work = frame.reset_index(drop=True).copy()
    work["_row_index"] = np.arange(len(work), dtype=np.int64)
    work["time"] = pd.to_datetime(work["time"])
    work["layer_hpa"] = pd.to_numeric(work["layer_hpa"], errors="coerce").astype("Int64")
    work = work[work["layer_hpa"].isin(layers)].copy()
    if prior_residual is not None:
        prior_values = np.asarray(prior_residual, dtype=np.float32)
        if len(prior_values) != len(frame):
            raise ValueError("prior_residual length must equal frame length")
        work["prior_residual"] = prior_values[work["_row_index"].to_numpy(dtype=int)]
    elif "prior_residual" not in work:
        work["prior_residual"] = 0.0
    if "station_id" not in work:
        work["station_id"] = ""
    if "split" not in work:
        work["split"] = "query"
    value_columns = ["n_era5", "era5_height_km", "prior_residual", "_row_index"]
    for optional in ["residual_n", "n_igra", "target_prior_residual", "crossfit_fold"]:
        if optional in work:
            value_columns.append(optional)
    value_columns.extend(
        column
        for column in work.columns
        if column.startswith("context_prior_residual_fold_") and column not in value_columns
    )
    keys = ["station_id", "time", "lat", "lon", "split"]
    wide = work.pivot_table(index=keys, columns="layer_hpa", values=value_columns, aggfunc="first")
    wide = wide.reindex(columns=pd.MultiIndex.from_product([value_columns, layers]))
    wide.columns = [f"{name}_{int(layer)}" for name, layer in wide.columns]
    wide = wide.reset_index()
    fold_columns = [f"crossfit_fold_{layer}" for layer in layers if f"crossfit_fold_{layer}" in wide.columns]
    if fold_columns:
        fold_values = wide[fold_columns].bfill(axis=1).iloc[:, 0]
        if wide[fold_columns].nunique(axis=1, dropna=True).gt(1).any():
            raise ValueError("A station-time profile has inconsistent crossfit_fold values across pressure levels.")
        wide["crossfit_fold"] = pd.to_numeric(fold_values, errors="coerce").fillna(-1).astype(np.int16)
    required_wide = [
        *(f"n_era5_{v}" for v in layers),
        *(f"era5_height_km_{v}" for v in layers),
        *(f"prior_residual_{v}" for v in layers),
        *(f"_row_index_{v}" for v in layers),
    ]
    if all(f"residual_n_{v}" in wide for v in layers):
        required_wide.extend(f"residual_n_{v}" for v in layers)
    wide = wide.dropna(subset=required_wide).reset_index(drop=True)
    for layer in layers:
        wide[f"_row_index_{layer}"] = wide[f"_row_index_{layer}"].astype(np.int64)
    return wide


def spherical_distance_and_bearing(
    target_lat: float,
    target_lon: float,
    context_lat: np.ndarray,
    context_lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lat1 = np.deg2rad(float(target_lat))
    lon1 = np.deg2rad(float(target_lon))
    lat2 = np.deg2rad(np.asarray(context_lat, dtype=np.float64))
    lon2 = np.deg2rad(np.asarray(context_lon, dtype=np.float64))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    distance = EARTH_RADIUS_KM * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.arctan2(y, x)
    return distance.astype(np.float32), bearing.astype(np.float32)


class ContextIndex:
    """Time-indexed training-station context profiles with BallTree nearest-neighbour queries."""

    def __init__(self, profiles: pd.DataFrame):
        self.groups: dict[pd.Timestamp, pd.DataFrame] = {}
        self.trees: dict[pd.Timestamp, BallTree] = {}
        for time, group in profiles.groupby("time", sort=False):
            group = group.reset_index(drop=True)
            key = pd.Timestamp(time)
            self.groups[key] = group
            coords_rad = np.deg2rad(group[["lat", "lon"]].to_numpy(dtype=np.float64))
            self.trees[key] = BallTree(coords_rad, metric="haversine")

    def nearest(self, time: pd.Timestamp, lat: float, lon: float, candidate_k: int) -> pd.DataFrame:
        key = pd.Timestamp(time)
        group = self.groups.get(key)
        if group is None or group.empty:
            return pd.DataFrame(columns=[])
        k = min(int(candidate_k), len(group))
        query = np.deg2rad(np.asarray([[lat, lon]], dtype=np.float64))
        _, indices = self.trees[key].query(query, k=k)
        return group.iloc[indices[0]].copy()


def _calendar_geo(row: pd.Series) -> np.ndarray:
    time = pd.Timestamp(row["time"])
    lon_rad = np.deg2rad(float(row["lon"]))
    return np.asarray(
        [
            float(row["lat"]) / 90.0,
            np.sin(lon_rad),
            np.cos(lon_rad),
            np.sin(2.0 * np.pi * time.dayofyear / 366.0),
            np.cos(2.0 * np.pi * time.dayofyear / 366.0),
            np.sin(2.0 * np.pi * time.hour / 24.0),
            np.cos(2.0 * np.pi * time.hour / 24.0),
        ],
        dtype=np.float32,
    )


class SpatialEpisodeDataset(Dataset):
    def __init__(
        self,
        targets: pd.DataFrame,
        contexts: pd.DataFrame,
        scalers: MappingScalers,
        nearest_context: int = 16,
        candidate_context: int = 32,
        training: bool = False,
        seed: int = 42,
        context_fraction: float = 1.0,
        mask_probabilities: tuple[float, float, float] = (0.5, 0.3, 0.2),
        use_context: bool = True,
        use_geometry: bool = True,
        use_hgb_prior: bool = True,
        target_prior_prefix: str = "prior_residual",
        cache_dir: str | Path | None = None,
    ):
        self.targets = targets.reset_index(drop=True)
        self.contexts = contexts.reset_index(drop=True)
        self.scalers = scalers
        self.layers = scalers.layers
        self.k = int(nearest_context)
        self.candidate_k = max(int(candidate_context), self.k)
        self.training = bool(training)
        self.seed = int(seed)
        self.epoch = 0
        self.context_fraction = float(context_fraction)
        self.mask_probabilities = np.asarray(mask_probabilities, dtype=float)
        self.mask_probabilities /= self.mask_probabilities.sum()
        self.use_context = bool(use_context)
        self.use_geometry = bool(use_geometry)
        self.use_hgb_prior = bool(use_hgb_prior)
        self.target_prior_prefix = target_prior_prefix
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._prepare_static_arrays()
        self._materialize_epoch_cache(epoch=0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if self.training:
            self._materialize_epoch_cache(epoch=self.epoch)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[int(index)] for key, value in self._cache.items()}

    def _prepare_static_arrays(self) -> None:
        target_n = self._columns(self.targets, "n_era5")
        target_height = self._columns(self.targets, "era5_height_km")
        target_prior = self._prior_columns(self.targets, self.target_prior_prefix)
        if not self.use_hgb_prior:
            target_prior = np.zeros_like(target_prior)
        target_prior = self.scalers.normalize_residual(target_prior).astype(np.float32)
        self._target_n = target_n
        self._query = torch.from_numpy(
            np.concatenate(
                [self.scalers.normalize_n(target_n), self.scalers.normalize_height(target_height), target_prior, self._calendar(self.targets)],
                axis=1,
            ).astype(np.float32)
        )
        self._prior = torch.from_numpy(target_prior)
        self._target_index = torch.arange(len(self.targets), dtype=torch.int64)
        self._target_fold = (
            pd.to_numeric(self.targets["crossfit_fold"], errors="coerce").fillna(-1).to_numpy(dtype=np.int16)
            if "crossfit_fold" in self.targets
            else np.full(len(self.targets), -1, dtype=np.int16)
        )
        self._target = None
        if all(f"residual_n_{v}" in self.targets.columns for v in self.layers):
            self._target = torch.from_numpy(
                self.scalers.normalize_residual(self._columns(self.targets, "residual_n")).astype(np.float32)
            )
        context_n = self._columns(self.contexts, "n_era5") if not self.contexts.empty else np.zeros((0, len(self.layers)), dtype=np.float32)
        context_height = self._columns(self.contexts, "era5_height_km") if not self.contexts.empty else np.zeros_like(context_n)
        context_prior = self._prior_columns(self.contexts, "prior_residual") if not self.contexts.empty else np.zeros_like(context_n)
        context_residual = self._columns(self.contexts, "residual_n") if not self.contexts.empty else np.zeros_like(context_n)
        if not self.use_hgb_prior:
            context_prior = np.zeros_like(context_prior)
        self._context_n = context_n
        self._context_features = self._make_context_features(context_n, context_height, context_prior, context_residual)
        self._context_features_by_fold: dict[int, np.ndarray] = {}
        if not self.contexts.empty:
            prefixes = sorted(
                {column.rsplit("_", 1)[0] for column in self.contexts.columns if column.startswith("context_prior_residual_fold_")}
            )
            for prefix in prefixes:
                try:
                    fold = int(prefix.removeprefix("context_prior_residual_fold_"))
                    fold_prior = self._prior_columns(self.contexts, prefix)
                except (KeyError, ValueError):
                    continue
                if not self.use_hgb_prior:
                    fold_prior = np.zeros_like(fold_prior)
                self._context_features_by_fold[fold] = self._make_context_features(
                    context_n, context_height, fold_prior, context_residual
                )
        if len(self._context_features) == 0:
            self._context_features = np.zeros((1, 19), dtype=np.float32)
            self._context_n = np.zeros((1, len(self.layers)), dtype=np.float32)
        self._build_candidate_cache()

    def _build_candidate_cache(self) -> None:
        count = len(self.targets)
        self._candidate_indices = np.full((count, self.candidate_k), -1, dtype=np.int32)
        self._candidate_distance = np.full((count, self.candidate_k), np.inf, dtype=np.float32)
        self._candidate_bearing = np.zeros((count, self.candidate_k), dtype=np.float32)
        self._target_block = self._block_ids(self.targets)
        self._context_block = self._block_ids(self.contexts) if not self.contexts.empty else np.zeros(1, dtype=np.int32)
        if self.contexts.empty:
            return
        cache_path = self._neighbor_cache_path()
        if cache_path is not None and self._load_neighbor_cache(cache_path):
            tqdm.write(f"[SRNP cache] Loaded neighbor index: {cache_path}")
            return
        context_by_time = {pd.Timestamp(time): group.index.to_numpy(dtype=np.int32) for time, group in self.contexts.groupby("time", sort=False)}
        target_groups = self.targets.groupby("time", sort=False).groups
        for time, target_rows in tqdm(target_groups.items(), desc="缓存空间近邻", unit="time", dynamic_ncols=True, leave=False):
            target_rows = np.asarray(list(target_rows), dtype=np.int32)
            context_rows = context_by_time.get(pd.Timestamp(time))
            if context_rows is None or len(context_rows) == 0:
                continue
            query_coords = self.targets.loc[target_rows, ["lat", "lon"]].to_numpy(dtype=np.float64)
            context_coords = self.contexts.loc[context_rows, ["lat", "lon"]].to_numpy(dtype=np.float64)
            tree = BallTree(np.deg2rad(context_coords), metric="haversine")
            query_k = min(len(context_rows), self.candidate_k + 1)
            _, local_indices = tree.query(np.deg2rad(query_coords), k=query_k)
            global_indices = context_rows[local_indices]
            candidate_lat = self.contexts.loc[global_indices.reshape(-1), "lat"].to_numpy(dtype=np.float64).reshape(global_indices.shape)
            candidate_lon = self.contexts.loc[global_indices.reshape(-1), "lon"].to_numpy(dtype=np.float64).reshape(global_indices.shape)
            distance, bearing = self._pairwise_distance_and_bearing(query_coords[:, 0], query_coords[:, 1], candidate_lat, candidate_lon)
            target_stations = self.targets.loc[target_rows, "station_id"].astype(str).to_numpy()[:, None]
            context_stations = self.contexts.loc[global_indices.reshape(-1), "station_id"].astype(str).to_numpy().reshape(global_indices.shape)
            valid = context_stations != target_stations
            order = np.argsort(~valid, axis=1, kind="stable")
            order = order[:, : min(self.candidate_k, query_k)]
            selected_indices = np.take_along_axis(global_indices, order, axis=1)
            selected_distance = np.take_along_axis(distance, order, axis=1)
            selected_bearing = np.take_along_axis(bearing, order, axis=1)
            selected_valid = np.take_along_axis(valid, order, axis=1)
            width = selected_indices.shape[1]
            self._candidate_indices[target_rows, :width] = np.where(selected_valid, selected_indices, -1)
            self._candidate_distance[target_rows, :width] = np.where(selected_valid, selected_distance, np.inf)
            self._candidate_bearing[target_rows, :width] = np.where(selected_valid, selected_bearing, 0.0)
        if cache_path is not None:
            self._save_neighbor_cache(cache_path)
            tqdm.write(f"[SRNP cache] Saved neighbor index: {cache_path}")

    def _neighbor_cache_path(self) -> Path | None:
        if self.cache_dir is None:
            return None
        identity_columns = ["station_id", "time", "lat", "lon", "split"]
        if "crossfit_fold" in self.targets:
            identity_columns.append("crossfit_fold")
        digest = hashlib.blake2b(digest_size=16)
        digest.update(b"hgb_srnp_neighbor_index_v2")
        digest.update(str(self.candidate_k).encode("ascii"))
        for frame in [self.targets, self.contexts]:
            values = pd.util.hash_pandas_object(frame[identity_columns], index=True).to_numpy(dtype=np.uint64)
            digest.update(values.tobytes())
        return self.cache_dir / f"neighbor_index_{digest.hexdigest()}.npz"

    def _load_neighbor_cache(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            with np.load(path, allow_pickle=False) as payload:
                indices = payload["indices"]
                distance = payload["distance"]
                bearing = payload["bearing"]
            expected = (len(self.targets), self.candidate_k)
            if indices.shape != expected or distance.shape != expected or bearing.shape != expected:
                return False
            self._candidate_indices = indices.astype(np.int32, copy=False)
            self._candidate_distance = distance.astype(np.float32, copy=False)
            self._candidate_bearing = bearing.astype(np.float32, copy=False)
            return True
        except (OSError, KeyError, ValueError):
            return False

    def _save_neighbor_cache(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.stem + ".tmp.npz")
        np.savez(
            temporary,
            indices=self._candidate_indices,
            distance=self._candidate_distance,
            bearing=self._candidate_bearing,
        )
        temporary.replace(path)

    def _materialize_epoch_cache(self, epoch: int) -> None:
        candidate_indices = self._candidate_indices
        valid = candidate_indices >= 0
        rng = np.random.default_rng(self.seed + int(epoch) * 1_000_003)
        keep = valid.copy()
        if not self.use_context:
            keep[:] = False
        elif self.training:
            policies = rng.choice(3, size=len(self.targets), p=self.mask_probabilities)
            random_rows = policies == 0
            if np.any(random_rows):
                fractions = rng.uniform(0.2, 0.8, size=int(random_rows.sum()))[:, None]
                keep[random_rows] &= rng.random((int(random_rows.sum()), self.candidate_k)) < fractions
            block_rows = policies == 1
            if np.any(block_rows):
                safe = np.maximum(candidate_indices[block_rows], 0)
                keep[block_rows] &= self._context_block[safe] != self._target_block[block_rows, None]
            buffer_rows = policies == 2
            if np.any(buffer_rows):
                thresholds = rng.choice(np.asarray([250.0, 500.0, 1000.0], dtype=np.float32), size=int(buffer_rows.sum()))[:, None]
                keep[buffer_rows] &= self._candidate_distance[buffer_rows] >= thresholds
        if self.context_fraction < 1.0:
            keep &= rng.random(keep.shape) < self.context_fraction
            original_valid = valid.any(axis=1)
            empty = original_valid & ~keep.any(axis=1)
            if np.any(empty):
                first = np.argmax(valid[empty], axis=1)
                keep[np.where(empty)[0], first] = True
        order = np.argsort(~keep, axis=1, kind="stable")[:, : self.k]
        selected_indices = np.take_along_axis(candidate_indices, order, axis=1)
        selected_valid = np.take_along_axis(keep, order, axis=1)
        selected_distance = np.take_along_axis(self._candidate_distance, order, axis=1)
        selected_bearing = np.take_along_axis(self._candidate_bearing, order, axis=1)
        safe_indices = np.maximum(selected_indices, 0)
        context = self._context_features[safe_indices].copy()
        for fold, fold_features in self._context_features_by_fold.items():
            target_rows = self._target_fold == int(fold)
            if np.any(target_rows):
                context[target_rows] = fold_features[safe_indices[target_rows]]
        context[~selected_valid] = 0.0
        relative = np.zeros((len(self.targets), self.k, 6), dtype=np.float32)
        if self.use_geometry:
            candidate_n = self._context_n[safe_indices]
            relative[:, :, 0] = np.log1p(np.where(selected_valid, selected_distance, 0.0) / 100.0)
            relative[:, :, 1] = np.sin(selected_bearing) * selected_valid
            relative[:, :, 2] = np.cos(selected_bearing) * selected_valid
            relative[:, :, 3:] = (candidate_n - self._target_n[:, None, :]) / np.asarray(self.scalers.n_std, dtype=np.float32)
            relative[~selected_valid] = 0.0
        count = selected_valid.sum(axis=1).astype(np.float32)
        min_distance = np.where(selected_valid, selected_distance, np.inf).min(axis=1).astype(np.float32)
        min_distance[~np.isfinite(min_distance)] = 1.0e6
        self._cache = {
            "query": self._query,
            "prior": self._prior,
            "context": torch.from_numpy(context),
            "relative": torch.from_numpy(relative),
            "context_mask": torch.from_numpy(selected_valid),
            "min_distance_km": torch.from_numpy(min_distance),
            "context_count": torch.from_numpy(count),
            "target_index": self._target_index,
            "target_fold": torch.from_numpy(self._target_fold.astype(np.int64)),
        }
        if self._target is not None:
            self._cache["target"] = self._target

    def _make_context_features(
        self,
        context_n: np.ndarray,
        context_height: np.ndarray,
        context_prior: np.ndarray,
        context_residual: np.ndarray,
    ) -> np.ndarray:
        innovation = context_residual - context_prior
        return np.concatenate(
            [
                self.scalers.normalize_n(context_n),
                self.scalers.normalize_height(context_height),
                self.scalers.normalize_residual(context_prior),
                self._calendar(self.contexts),
                self.scalers.normalize_residual(innovation),
            ],
            axis=1,
        ).astype(np.float32)

    def _columns(self, frame: pd.DataFrame, prefix: str) -> np.ndarray:
        return frame[[f"{prefix}_{layer}" for layer in self.layers]].to_numpy(dtype=np.float32)

    def _prior_columns(self, frame: pd.DataFrame, prefix: str) -> np.ndarray:
        if all(f"{prefix}_{layer}" in frame.columns for layer in self.layers):
            return self._columns(frame, prefix)
        return self._columns(frame, "prior_residual")

    @staticmethod
    def _calendar(frame: pd.DataFrame) -> np.ndarray:
        if frame.empty:
            return np.zeros((0, 7), dtype=np.float32)
        time = pd.to_datetime(frame["time"])
        lon_rad = np.deg2rad(frame["lon"].to_numpy(dtype=np.float32))
        return np.column_stack(
            [
                frame["lat"].to_numpy(dtype=np.float32) / 90.0,
                np.sin(lon_rad),
                np.cos(lon_rad),
                np.sin(2.0 * np.pi * time.dt.dayofyear.to_numpy(dtype=np.float32) / 366.0),
                np.cos(2.0 * np.pi * time.dt.dayofyear.to_numpy(dtype=np.float32) / 366.0),
                np.sin(2.0 * np.pi * time.dt.hour.to_numpy(dtype=np.float32) / 24.0),
                np.cos(2.0 * np.pi * time.dt.hour.to_numpy(dtype=np.float32) / 24.0),
            ]
        ).astype(np.float32)

    @staticmethod
    def _block_ids(frame: pd.DataFrame) -> np.ndarray:
        if frame.empty:
            return np.zeros(0, dtype=np.int32)
        lat_bin = np.floor((frame["lat"].to_numpy(dtype=float) + 90.0) / 30.0).astype(np.int32)
        lon_bin = np.floor((frame["lon"].to_numpy(dtype=float) + 180.0) / 60.0).astype(np.int32)
        return lat_bin * 10 + lon_bin

    @staticmethod
    def _pairwise_distance_and_bearing(
        target_lat: np.ndarray, target_lon: np.ndarray, context_lat: np.ndarray, context_lon: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        lat1 = np.deg2rad(target_lat)[:, None]
        lon1 = np.deg2rad(target_lon)[:, None]
        lat2 = np.deg2rad(context_lat)
        lon2 = np.deg2rad(context_lon)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        distance = EARTH_RADIUS_KM * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        return distance.astype(np.float32), np.arctan2(y, x).astype(np.float32)


def select_profile_targets(
    profiles: pd.DataFrame,
    split: str,
    max_times: int | None = None,
    max_targets_per_time: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    out = profiles[profiles["split"].eq(split)].copy()
    if max_times is not None:
        times = sorted(out["time"].drop_duplicates().tolist())[: int(max_times)]
        out = out[out["time"].isin(times)]
    if max_targets_per_time is not None:
        out = (
            out.groupby("time", group_keys=False, sort=False)
            .sample(n=min(int(max_targets_per_time), max(1, out.groupby("time").size().min())), random_state=seed)
            .reset_index(drop=True)
        )
    return out.reset_index(drop=True)
