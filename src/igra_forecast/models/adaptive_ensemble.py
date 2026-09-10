from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from igra_forecast.data import ExperimentData, SplitArrays, predictions_to_frame
from igra_forecast.logging_utils import info
from igra_forecast.models.base import ForecastModel
from igra_forecast.models.common import save_pickle


class AdaptiveEnsembleModel(ForecastModel):
    """Target-horizon adaptive convex ensemble for small meteorological series."""

    def fit(self, data: ExperimentData) -> None:
        self.target_cols = list(data.target_cols)
        self.horizons = list(data.horizons)
        self.target_input_indices = [data.input_cols.index(col) for col in data.target_cols]
        self.candidate_names = list(
            self.params.get(
                "candidates",
                ["persistence", "climatology", "mean7", "mean30", "trend7", "lag_ridge"],
            )
        )
        self.meta_alpha = float(self.params.get("meta_alpha", 1e-3))
        self.ridge_alpha = float(self.params.get("ridge_alpha", 10.0))
        self.nonnegative = bool(self.params.get("nonnegative", True))
        self.sum_to_one = bool(self.params.get("sum_to_one", True))
        self.fallback_to_best_candidate = bool(self.params.get("fallback_to_best_candidate", True))
        self.trend_clip = float(self.params.get("trend_clip", 3.0))
        self.weight_mode = str(self.params.get("weight_mode", "target_horizon")).lower()
        self.training_history = None

        info(f"Adaptive ensemble candidates: {', '.join(self.candidate_names)}")
        info(f"Adaptive ensemble weight mode: {self.weight_mode}")
        if "lag_ridge" in self.candidate_names:
            self._fit_lag_ridge(data)
        else:
            self.lag_ridge = None

        val_candidates = self._candidate_cube(data.val)
        self.weights = self._fit_weights(val_candidates, data.val.y)
        self.weight_table = self._weights_to_frame()
        info("Adaptive ensemble fitted weights on validation split")

    def predict(self, data: ExperimentData, split: str = "test") -> pd.DataFrame:
        split_data = data.splits[split]
        candidates = self._candidate_cube(split_data)
        y_pred = np.einsum("nhtc,htc->nht", candidates, self.weights).astype(np.float32)
        return predictions_to_frame(data, split, split_data.y, y_pred)

    def save(self, path: Path) -> None:
        save_pickle(self, path)
        if hasattr(self, "weight_table"):
            self.weight_table.to_csv(path / "target_horizon_weights.csv", index=False)

    def _fit_lag_ridge(self, data: ExperimentData) -> None:
        self.lag_ridge = Ridge(alpha=self.ridge_alpha)
        x_train = data.train.x.reshape(data.train.x.shape[0], -1)
        y_train = data.train.y.reshape(data.train.y.shape[0], -1)
        info(f"Fitting lag-ridge candidate: X={x_train.shape}, y={y_train.shape}, alpha={self.ridge_alpha}")
        self.lag_ridge.fit(x_train, y_train)

    def _candidate_cube(self, split: SplitArrays) -> np.ndarray:
        candidates = []
        for name in self.candidate_names:
            if name == "persistence":
                arr = split.persistence_prior
            elif name == "climatology":
                arr = split.climatology_prior
            elif name.startswith("mean"):
                window = int(name.replace("mean", ""))
                arr = self._recent_mean(split, window)
            elif name.startswith("trend"):
                window = int(name.replace("trend", ""))
                arr = self._trend_extrapolation(split, window)
            elif name == "lag_ridge":
                if self.lag_ridge is None:
                    raise ValueError("lag_ridge candidate requested before fitting.")
                flat = self.lag_ridge.predict(split.x.reshape(split.x.shape[0], -1))
                arr = flat.reshape(len(split.x), len(self.horizons), len(self.target_cols))
            else:
                raise ValueError(f"Unsupported adaptive ensemble candidate: {name}")
            candidates.append(arr.astype(np.float32))
        return np.stack(candidates, axis=-1)

    def _recent_mean(self, split: SplitArrays, window: int) -> np.ndarray:
        window = max(1, min(window, split.x.shape[1]))
        values = split.x[:, -window:, self.target_input_indices].mean(axis=1)
        return np.repeat(values[:, None, :], len(self.horizons), axis=1)

    def _trend_extrapolation(self, split: SplitArrays, window: int) -> np.ndarray:
        window = max(1, min(window, split.x.shape[1] - 1))
        target_series = split.x[:, :, self.target_input_indices]
        last = target_series[:, -1, :]
        previous = target_series[:, -(window + 1), :]
        slope = np.clip((last - previous) / float(window), -self.trend_clip, self.trend_clip)
        outputs = []
        for horizon in self.horizons:
            outputs.append(last + float(horizon) * slope)
        return np.stack(outputs, axis=1)

    def _fit_target_horizon_weights(self, candidates: np.ndarray, truth: np.ndarray) -> np.ndarray:
        _, n_horizons, n_targets, n_candidates = candidates.shape
        weights = np.zeros((n_horizons, n_targets, n_candidates), dtype=np.float32)
        for h_idx in range(n_horizons):
            for t_idx in range(n_targets):
                x = candidates[:, h_idx, t_idx, :]
                y = truth[:, h_idx, t_idx]
                w = self._solve_weights(x, y)
                if self.fallback_to_best_candidate:
                    w = self._fallback_if_needed(x, y, w)
                weights[h_idx, t_idx, :] = w
        return weights

    def _fit_weights(self, candidates: np.ndarray, truth: np.ndarray) -> np.ndarray:
        _, n_horizons, n_targets, n_candidates = candidates.shape
        if self.weight_mode == "target_horizon":
            return self._fit_target_horizon_weights(candidates, truth)
        if self.weight_mode == "equal":
            return np.full((n_horizons, n_targets, n_candidates), 1.0 / n_candidates, dtype=np.float32)
        if self.weight_mode == "global":
            x = candidates.reshape(-1, n_candidates)
            y = truth.reshape(-1)
            w = self._solve_weights(x, y)
            if self.fallback_to_best_candidate:
                w = self._fallback_if_needed(x, y, w)
            return np.broadcast_to(w, (n_horizons, n_targets, n_candidates)).astype(np.float32)
        if self.weight_mode == "target_only":
            weights = np.zeros((n_horizons, n_targets, n_candidates), dtype=np.float32)
            for t_idx in range(n_targets):
                x = candidates[:, :, t_idx, :].reshape(-1, n_candidates)
                y = truth[:, :, t_idx].reshape(-1)
                w = self._solve_weights(x, y)
                if self.fallback_to_best_candidate:
                    w = self._fallback_if_needed(x, y, w)
                weights[:, t_idx, :] = w
            return weights
        if self.weight_mode == "horizon_only":
            weights = np.zeros((n_horizons, n_targets, n_candidates), dtype=np.float32)
            for h_idx in range(n_horizons):
                x = candidates[:, h_idx, :, :].reshape(-1, n_candidates)
                y = truth[:, h_idx, :].reshape(-1)
                w = self._solve_weights(x, y)
                if self.fallback_to_best_candidate:
                    w = self._fallback_if_needed(x, y, w)
                weights[h_idx, :, :] = w
            return weights
        raise ValueError(
            "Unsupported adaptive ensemble weight_mode: "
            f"{self.weight_mode}. Use target_horizon, equal, global, target_only, or horizon_only."
        )

    def _solve_weights(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xtx = x.T @ x
        penalty = self.meta_alpha * np.eye(xtx.shape[0], dtype=np.float64)
        rhs = x.T @ y
        try:
            w = np.linalg.solve(xtx + penalty, rhs)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(xtx + penalty) @ rhs
        if self.nonnegative:
            w = np.clip(w, 0.0, None)
        if self.sum_to_one:
            total = float(w.sum())
            if total <= 1e-8:
                best_idx = int(np.argmin(((x - y[:, None]) ** 2).mean(axis=0)))
                w = np.zeros_like(w)
                w[best_idx] = 1.0
            else:
                w = w / total
        return w.astype(np.float32)

    @staticmethod
    def _fallback_if_needed(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        ensemble_mse = float(np.mean((x @ weights - y) ** 2))
        candidate_mse = ((x - y[:, None]) ** 2).mean(axis=0)
        best_idx = int(np.argmin(candidate_mse))
        if ensemble_mse <= float(candidate_mse[best_idx]):
            return weights
        fallback = np.zeros_like(weights)
        fallback[best_idx] = 1.0
        return fallback

    def _weights_to_frame(self) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for h_idx, horizon in enumerate(self.horizons):
            for t_idx, target in enumerate(self.target_cols):
                for c_idx, candidate in enumerate(self.candidate_names):
                    records.append(
                        {
                            "horizon": int(horizon),
                            "target": target,
                            "candidate": candidate,
                            "weight_mode": self.weight_mode,
                            "weight": float(self.weights[h_idx, t_idx, c_idx]),
                        }
                    )
        return pd.DataFrame.from_records(records)
