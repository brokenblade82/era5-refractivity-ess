from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX

from igra_forecast.data import ExperimentData, predictions_to_frame
from igra_forecast.logging_utils import info
from igra_forecast.models.base import ForecastModel
from igra_forecast.models.common import save_pickle


class SarimaxModel(ForecastModel):
    def fit(self, data: ExperimentData) -> None:
        self.target_cols = data.target_cols
        self.horizons = data.horizons
        max_samples = self.params.get("max_train_samples")
        train = data.train.frame[self.target_cols]
        if max_samples:
            train = train.iloc[-int(max_samples) :]
        order = tuple(self.params.get("order", [1, 0, 1]))
        seasonal_order = tuple(self.params.get("seasonal_order", [0, 0, 0, 0]))
        info(f"SARIMAX 拟合参数：order={order}, seasonal_order={seasonal_order}, 样本数={len(train)}")
        self.models = {}
        for target in tqdm(self.target_cols, desc="拟合 SARIMAX 目标变量", unit="target"):
            fitted = SARIMAX(
                train[target],
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            self.models[target] = fitted
        info("SARIMAX 初始拟合完成")

    def predict(self, data: ExperimentData, split: str = "test") -> pd.DataFrame:
        split_data = data.splits[split]
        history = pd.concat([data.train.frame, data.val.frame], axis=0, ignore_index=True)
        if split == "val":
            history = data.train.frame.copy()
        elif split == "train":
            history = data.train.frame.iloc[: data.input_length].copy()

        y_pred = np.zeros_like(split_data.y, dtype=np.float32)
        combined = pd.concat([history, split_data.frame], axis=0, ignore_index=True)
        split_offset = len(history)
        for sample_idx, issue_time in enumerate(tqdm(split_data.issue_times, desc=f"SARIMAX 递推预测 {split}", unit="window")):
            issue_pos = split_offset + split_data.frame.index[split_data.frame["datetime"].eq(issue_time)][0]
            for target_idx, target in enumerate(self.target_cols):
                series = combined.loc[:issue_pos, target]
                try:
                    fitted = SARIMAX(
                        series,
                        order=tuple(self.params.get("order", [1, 0, 1])),
                        seasonal_order=tuple(self.params.get("seasonal_order", [0, 0, 0, 0])),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False, maxiter=50)
                    forecast = fitted.forecast(steps=max(data.horizons))
                    for horizon_idx, horizon in enumerate(data.horizons):
                        y_pred[sample_idx, horizon_idx, target_idx] = float(forecast.iloc[horizon - 1])
                except Exception:
                    y_pred[sample_idx, :, target_idx] = float(series.iloc[-1])
        return predictions_to_frame(data, split, split_data.y, y_pred)

    def save(self, path: Path) -> None:
        save_pickle({"name": self.name, "params": self.params, "models": self.models}, path)


class VarModel(ForecastModel):
    def fit(self, data: ExperimentData) -> None:
        train_all = data.train.frame[data.input_cols]
        self.input_cols = [col for col in data.input_cols if float(train_all[col].std()) > 1e-8]
        self.target_cols = data.target_cols
        missing_targets = [col for col in self.target_cols if col not in self.input_cols]
        if missing_targets:
            raise ValueError(f"VAR requires non-constant target columns, missing: {missing_targets}")
        max_samples = self.params.get("max_train_samples")
        train = data.train.frame[self.input_cols]
        if max_samples:
            train = train.iloc[-int(max_samples) :]
        info(
            f"VAR 输入变量数：{len(self.input_cols)}，目标变量数：{len(self.target_cols)}，"
            f"拟合样本数：{len(train)}, maxlags={self.params.get('maxlags', 14)}, ic={self.params.get('ic', 'aic')}"
        )
        model = VAR(train)
        self.fitted = model.fit(maxlags=int(self.params.get("maxlags", 14)), ic=self.params.get("ic", "aic"))
        info(f"VAR 拟合完成，实际滞后阶数 k_ar={getattr(self.fitted, 'k_ar', 'unknown')}")

    def predict(self, data: ExperimentData, split: str = "test") -> pd.DataFrame:
        split_data = data.splits[split]
        history = pd.concat([data.train.frame, data.val.frame], axis=0, ignore_index=True)
        if split == "val":
            history = data.train.frame.copy()
        elif split == "train":
            history = data.train.frame.iloc[: data.input_length].copy()
        combined = pd.concat([history, split_data.frame], axis=0, ignore_index=True)
        split_offset = len(history)
        target_indices = [self.input_cols.index(col) for col in data.target_cols]
        lag_order = max(int(getattr(self.fitted, "k_ar", 1)), 1)
        y_pred = np.zeros_like(split_data.y, dtype=np.float32)

        for sample_idx, issue_time in enumerate(tqdm(split_data.issue_times, desc=f"VAR 递推预测 {split}", unit="window")):
            issue_pos = split_offset + split_data.frame.index[split_data.frame["datetime"].eq(issue_time)][0]
            start = max(0, issue_pos - lag_order + 1)
            recent = combined.loc[start:issue_pos, self.input_cols].to_numpy(dtype=float)
            if len(recent) < lag_order:
                pad = np.repeat(recent[:1], lag_order - len(recent), axis=0)
                recent = np.vstack([pad, recent])
            try:
                forecast = self.fitted.forecast(recent[-lag_order:], steps=max(data.horizons))
                for horizon_idx, horizon in enumerate(data.horizons):
                    y_pred[sample_idx, horizon_idx, :] = forecast[horizon - 1, target_indices]
            except Exception:
                last = recent[-1, target_indices]
                y_pred[sample_idx, :, :] = last
        return predictions_to_frame(data, split, split_data.y, y_pred)

    def save(self, path: Path) -> None:
        save_pickle({"name": self.name, "params": self.params, "fitted": self.fitted}, path)
