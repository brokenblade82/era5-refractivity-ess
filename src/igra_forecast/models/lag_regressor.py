from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

from igra_forecast.data import ExperimentData, predictions_to_frame
from igra_forecast.logging_utils import info
from igra_forecast.models.base import ForecastModel
from igra_forecast.models.common import save_pickle


class LagFeatureRegressor(ForecastModel):
    def fit(self, data: ExperimentData) -> None:
        self.estimator_name = self.name
        info(f"开始拟合 lag-feature 模型：{self.name}")
        if self.name == "lag_ridge":
            estimator = Ridge(alpha=float(self.params.get("alpha", 1.0)))
        elif self.name == "lag_hgb":
            estimator = HistGradientBoostingRegressor(
                max_iter=int(self.params.get("max_iter", 200)),
                learning_rate=float(self.params.get("learning_rate", 0.05)),
                max_leaf_nodes=int(self.params.get("max_leaf_nodes", 31)),
                min_samples_leaf=int(self.params.get("min_samples_leaf", 20)),
                l2_regularization=float(self.params.get("l2_regularization", 0.0)),
                early_stopping=bool(self.params.get("early_stopping", True)),
                validation_fraction=float(self.params.get("validation_fraction", 0.1)),
                random_state=42,
            )
        elif self.name in {"lag_lightgbm", "lag_xgboost"}:
            raise ImportError(
                f"{self.name} is reserved but the dependency is not installed. "
                "Use lag_ridge or lag_hgb first, or install the package explicitly."
            )
        else:
            raise ValueError(f"Unsupported lag-feature regressor: {self.name}")

        self.model = MultiOutputRegressor(estimator)
        x_train = self._flatten_x(data.train.x)
        y_train = self._flatten_y(data.train.y)
        info(f"监督特征矩阵：X={x_train.shape}, y={y_train.shape}")
        self.model.fit(x_train, y_train)
        info(f"lag-feature 模型拟合完成：{self.name}")

    def predict(self, data: ExperimentData, split: str = "test") -> pd.DataFrame:
        split_data = data.splits[split]
        pred_flat = self.model.predict(self._flatten_x(split_data.x))
        y_pred = pred_flat.reshape(len(split_data.x), len(data.horizons), len(data.target_cols))
        return predictions_to_frame(data, split, split_data.y, y_pred.astype(np.float32))

    def save(self, path: Path) -> None:
        save_pickle(self, path)

    @staticmethod
    def _flatten_x(x: np.ndarray) -> np.ndarray:
        return x.reshape(x.shape[0], -1)

    @staticmethod
    def _flatten_y(y: np.ndarray) -> np.ndarray:
        return y.reshape(y.shape[0], -1)
