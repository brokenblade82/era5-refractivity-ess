from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from igra_forecast.data import ExperimentData, predictions_to_frame
from igra_forecast.logging_utils import info
from igra_forecast.models.base import ForecastModel
from igra_forecast.models.common import save_pickle


class PersistenceModel(ForecastModel):
    def fit(self, data: ExperimentData) -> None:
        self.target_input_indices = [data.input_cols.index(col) for col in data.target_cols]
        info("Persistence 不需要参数训练，已定位目标变量在输入中的索引")

    def predict(self, data: ExperimentData, split: str = "test") -> pd.DataFrame:
        split_data = data.splits[split]
        last_values = split_data.x[:, -1, self.target_input_indices]
        y_pred = np.repeat(last_values[:, None, :], len(data.horizons), axis=1)
        return predictions_to_frame(data, split, split_data.y, y_pred)

    def save(self, path: Path) -> None:
        save_pickle(self, path)
