from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from igra_forecast.data import ExperimentData, predictions_to_frame
from igra_forecast.logging_utils import info
from igra_forecast.models.base import ForecastModel
from igra_forecast.models.common import save_pickle


class ClimatologyModel(ForecastModel):
    def fit(self, data: ExperimentData) -> None:
        self.groupby = self.params.get("groupby", "dayofyear")
        info(f"Climatology 分组方式：{self.groupby}")
        train = data.train.frame.copy()
        if self.groupby == "dayofyear":
            train["_clim_key"] = pd.to_datetime(train["datetime"]).dt.dayofyear
        elif self.groupby == "month":
            train["_clim_key"] = pd.to_datetime(train["datetime"]).dt.month
        else:
            raise ValueError(f"Unsupported climatology grouping: {self.groupby}")
        self.target_cols = data.target_cols
        self.climatology = train.groupby("_clim_key")[self.target_cols].mean()
        self.global_mean = train[self.target_cols].mean().to_numpy(dtype=np.float32)
        info(f"Climatology 已拟合，分组数：{len(self.climatology)}")

    def predict(self, data: ExperimentData, split: str = "test") -> pd.DataFrame:
        split_data = data.splits[split]
        y_pred = np.zeros_like(split_data.y, dtype=np.float32)
        for sample_idx, issue_time in enumerate(split_data.issue_times):
            for horizon_idx, horizon in enumerate(data.horizons):
                target_time = issue_time + pd.Timedelta(days=int(horizon))
                if self.groupby == "dayofyear":
                    key = target_time.dayofyear
                else:
                    key = target_time.month
                if key in self.climatology.index:
                    y_pred[sample_idx, horizon_idx, :] = self.climatology.loc[key].to_numpy(dtype=np.float32)
                else:
                    y_pred[sample_idx, horizon_idx, :] = self.global_mean
        return predictions_to_frame(data, split, split_data.y, y_pred)

    def save(self, path: Path) -> None:
        save_pickle(self, path)
