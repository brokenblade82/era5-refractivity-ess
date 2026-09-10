from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from igra_forecast.data import ExperimentData


class ForecastModel(ABC):
    def __init__(self, name: str, params: dict | None = None) -> None:
        self.name = name
        self.params = params or {}

    @abstractmethod
    def fit(self, data: ExperimentData) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, data: ExperimentData, split: str = "test") -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError
