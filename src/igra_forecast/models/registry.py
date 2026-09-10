from __future__ import annotations

from typing import Any

from igra_forecast.models.base import ForecastModel
from igra_forecast.models.adaptive_ensemble import AdaptiveEnsembleModel
from igra_forecast.models.climatology import ClimatologyModel
from igra_forecast.models.deep import DEEP_MODEL_NAMES, DeepForecastModel
from igra_forecast.models.lag_regressor import LagFeatureRegressor
from igra_forecast.models.persistence import PersistenceModel
from igra_forecast.models.statistical import SarimaxModel, VarModel


def build_model(cfg: dict[str, Any]) -> ForecastModel:
    name = cfg["model"]["name"]
    params = _model_params(cfg, name)
    if name == "persistence":
        return PersistenceModel(name=name, params=params)
    if name == "climatology":
        return ClimatologyModel(name=name, params=params)
    if name == "sarimax":
        return SarimaxModel(name=name, params=params)
    if name == "var":
        return VarModel(name=name, params=params)
    if name in {"lag_ridge", "lag_hgb", "lag_lightgbm", "lag_xgboost"}:
        return LagFeatureRegressor(name=name, params=params)
    if name == "adaptive_ensemble":
        return AdaptiveEnsembleModel(name=name, params=params)
    if name in DEEP_MODEL_NAMES:
        return DeepForecastModel(name=name, params=params)
    raise ValueError(f"Unknown model name: {name}")


def _model_params(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    params = dict(cfg.get("model", {}).get("params", {}) or {})
    params.update(dict(cfg.get("traditional_models", {}).get(name, {}) or {}))
    if name in DEEP_MODEL_NAMES:
        params = {
            "common": dict(cfg.get("deep_models", {}).get("common", {}) or {}),
            name: dict(cfg.get("deep_models", {}).get(name, {}) or {}),
            **params,
        }
    return params
