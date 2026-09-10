from __future__ import annotations

import math
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr
from scipy.stats import chi2
# joblib/loky occasionally cannot query physical cores on recent Windows
# systems.  A conservative physical-core approximation prevents loky from
# invoking the failing Windows probe.  Revision-2 HGB uses OpenMP internally,
# so this does not change its fitted values.
if not os.environ.get("LOKY_MAX_CPU_COUNT", "").isdigit() or int(os.environ.get("LOKY_MAX_CPU_COUNT", "0")) < 1:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(max(1, min(60, (os.cpu_count() or 2) // 2)))
warnings.filterwarnings("ignore", message=r"Could not find the number of physical cores.*", category=UserWarning)

from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


LEVELS = np.asarray([1000, 925, 850, 700, 500, 300], dtype=np.int16)
COMPONENTS = ("dry", "wet", "total")


@dataclass
class ProfilePrediction:
    mean_dry: np.ndarray
    mean_wet: np.ndarray
    mean_total: np.ndarray
    std_dry: np.ndarray | None
    std_wet: np.ndarray | None
    std_total: np.ndarray | None
    covariance_total: np.ndarray | None
    level_mask: np.ndarray

    def validate(self, atol: float = 1e-8) -> None:
        expected = np.asarray(self.mean_dry) + np.asarray(self.mean_wet)
        if not np.allclose(self.mean_total, expected, atol=atol, equal_nan=True):
            raise ValueError("ProfilePrediction violates mean_total = mean_dry + mean_wet")
        if np.asarray(self.level_mask).shape != np.asarray(self.mean_total).shape:
            raise ValueError("level_mask shape does not match the predictive mean")
        if self.std_total is not None and np.asarray(self.std_total).shape != np.asarray(self.mean_total).shape:
            raise ValueError("std_total shape does not match the predictive mean")
        if self.covariance_total is not None:
            expected_shape = (*np.asarray(self.mean_total).shape, np.asarray(self.mean_total).shape[-1])
            if np.asarray(self.covariance_total).shape != expected_shape:
                raise ValueError("covariance_total shape does not match the profile dimensions")


@dataclass
class ProfileArrays:
    keys: pd.DataFrame
    features: np.ndarray
    target_dry: np.ndarray
    target_wet: np.ndarray
    target_total: np.ndarray
    mask: np.ndarray
    row_frame: pd.DataFrame


def make_row_features(frame: pd.DataFrame) -> np.ndarray:
    time = pd.DatetimeIndex(pd.to_datetime(frame["time"], utc=True))
    day = time.dayofyear.to_numpy(dtype=float)
    hour = time.hour.to_numpy(dtype=float)
    lat = np.deg2rad(frame["latitude"].to_numpy(float))
    lon = np.deg2rad(frame["longitude"].to_numpy(float))
    pressure = frame["pressure_hpa"].to_numpy(float)
    sp = frame["era5_surface_pressure_pa"].to_numpy(float) / 100.0
    columns = [
        frame["era5_temperature_k"].to_numpy(float),
        frame["era5_specific_humidity"].to_numpy(float),
        frame["era5_height_m"].to_numpy(float) / 10000.0,
        frame["era5_n_dry"].to_numpy(float),
        frame["era5_n_wet"].to_numpy(float),
        frame["era5_n"].to_numpy(float),
        pressure / 1000.0,
        sp / 1000.0,
        np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon),
        np.sin(2 * np.pi * day / 365.25), np.cos(2 * np.pi * day / 365.25),
        np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0),
        frame["below_ground_level"].astype(float).to_numpy(),
        frame["level_mask"].astype(float).to_numpy(),
    ]
    result = np.column_stack(columns).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Revision-2 input features contain non-finite values")
    return result


def rows_to_profiles(frame: pd.DataFrame, levels: np.ndarray = LEVELS) -> ProfileArrays:
    expected = list(map(int, levels))
    frame = frame.copy()
    frame["pressure_hpa"] = frame["pressure_hpa"].astype(int)
    frame["_level_order"] = pd.Categorical(frame["pressure_hpa"], categories=expected, ordered=True)
    frame = frame.sort_values(["station_id", "time", "_level_order"]).drop(columns="_level_order").reset_index(drop=True)
    sizes = frame.groupby(["station_id", "time"], observed=True).size()
    if not bool((sizes == len(levels)).all()):
        raise ValueError("Every profile must contain exactly the configured six rows")
    if frame.duplicated(["station_id", "time", "pressure_hpa"]).any():
        raise ValueError("Duplicate station-time-level rows are not allowed")
    actual = frame.groupby(["station_id", "time"], observed=True)["pressure_hpa"].apply(list)
    if not actual.map(lambda value: value == expected).all():
        raise ValueError("Profile pressure-level order does not match the configured six levels")
    n_profile = len(sizes)
    features = make_row_features(frame).reshape(n_profile, len(levels), -1)
    target_dry = frame["residual_dry"].to_numpy(float).reshape(n_profile, len(levels))
    target_wet = frame["residual_wet"].to_numpy(float).reshape(n_profile, len(levels))
    target_total = frame["residual_total"].to_numpy(float).reshape(n_profile, len(levels))
    mask = (
        frame["level_mask"].astype(bool).to_numpy()
        & ~frame["below_ground_level"].astype(bool).to_numpy()
        & np.isfinite(frame["residual_total"].to_numpy(float))
    ).reshape(n_profile, len(levels))
    keys = frame[["station_id", "time", "latitude", "longitude"]].iloc[:: len(levels)].reset_index(drop=True)
    result = ProfileArrays(keys, features, target_dry, target_wet, target_total, mask, frame)
    identity = np.abs((target_dry + target_wet - target_total)[mask])
    if identity.size and float(identity.max()) > 1e-8:
        raise ValueError("Observed dry/wet/total residual identity failed")
    return result


def row_training_mask(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["level_mask"].astype(bool).to_numpy()
        & ~frame["below_ground_level"].astype(bool).to_numpy()
        & np.isfinite(frame["residual_dry"].to_numpy(float))
        & np.isfinite(frame["residual_wet"].to_numpy(float))
    )


def hgb_factory(settings: dict[str, Any]) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=float(settings.get("learning_rate", 0.05)),
        max_iter=int(settings.get("max_iter", 300)),
        max_leaf_nodes=int(settings.get("max_leaf_nodes", 31)),
        min_samples_leaf=int(settings.get("min_samples_leaf", 30)),
        l2_regularization=float(settings.get("l2_regularization", 0.1)),
        random_state=int(settings.get("random_state", 42)),
    )


def fit_component_regressors(model_name: str, train: pd.DataFrame, settings: dict[str, Any]) -> dict[str, Any]:
    features = make_row_features(train)
    valid = row_training_mask(train)
    if model_name == "ridge":
        factory = lambda: make_pipeline(StandardScaler(), Ridge(alpha=float(settings.get("alpha", 1.0)), solver=str(settings.get("solver", "lsqr"))))
    elif model_name == "hgb":
        factory = lambda: hgb_factory(settings)
    else:
        raise ValueError(f"Unsupported component regressor: {model_name}")
    models = {}
    for component in tqdm(("dry", "wet"), desc=f"Fit {model_name} components", leave=False):
        model = factory()
        model.fit(features[valid], train.loc[valid, f"residual_{component}"].to_numpy(float))
        models[component] = model
    return models


def predict_component_regressors(models: dict[str, Any], frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    features = make_row_features(frame)
    return np.asarray(models["dry"].predict(features)), np.asarray(models["wet"].predict(features))


def station_crossfit_folds(stations: pd.Series, n_splits: int, seed: int) -> np.ndarray:
    unique = np.asarray(sorted(map(str, stations.unique())))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    fold_lookup = {station: index % n_splits for index, station in enumerate(unique)}
    return stations.astype(str).map(fold_lookup).to_numpy(int)


def fit_probabilistic_hgb(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    hgb_settings: dict[str, Any],
    probabilistic_settings: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    x = make_row_features(train)
    valid = row_training_mask(train)
    train_valid = train.loc[valid].reset_index(drop=True)
    x_valid = x[valid]
    folds = station_crossfit_folds(train_valid["station_id"], int(probabilistic_settings.get("crossfit_folds", 5)), seed)
    result: dict[str, Any] = {"folds": folds, "components": {}}
    floor = float(probabilistic_settings.get("scale_floor", 0.05))
    log_min = float(probabilistic_settings.get("log_scale_min", -5.0))
    log_max = float(probabilistic_settings.get("log_scale_max", 4.0))
    x_cal = make_row_features(calibration)
    cal_valid = row_training_mask(calibration)
    for component in ("dry", "wet"):
        y = train_valid[f"residual_{component}"].to_numpy(float)
        oof = np.full(len(y), np.nan)
        for fold in tqdm(sorted(np.unique(folds)), desc=f"Probability HGB cross-fit {component}", leave=False):
            mean_model = hgb_factory({**hgb_settings, "random_state": seed + int(fold)})
            mean_model.fit(x_valid[folds != fold], y[folds != fold])
            oof[folds == fold] = mean_model.predict(x_valid[folds == fold])
        if not np.isfinite(oof).all():
            raise ValueError("Probability HGB cross-fit left missing prior predictions")
        scale_target = np.log(np.maximum(np.abs(y - oof), floor))
        scale_model = hgb_factory({**hgb_settings, "random_state": seed + 100})
        scale_model.fit(x_valid, scale_target)
        mean_model = hgb_factory({**hgb_settings, "random_state": seed})
        mean_model.fit(x_valid, y)
        raw_std_cal = np.exp(np.clip(scale_model.predict(x_cal[cal_valid]), log_min, log_max))
        error_cal = calibration.loc[cal_valid, f"residual_{component}"].to_numpy(float) - mean_model.predict(x_cal[cal_valid])
        calibration_scale = calibrate_standard_deviation(error_cal, raw_std_cal, coverage=0.90)
        result["components"][component] = {
            "mean": mean_model, "scale": scale_model, "calibration_scale": calibration_scale,
            "log_scale_bounds": [log_min, log_max],
        }
    return result


def predict_probabilistic_hgb(model: dict[str, Any], frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = make_row_features(frame)
    outputs = []
    for component in ("dry", "wet"):
        item = model["components"][component]
        mean = item["mean"].predict(x)
        raw = np.exp(np.clip(item["scale"].predict(x), *item["log_scale_bounds"]))
        outputs.extend([mean, raw * float(item["calibration_scale"])])
    return tuple(np.asarray(value) for value in outputs)  # type: ignore[return-value]


def calibrate_standard_deviation(error: np.ndarray, std: np.ndarray, coverage: float = 0.90) -> float:
    valid = np.isfinite(error) & np.isfinite(std) & (std > 0)
    if not valid.any():
        return 1.0
    z = float(torch.distributions.Normal(0, 1).icdf(torch.tensor((1 + coverage) / 2)).item())
    return max(float(np.quantile(np.abs(error[valid]) / std[valid], coverage) / z), 1e-6)


class ProfileGaussianMLP(nn.Module):
    def __init__(self, feature_dim: int, levels: int = 6, width: int = 128, layers: int = 2, dropout: float = 0.1, extra_dim: int = 0):
        super().__init__()
        input_dim = levels * (feature_dim + 2) + levels + extra_dim
        blocks: list[nn.Module] = []
        current = input_dim
        for _ in range(layers):
            blocks.extend([nn.Linear(current, width), nn.GELU(), nn.Dropout(dropout)])
            current = width
        self.backbone = nn.Sequential(*blocks)
        self.output = nn.Linear(current, levels * 4)
        self.levels = levels

    def forward(self, features: torch.Tensor, mask: torch.Tensor, prior: torch.Tensor, extra: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        values = [features.reshape(features.shape[0], -1), prior.reshape(prior.shape[0], -1), mask.float()]
        if extra is not None:
            values.append(extra)
        hidden = self.backbone(torch.cat(values, dim=-1))
        output = self.output(hidden).view(features.shape[0], self.levels, 4)
        mean = prior + output[..., :2]
        log_std = output[..., 2:].clamp(-5.0, 4.0)
        return mean, log_std


def masked_gaussian_nll(mean: torch.Tensor, log_std: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    safe_target = torch.where(mask.unsqueeze(-1), target, mean.detach())
    term = 0.5 * ((safe_target - mean) / torch.exp(log_std)).square() + log_std
    expanded = mask.unsqueeze(-1).expand_as(term)
    return term[expanded].mean()


def rbf_features(latitude: np.ndarray, longitude: np.ndarray, anchors: np.ndarray, scale_km: float) -> np.ndarray:
    lat1 = np.deg2rad(np.asarray(latitude, dtype=float))[:, None]
    lon1 = np.deg2rad(np.asarray(longitude, dtype=float))[:, None]
    lat2 = np.deg2rad(anchors[:, 0])[None, :]
    lon2 = np.deg2rad(anchors[:, 1])[None, :]
    a = np.sin((lat1 - lat2) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon1 - lon2) / 2) ** 2
    distance = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return np.exp(-0.5 * np.square(distance / float(scale_km))).astype(np.float32)


def choose_rbf_anchors(keys: pd.DataFrame, count: int, seed: int) -> np.ndarray:
    locations = keys[["latitude", "longitude"]].drop_duplicates().to_numpy(float)
    if len(locations) <= count:
        return locations
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(len(locations)))]
    xyz = np.column_stack([
        np.cos(np.deg2rad(locations[:, 0])) * np.cos(np.deg2rad(locations[:, 1])),
        np.cos(np.deg2rad(locations[:, 0])) * np.sin(np.deg2rad(locations[:, 1])),
        np.sin(np.deg2rad(locations[:, 0])),
    ])
    min_distance = np.full(len(locations), np.inf)
    for _ in range(1, count):
        min_distance = np.minimum(min_distance, np.square(xyz - xyz[selected[-1]]).sum(axis=1))
        selected.append(int(np.argmax(min_distance)))
    return locations[selected]


def _network_tensors(
    arrays: ProfileArrays,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    prior: np.ndarray,
    extra: np.ndarray | None,
) -> TensorDataset:
    features = (arrays.features - feature_mean) / feature_std
    target = np.stack([arrays.target_dry, arrays.target_wet], axis=-1)
    tensors = [
        torch.from_numpy(features.astype(np.float32)), torch.from_numpy(arrays.mask),
        torch.from_numpy(prior.astype(np.float32)), torch.from_numpy(target.astype(np.float32)),
    ]
    tensors.append(torch.from_numpy((extra if extra is not None else np.empty((len(features), 0), np.float32)).astype(np.float32)))
    return TensorDataset(*tensors)


def fit_profile_network(
    train: ProfileArrays,
    early: ProfileArrays,
    prior_train: np.ndarray,
    prior_early: np.ndarray,
    settings: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    seed: int,
    extra_train: np.ndarray | None = None,
    extra_early: np.ndarray | None = None,
    max_epochs_override: int | None = None,
    resume: bool = False,
) -> tuple[ProfileGaussianMLP, dict[str, Any],pd.DataFrame]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    feature_dim = train.features.shape[-1]
    # Inputs remain defined at missing/below-ground levels because ERA5 is complete.
    # Fit input scaling on every profile row so binary mask/support features are
    # not transformed from 0/1 into +/-100000 when the valid subset is constant.
    all_features = train.features.reshape(-1, feature_dim)
    feature_mean = all_features.mean(axis=0, keepdims=True).reshape(1, 1, -1)
    feature_std = np.maximum(all_features.std(axis=0, keepdims=True).reshape(1, 1, -1), 1e-5)
    train_data = _network_tensors(train, feature_mean, feature_std, prior_train, extra_train)
    early_data = _network_tensors(early, feature_mean, feature_std, prior_early, extra_early)
    batch_size = int(settings.get("batch_size", 256))
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=int(settings.get("num_workers", 0)), pin_memory=device.type == "cuda")
    early_loader = DataLoader(early_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = ProfileGaussianMLP(
        feature_dim, levels=train.features.shape[1], width=int(settings.get("hidden_width", 128)),
        layers=int(settings.get("hidden_layers", 2)), dropout=float(settings.get("dropout", 0.1)),
        extra_dim=0 if extra_train is None else extra_train.shape[1],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings.get("learning_rate", 3e-4)), weight_decay=float(settings.get("weight_decay", 1e-4)))
    amp_enabled = bool(settings.get("mixed_precision", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    epochs = int(max_epochs_override or settings.get("max_epochs", 80))
    patience = int(settings.get("patience", 12))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "model.pt"
    latest_checkpoint = output_dir / "latest.pt"
    best, stale = np.inf, 0
    history: list[dict[str, float]] = []
    start_epoch = 1
    if resume and latest_checkpoint.is_file():
        latest = torch.load(latest_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(latest["model"])
        optimizer.load_state_dict(latest["optimizer"])
        if amp_enabled and latest.get("scaler"):
            scaler.load_state_dict(latest["scaler"])
        start_epoch = int(latest["epoch"]) + 1
        best, stale = float(latest["best"]), int(latest["stale"])
        history = list(latest.get("history", []))
        print(f"Resume neural training at epoch {start_epoch}; previous best early NLL={best:.5f}")
    for epoch in range(start_epoch, epochs + 1):
        started = time.perf_counter()
        model.train()
        train_losses = []
        progress = tqdm(loader, desc=f"Epoch {epoch}/{epochs} train", leave=False)
        for features, mask, prior, target, extra in progress:
            features, mask, prior, target, extra = [value.to(device, non_blocking=True) for value in (features, mask, prior, target, extra)]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                mean, log_std = model(features, mask, prior, extra if extra.shape[1] else None)
                loss = masked_gaussian_nll(mean, log_std, target, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite masked Gaussian NLL; "
                    f"features_finite={bool(torch.isfinite(features).all())}, "
                    f"prior_finite={bool(torch.isfinite(prior).all())}, valid_targets={int(mask.sum())}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings.get("gradient_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))
            progress.set_postfix(loss=f"{train_losses[-1]:.4f}", gpu=device.type)
        model.eval()
        early_losses = []
        with torch.no_grad():
            for features, mask, prior, target, extra in tqdm(early_loader, desc=f"Epoch {epoch}/{epochs} validation", leave=False):
                features, mask, prior, target, extra = [value.to(device, non_blocking=True) for value in (features, mask, prior, target, extra)]
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    mean, log_std = model(features, mask, prior, extra if extra.shape[1] else None)
                    early_losses.append(float(masked_gaussian_nll(mean, log_std, target, mask).cpu()))
        train_loss, early_loss = float(np.mean(train_losses)), float(np.mean(early_losses))
        elapsed = time.perf_counter() - started
        history.append({"epoch": epoch, "train_nll": train_loss, "early_nll": early_loss, "epoch_seconds": elapsed})
        print(f"Epoch {epoch}/{epochs}: train_nll={train_loss:.5f}, early_nll={early_loss:.5f}, seconds={elapsed:.1f}")
        if early_loss < best - 1e-6:
            best, stale = early_loss, 0
            torch.save({"model": model.state_dict(), "feature_mean": feature_mean, "feature_std": feature_std, "settings": settings, "seed": seed}, checkpoint)
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping after epoch {epoch}; best early NLL={best:.5f}")
                break
        torch.save(
            {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "best": best, "stale": stale, "history": history,
                "feature_mean": feature_mean, "feature_std": feature_std, "settings": settings, "seed": seed,
            },
            latest_checkpoint,
        )
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    metadata = {"feature_mean": payload["feature_mean"], "feature_std": payload["feature_std"], "checkpoint": str(checkpoint), "best_early_nll": best}
    return model, metadata, pd.DataFrame(history)


def predict_profile_network(
    model: ProfileGaussianMLP,
    arrays: ProfileArrays,
    prior: np.ndarray,
    metadata: dict[str, Any],
    settings: dict[str, Any],
    device: torch.device,
    extra: np.ndarray | None = None,
) -> ProfilePrediction:
    dataset = _network_tensors(arrays, metadata["feature_mean"], metadata["feature_std"], prior, extra)
    loader = DataLoader(dataset, batch_size=int(settings.get("batch_size", 256)), shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    means, stds = [], []
    model.eval()
    with torch.no_grad():
        for features, mask, batch_prior, _, batch_extra in tqdm(loader, desc="Neural inference", leave=False):
            features, mask, batch_prior, batch_extra = [value.to(device, non_blocking=True) for value in (features, mask, batch_prior, batch_extra)]
            mean, log_std = model(features, mask, batch_prior, batch_extra if batch_extra.shape[1] else None)
            means.append(mean.cpu().numpy())
            stds.append(np.exp(log_std.cpu().numpy()))
    mean = np.concatenate(means)
    std = np.concatenate(stds)
    covariance = np.zeros((len(mean), mean.shape[1], mean.shape[1]), dtype=np.float32)
    diagonal = np.square(std[..., 0]) + np.square(std[..., 1])
    index = np.arange(mean.shape[1])
    covariance[:, index, index] = diagonal
    prediction = ProfilePrediction(
        mean[..., 0], mean[..., 1], mean[..., 0] + mean[..., 1],
        std[..., 0], std[..., 1], np.sqrt(diagonal), covariance, arrays.mask.copy(),
    )
    prediction.validate()
    return prediction


def normal_crps(error: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = np.maximum(np.asarray(std, dtype=float), 1e-8)
    z = np.asarray(error, dtype=float) / std
    return std * (z * (2 * ndtr(z) - 1) + 2 * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi))


def central_interval_score(error: np.ndarray, std: np.ndarray, coverage: float = 0.90) -> np.ndarray:
    alpha = 1.0 - coverage
    z = float(torch.distributions.Normal(0, 1).icdf(torch.tensor(1 - alpha / 2)).item())
    lower, upper = -z * std, z * std
    return (upper - lower) + 2 / alpha * (lower - error) * (error < lower) + 2 / alpha * (error - upper) * (error > upper)


def prediction_to_rows(arrays: ProfileArrays, prediction: ProfilePrediction, model_name: str, seed: int) -> pd.DataFrame:
    prediction.validate()
    preferred = [
        "station_id", "time", "latitude", "longitude", "station_elevation_m", "pressure_hpa",
        "igra_height_m", "era5_height_m", "era5_n_dry", "era5_n_wet", "era5_n",
        "era5_temperature_k", "era5_specific_humidity", "below_ground_level",
        "residual_dry", "residual_wet", "residual_total",
    ]
    frame = arrays.row_frame[[column for column in preferred if column in arrays.row_frame]].copy()
    frame["model"] = model_name
    frame["seed"] = seed
    frame["evaluation_mask"] = arrays.mask.reshape(-1)
    frame["prediction_dry"] = prediction.mean_dry.reshape(-1)
    frame["prediction_wet"] = prediction.mean_wet.reshape(-1)
    frame["prediction_total"] = prediction.mean_total.reshape(-1)
    if prediction.std_dry is not None and prediction.std_wet is not None:
        frame["std_dry"] = prediction.std_dry.reshape(-1)
        frame["std_wet"] = prediction.std_wet.reshape(-1)
        if prediction.std_total is not None:
            frame["std_total"] = prediction.std_total.reshape(-1)
        else:
            frame["std_total"] = np.sqrt(np.square(frame["std_dry"]) + np.square(frame["std_wet"]))
    else:
        frame[["std_dry", "std_wet", "std_total"]] = np.nan
    return frame


def metric_tables(
    predictions: pd.DataFrame,
    confidence_levels: list[float],
    bootstrap_replicates: int,
    seed: int,
    energy_score_draws: int = 64,
    arrays: ProfileArrays | None = None,
    prediction: ProfilePrediction | None = None,
    components: tuple[str, ...] = COMPONENTS,
) -> dict[str, pd.DataFrame]:
    unknown = set(components) - set(COMPONENTS)
    if unknown or "total" not in components:
        raise ValueError(f"Metric components must include total and use known names; unknown={sorted(unknown)}")
    valid = predictions.loc[predictions["evaluation_mask"]].copy()
    point_rows = []
    layer_rows = []
    for component in components:
        error = valid[f"prediction_{component}"] - valid[f"residual_{component}"]
        finite_error = error.loc[np.isfinite(error)].to_numpy(float)
        point_rows.append({
            "component": component, "n": len(finite_error),
            "rmse": float(np.sqrt(np.mean(finite_error**2))) if len(finite_error) else np.nan,
            "mae": float(np.mean(np.abs(finite_error))) if len(finite_error) else np.nan,
            "bias": float(np.mean(finite_error)) if len(finite_error) else np.nan,
        })
        for level, group in valid.assign(_error=error).groupby("pressure_hpa"):
            value = group["_error"].to_numpy(float)
            value = value[np.isfinite(value)]
            layer_rows.append({
                "pressure_hpa": int(level), "component": component, "n": len(value),
                "rmse": float(np.sqrt(np.mean(value**2))) if len(value) else np.nan,
                "mae": float(np.mean(np.abs(value))) if len(value) else np.nan,
                "bias": float(np.mean(value)) if len(value) else np.nan,
            })
    probabilistic_rows, coverage_rows = [], []
    if valid["std_total"].notna().any():
        for component in components:
            std = valid[f"std_{component}"].to_numpy(float)
            error = (valid[f"prediction_{component}"] - valid[f"residual_{component}"]).to_numpy(float)
            ok = np.isfinite(std) & (std > 0) & np.isfinite(error)
            nll = 0.5 * np.square(error[ok] / std[ok]) + np.log(std[ok]) + 0.5 * np.log(2 * np.pi)
            probabilistic_rows.append({
                "component": component, "n": int(ok.sum()), "gaussian_nll": float(nll.mean()),
                "crps": float(normal_crps(error[ok], std[ok]).mean()),
                "central_interval_score_90": float(central_interval_score(error[ok], std[ok], 0.90).mean()),
            })
            for coverage in confidence_levels:
                z = float(torch.distributions.Normal(0, 1).icdf(torch.tensor((1 + coverage) / 2)).item())
                coverage_rows.append({"component": component, "nominal_coverage": coverage, "empirical_coverage": float((np.abs(error[ok]) <= z * std[ok]).mean()), "n": int(ok.sum())})
        if arrays is None or prediction is None:
            rng_energy = np.random.default_rng(seed + 1000)
            profile_scores = []
            groups = valid.groupby(["station_id", "time"], observed=True, sort=False)
            for _, group in tqdm(groups, total=groups.ngroups, desc="Profile energy score", leave=False):
                mean = group["prediction_total"].to_numpy(float)
                std = group["std_total"].to_numpy(float)
                truth = group["residual_total"].to_numpy(float)
                if not (np.isfinite(std).all() and (std > 0).all()):
                    continue
                first = mean + std * rng_energy.standard_normal((energy_score_draws, len(mean)))
                second = mean + std * rng_energy.standard_normal((energy_score_draws, len(mean)))
                profile_scores.append(float(np.linalg.norm(first - truth, axis=1).mean() - 0.5 * np.linalg.norm(first - second, axis=1).mean()))
            for row in probabilistic_rows:
                row["profile_energy_score"] = float(np.mean(profile_scores)) if row["component"] == "total" and profile_scores else np.nan
    station = valid.assign(
        squared_model_error=np.square(valid["prediction_total"] - valid["residual_total"]),
        squared_era5_error=np.square(valid["residual_total"]),
    ).groupby("station_id").agg(model_mse=("squared_model_error", "mean"), era5_mse=("squared_era5_error", "mean"))
    station["rmse_difference_vs_era5"] = np.sqrt(station["model_mse"]) - np.sqrt(station["era5_mse"])
    rng = np.random.default_rng(seed)
    values = station["rmse_difference_vs_era5"].to_numpy(float)
    boot = np.empty(bootstrap_replicates)
    for index in tqdm(range(bootstrap_replicates), desc="Station bootstrap", leave=False):
        boot[index] = rng.choice(values, size=len(values), replace=True).mean()
    bootstrap = pd.DataFrame([{
        "comparison": "model_minus_era5_station_mean_rmse", "stations": len(values),
        "mean_difference": float(values.mean()), "ci_lower": float(np.quantile(boot, 0.025)), "ci_upper": float(np.quantile(boot, 0.975)),
        "replicates": bootstrap_replicates, "seed": seed,
    }])
    result = {
        "point_metrics": pd.DataFrame(point_rows), "layer_component_metrics": pd.DataFrame(layer_rows),
        "probabilistic_metrics": pd.DataFrame(probabilistic_rows), "coverage_curve": pd.DataFrame(coverage_rows),
        "station_bootstrap": bootstrap,
    }
    if arrays is not None and prediction is not None:
        from .revision2_confirmatory import bootstrap_inference_tables, joint_probability_metrics

        joint, joint_coverage = joint_probability_metrics(
            arrays, prediction, confidence_levels, energy_score_draws, seed
        )
        result["joint_probability_metrics"] = joint
        result["joint_coverage_curve"] = joint_coverage
        result.update(bootstrap_inference_tables(predictions, bootstrap_replicates, seed))
        if not joint.empty and not result["probabilistic_metrics"].empty:
            total = result["probabilistic_metrics"]["component"].eq("total")
            result["probabilistic_metrics"].loc[total, "profile_energy_score"] = float(joint.iloc[0]["profile_energy_score"])
            result["probabilistic_metrics"].loc[total, "multivariate_gaussian_nll"] = float(joint.iloc[0]["multivariate_gaussian_nll"])
    return result
