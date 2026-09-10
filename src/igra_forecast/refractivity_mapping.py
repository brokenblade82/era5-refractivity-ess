from __future__ import annotations

from dataclasses import dataclass
import glob
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


G0 = 9.80665
EPSILON = 0.622
DEFAULT_LAYERS = [925, 850, 700]
ERA5_GRID_DIMS = ("time", "level_hpa", "lat", "lon")


@dataclass
class MappingPrediction:
    mean: np.ndarray
    std: np.ndarray | None = None
    gate: np.ndarray | None = None
    nearest_context_distance_km: np.ndarray | None = None
    context_station_count: np.ndarray | None = None


@dataclass
class ResidualMappingModel:
    model: Any
    feature_columns: list[str]
    layers: list[int]
    target_column: str = "residual_n"
    residual_bounds: dict[int, tuple[float, float]] | None = None

    def predict_residual(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [col for col in self.feature_columns if col not in frame.columns]
        if missing:
            raise ValueError(f"Missing residual-model features: {missing}")
        features = frame[self.feature_columns].to_numpy(dtype=np.float32)
        residuals = np.asarray(self.model.predict(features), dtype=np.float32)
        if self.residual_bounds:
            residuals = clip_residuals_by_layer(residuals, frame, self.residual_bounds)
        return residuals

    def predict(
        self,
        target_frame: pd.DataFrame,
        context_frame: pd.DataFrame | None = None,
    ) -> MappingPrediction:
        del context_frame
        return MappingPrediction(mean=self.predict_residual(target_frame))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_type": "shared_hgb",
                "model": self.model,
                "feature_columns": self.feature_columns,
                "layers": self.layers,
                "target_column": self.target_column,
                "residual_bounds": self.residual_bounds,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ResidualMappingModel":
        payload = joblib.load(path)
        if payload.get("model_type") == "per_layer_hgb":
            return PerLayerResidualMappingModel(
                models={int(layer): model for layer, model in payload["models"].items()},
                feature_columns=list(payload["feature_columns"]),
                layers=[int(v) for v in payload["layers"]],
                target_column=payload.get("target_column", "residual_n"),
                residual_bounds=parse_residual_bounds(payload.get("residual_bounds")),
            )
        return cls(
            model=payload["model"],
            feature_columns=list(payload["feature_columns"]),
            layers=[int(v) for v in payload["layers"]],
            target_column=payload.get("target_column", "residual_n"),
            residual_bounds=parse_residual_bounds(payload.get("residual_bounds")),
        )


@dataclass
class PerLayerResidualMappingModel:
    models: dict[int, Any]
    feature_columns: list[str]
    layers: list[int]
    target_column: str = "residual_n"
    residual_bounds: dict[int, tuple[float, float]] | None = None

    def predict_residual(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [col for col in self.feature_columns if col not in frame.columns]
        if missing:
            raise ValueError(f"Missing residual-model features: {missing}")
        if "layer_hpa" not in frame:
            raise ValueError("Per-layer residual model requires a layer_hpa column.")
        residuals = np.full(len(frame), np.nan, dtype=np.float32)
        layers = frame["layer_hpa"].to_numpy(dtype=float)
        for layer, model in self.models.items():
            mask = np.isclose(layers, float(layer))
            if not np.any(mask):
                continue
            features = frame.loc[mask, self.feature_columns].to_numpy(dtype=np.float32)
            residuals[mask] = np.asarray(model.predict(features), dtype=np.float32)
        if np.isnan(residuals).any():
            missing_layers = sorted(set(frame.loc[np.isnan(residuals), "layer_hpa"].astype(int).tolist()))
            raise ValueError(f"Per-layer model has no estimator for layer(s): {missing_layers}")
        if self.residual_bounds:
            residuals = clip_residuals_by_layer(residuals, frame, self.residual_bounds)
        return residuals

    def predict(
        self,
        target_frame: pd.DataFrame,
        context_frame: pd.DataFrame | None = None,
    ) -> MappingPrediction:
        del context_frame
        return MappingPrediction(mean=self.predict_residual(target_frame))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_type": "per_layer_hgb",
                "models": self.models,
                "feature_columns": self.feature_columns,
                "layers": self.layers,
                "target_column": self.target_column,
                "residual_bounds": self.residual_bounds,
            },
            path,
        )


def open_era5_dataset(paths: str | Path | Iterable[str | Path]) -> xr.Dataset:
    if isinstance(paths, (str, Path)):
        text = str(paths)
        if any(ch in text for ch in "*?[]"):
            path_list = [Path(p) for p in sorted(glob.glob(text))]
        else:
            path = Path(paths)
            if path.is_dir() and path.suffix.lower() != ".zarr":
                path_list = sorted(path.glob("*.nc"))
            else:
                path_list = [path]
    else:
        path_list = [Path(p) for p in paths]
    if not path_list:
        raise FileNotFoundError(f"No ERA5 files matched: {paths}")
    missing = [str(p) for p in path_list if not p.exists()]
    if missing:
        raise FileNotFoundError(f"ERA5 file(s) not found: {missing}")
    if len(path_list) == 1:
        if is_zarr_path(path_list[0]):
            ds = xr.open_zarr(path_list[0])
        else:
            ds = xr.open_dataset(path_list[0])
    elif all(is_zarr_path(path) for path in path_list):
        ds = xr.combine_by_coords([xr.open_zarr(path) for path in path_list])
    else:
        ds = open_netcdf_collection(path_list)
    return standardize_era5_dataset(ds)


def open_netcdf_collection(path_list: list[Path]) -> xr.Dataset:
    """Open multiple NetCDF files without requiring dask in lightweight environments."""
    try:
        return xr.open_mfdataset([str(p) for p in path_list], combine="by_coords")
    except ImportError as exc:
        message = str(exc).lower()
        if "dask" not in message and "chunk manager" not in message:
            raise

    datasets = [xr.open_dataset(path) for path in path_list]
    try:
        combined = xr.concat(
            datasets,
            dim="time",
            data_vars="all",
            coords="minimal",
            compat="override",
            combine_attrs="override",
        )
        if "time" in combined:
            combined = combined.sortby("time")
        return combined
    except Exception:
        for dataset in datasets:
            dataset.close()
        raise


def standardize_era5_dataset(ds: xr.Dataset) -> xr.Dataset:
    rename: dict[str, str] = {}
    for candidates, canonical in [
        (("valid_time", "time"), "time"),
        (("pressure_level", "level", "isobaricInhPa"), "level_hpa"),
        (("latitude", "lat"), "lat"),
        (("longitude", "lon"), "lon"),
    ]:
        found = _find_name(ds, candidates)
        if found is not None and found != canonical:
            rename[found] = canonical
    if rename:
        ds = ds.rename(rename)
    required_coords = {"time", "level_hpa", "lat", "lon"}
    missing_coords = sorted(required_coords.difference(set(ds.coords) | set(ds.dims)))
    if missing_coords:
        raise ValueError(f"ERA5 dataset is missing coordinates/dimensions: {missing_coords}")
    if np.nanmax(np.asarray(ds["lon"].to_numpy(), dtype=float)) > 180.0:
        ds = ds.assign_coords(lon=(((ds["lon"] + 180.0) % 360.0) - 180.0)).sortby("lon")
    if "lat" in ds.coords:
        ds = ds.sortby("lat")
    return add_era5_refractivity(ds)


def add_era5_refractivity(ds: xr.Dataset) -> xr.Dataset:
    temp_name = _find_name(ds, ("t", "temperature"))
    q_name = _find_name(ds, ("q", "specific_humidity"))
    z_name = _find_name(ds, ("z", "geopotential"))
    out = ds
    if "n_era5" not in out:
        if temp_name is None or q_name is None:
            raise ValueError("ERA5 dataset must contain n_era5 or temperature (t) and specific humidity (q).")
        pressure_hpa = ds["level_hpa"]
        temperature_k = ds[temp_name]
        specific_humidity = ds[q_name].clip(min=0.0)
        vapor_pressure_hpa = vapor_pressure_from_specific_humidity(specific_humidity, pressure_hpa)
        n_era5 = radio_refractivity(pressure_hpa, temperature_k, vapor_pressure_hpa)
        out = out.assign(n_era5=n_era5.astype("float32"))
    if "height_m" not in out and z_name is not None:
        out = out.assign(height_m=(out[z_name] / G0).astype("float32"))
    return ensure_era5_dimension_order(out)


def vapor_pressure_from_specific_humidity(q: xr.DataArray, pressure_hpa: xr.DataArray) -> xr.DataArray:
    return (q * pressure_hpa) / (EPSILON + (1.0 - EPSILON) * q)


def radio_refractivity(pressure_hpa: xr.DataArray, temperature_k: xr.DataArray, vapor_pressure_hpa: xr.DataArray) -> xr.DataArray:
    return 77.6 * pressure_hpa / temperature_k + 3.73e5 * vapor_pressure_hpa / (temperature_k**2)


def load_igra_standard_layers(
    root: str | Path,
    layers: Iterable[int] = DEFAULT_LAYERS,
    allowed_hours: Iterable[int] = (0, 12),
    min_complete_times: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    files = sorted(root.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No station CSV files found in {root}")
    layers_set = {int(v) for v in layers}
    allowed = {int(v) for v in allowed_hours}
    rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    required = {
        "Lon",
        "Lat",
        "Time",
        "Pressure_hPa",
        "Height_m",
        "Temperature_C",
        "Specific_Humidity_gkg",
        "Layer_Type",
        "Refractivity_N",
    }
    for path in files:
        station_id = path.stem
        df = pd.read_csv(path)
        missing = required.difference(df.columns)
        if missing:
            audit_rows.append({"station_id": station_id, "file": path.name, "error": f"missing columns: {sorted(missing)}"})
            continue
        df["time"] = pd.to_datetime(df["Time"], errors="coerce")
        df = df[df["time"].notna()].copy()
        df = df[df["time"].dt.hour.isin(allowed)]
        df = df[df["Layer_Type"].eq("Standard")].copy()
        df["layer_hpa"] = pd.to_numeric(df["Pressure_hPa"], errors="coerce").round().astype("Int64")
        df = df[df["layer_hpa"].isin(layers_set)].copy()
        if df.empty:
            audit_rows.append({"station_id": station_id, "file": path.name, "rows": 0, "complete_times": 0})
            continue
        complete = (
            df.assign(ok=pd.to_numeric(df["Refractivity_N"], errors="coerce").notna())
            .pivot_table(index="time", columns="layer_hpa", values="ok", aggfunc="first")
            .reindex(columns=sorted(layers_set))
            .eq(True)
            .all(axis=1)
        )
        complete_times = complete[complete].index
        if len(complete_times) < min_complete_times:
            audit_rows.append(
                {
                    "station_id": station_id,
                    "file": path.name,
                    "rows": int(len(df)),
                    "complete_times": int(len(complete_times)),
                    "eligible": False,
                }
            )
            continue
        df = df[df["time"].isin(complete_times)].copy()
        out = pd.DataFrame(
            {
                "station_id": station_id,
                "time": df["time"],
                "lat": pd.to_numeric(df["Lat"], errors="coerce"),
                "lon": normalize_longitude(pd.to_numeric(df["Lon"], errors="coerce")),
                "layer_hpa": df["layer_hpa"].astype(int),
                "height_m": pd.to_numeric(df["Height_m"], errors="coerce"),
                "temperature_c": pd.to_numeric(df["Temperature_C"], errors="coerce"),
                "specific_humidity_gkg": pd.to_numeric(df["Specific_Humidity_gkg"], errors="coerce"),
                "n_igra": pd.to_numeric(df["Refractivity_N"], errors="coerce"),
            }
        )
        out = out.dropna(subset=["lat", "lon", "layer_hpa", "n_igra"])
        rows.append(out)
        audit_rows.append(
            {
                "station_id": station_id,
                "file": path.name,
                "rows": int(len(df)),
                "complete_times": int(len(complete_times)),
                "eligible": True,
                "lat": float(out["lat"].iloc[0]) if not out.empty else np.nan,
                "lon": float(out["lon"].iloc[0]) if not out.empty else np.nan,
            }
        )
    if not rows:
        raise ValueError("No eligible IGRA standard-layer records were found.")
    return pd.concat(rows, ignore_index=True), pd.DataFrame(audit_rows)


def build_station_split(station_ids: Iterable[str], split_cfg: dict[str, float], seed: int) -> dict[str, list[str]]:
    ids = np.array(sorted(set(str(v) for v in station_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    train_frac = float(split_cfg.get("train", 0.7))
    val_frac = float(split_cfg.get("val", 0.1))
    train_end = int(len(ids) * train_frac)
    val_end = train_end + int(len(ids) * val_frac)
    return {
        "train": sorted(ids[:train_end].tolist()),
        "val": sorted(ids[train_end:val_end].tolist()),
        "test": sorted(ids[val_end:].tolist()),
    }


def attach_split(frame: pd.DataFrame, station_split: dict[str, list[str]]) -> pd.DataFrame:
    lookup = {station_id: split for split, ids in station_split.items() for station_id in ids}
    out = frame.copy()
    out["split"] = out["station_id"].map(lookup)
    return out[out["split"].notna()].reset_index(drop=True)


def sample_era5_at_igra(era5: xr.Dataset, igra: pd.DataFrame) -> pd.DataFrame:
    points = igra.reset_index(drop=True).copy()
    sample_dim = "sample"
    sample_coords = {
        "time": xr.DataArray(pd.to_datetime(points["time"]).to_numpy(dtype="datetime64[ns]"), dims=sample_dim),
        "level_hpa": xr.DataArray(points["layer_hpa"].to_numpy(dtype=np.float32), dims=sample_dim),
        "lat": xr.DataArray(points["lat"].to_numpy(dtype=np.float32), dims=sample_dim),
        "lon": xr.DataArray(normalize_longitude(points["lon"]).to_numpy(dtype=np.float32), dims=sample_dim),
    }
    sampled = era5.interp(sample_coords)
    for src, dst in [
        ("n_era5", "n_era5"),
        ("height_m", "era5_height_m"),
        ("t", "era5_temperature_k"),
        ("temperature", "era5_temperature_k"),
        ("q", "era5_specific_humidity"),
        ("specific_humidity", "era5_specific_humidity"),
    ]:
        if src in sampled:
            points[dst] = sampled[src].to_numpy()
    if "n_era5" not in points:
        raise ValueError("ERA5 sampling did not produce n_era5.")
    points["residual_n"] = points["n_igra"] - points["n_era5"]
    return add_mapping_features(points).dropna(subset=["n_igra", "n_era5", "residual_n"])


def add_mapping_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    time = pd.to_datetime(out["time"])
    doy = time.dt.dayofyear.astype(float)
    hour = time.dt.hour.astype(float)
    out["lat_norm"] = out["lat"].astype(float) / 90.0
    lon_rad = np.deg2rad(out["lon"].astype(float))
    out["lon_sin"] = np.sin(lon_rad)
    out["lon_cos"] = np.cos(lon_rad)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 366.0)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 366.0)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["layer_norm"] = out["layer_hpa"].astype(float) / 1000.0
    if "era5_height_m" in out:
        out["era5_height_km"] = out["era5_height_m"].astype(float) / 1000.0
    if "height_m" in out:
        out["height_km"] = out["height_m"].astype(float) / 1000.0
    return out


def default_feature_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "n_era5",
        "era5_temperature_k",
        "era5_specific_humidity",
        "era5_height_km",
        "lat_norm",
        "lon_sin",
        "lon_cos",
        "doy_sin",
        "doy_cos",
        "hour_sin",
        "hour_cos",
        "layer_norm",
    ]
    return [col for col in candidates if col in frame.columns and frame[col].notna().any()]


def train_residual_model(
    samples: pd.DataFrame,
    feature_columns: list[str] | None = None,
    model_cfg: dict[str, Any] | None = None,
) -> ResidualMappingModel:
    model_cfg = model_cfg or {}
    feature_columns = feature_columns or default_feature_columns(samples)
    train = samples[samples["split"].eq("train")].copy()
    if train.empty:
        raise ValueError("No train samples are available for residual mapping.")
    train = train.dropna(subset=[*feature_columns, "residual_n"])
    layers = sorted(int(v) for v in samples["layer_hpa"].dropna().unique())
    residual_bounds = build_residual_bounds(train, model_cfg)
    strategy = str(model_cfg.get("strategy", "shared_hgb")).lower()
    if strategy in {"per_layer", "per_layer_hgb", "layerwise"}:
        models: dict[int, Any] = {}
        for layer in layers:
            layer_train = train[train["layer_hpa"].astype(int).eq(layer)]
            if layer_train.empty:
                raise ValueError(f"No train samples are available for layer {layer}.")
            model = build_hgb_regressor(model_cfg)
            model.fit(
                layer_train[feature_columns].to_numpy(dtype=np.float32),
                layer_train["residual_n"].to_numpy(dtype=np.float32),
            )
            models[layer] = model
        return PerLayerResidualMappingModel(
            models=models,
            feature_columns=feature_columns,
            layers=layers,
            residual_bounds=residual_bounds,
        )
    model = build_hgb_regressor(model_cfg)
    model.fit(train[feature_columns].to_numpy(dtype=np.float32), train["residual_n"].to_numpy(dtype=np.float32))
    return ResidualMappingModel(
        model=model,
        feature_columns=feature_columns,
        layers=layers,
        residual_bounds=residual_bounds,
    )


def build_hgb_regressor(model_cfg: dict[str, Any]) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=int(model_cfg.get("max_iter", 400)),
        learning_rate=float(model_cfg.get("learning_rate", 0.04)),
        max_leaf_nodes=int(model_cfg.get("max_leaf_nodes", 31)),
        min_samples_leaf=int(model_cfg.get("min_samples_leaf", 20)),
        max_bins=int(model_cfg.get("max_bins", 255)),
        l2_regularization=float(model_cfg.get("l2_regularization", 0.01)),
        early_stopping=model_cfg.get("early_stopping", "auto"),
        validation_fraction=float(model_cfg.get("validation_fraction", 0.1)),
        n_iter_no_change=int(model_cfg.get("n_iter_no_change", 10)),
        tol=float(model_cfg.get("tol", 1e-7)),
        random_state=int(model_cfg.get("random_state", 42)),
    )


def build_residual_bounds(train: pd.DataFrame, model_cfg: dict[str, Any]) -> dict[int, tuple[float, float]] | None:
    quantiles = model_cfg.get("residual_clip_quantiles")
    if not quantiles:
        return None
    if len(quantiles) != 2:
        raise ValueError("model.hgb.residual_clip_quantiles must contain two values, e.g. [0.005, 0.995].")
    lower_q, upper_q = float(quantiles[0]), float(quantiles[1])
    if not 0.0 <= lower_q < upper_q <= 1.0:
        raise ValueError("Residual clip quantiles must satisfy 0 <= lower < upper <= 1.")
    bounds: dict[int, tuple[float, float]] = {}
    for layer, group in train.groupby("layer_hpa", sort=True):
        lower = float(group["residual_n"].quantile(lower_q))
        upper = float(group["residual_n"].quantile(upper_q))
        bounds[int(layer)] = (lower, upper)
    return bounds


def parse_residual_bounds(payload: Any) -> dict[int, tuple[float, float]] | None:
    if not payload:
        return None
    return {int(layer): (float(values[0]), float(values[1])) for layer, values in payload.items()}


def clip_residuals_by_layer(
    residuals: np.ndarray,
    frame: pd.DataFrame,
    bounds: dict[int, tuple[float, float]],
) -> np.ndarray:
    if "layer_hpa" not in frame:
        return residuals
    clipped = residuals.copy()
    layers = frame["layer_hpa"].to_numpy(dtype=float)
    for layer, (lower, upper) in bounds.items():
        mask = np.isclose(layers, float(layer))
        if np.any(mask):
            clipped[mask] = np.clip(clipped[mask], lower, upper)
    return clipped.astype(np.float32, copy=False)


def evaluate_residual_mapping(samples: pd.DataFrame, residual_model: ResidualMappingModel) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame = samples.dropna(subset=[*residual_model.feature_columns, "n_igra", "n_era5"]).copy()
    frame["n_corrected"] = frame["n_era5"] + residual_model.predict_residual(frame)
    for split, group in frame.groupby("split", sort=True):
        rows.extend(_metric_rows(group, split, "all", "all"))
        for layer, layer_group in group.groupby("layer_hpa", sort=True):
            rows.extend(_metric_rows(layer_group, split, int(layer), "all"))
        for band, band_group in add_latitude_band(group).groupby("latitude_band", sort=True):
            rows.extend(_metric_rows(band_group, split, "all", band))
    return pd.DataFrame(rows)


def add_latitude_band(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    alat = out["lat"].abs()
    out["latitude_band"] = np.select(
        [alat < 23.5, alat < 35.0, alat < 60.0],
        ["Tropics", "Subtropics", "Mid-latitudes"],
        default="High latitudes",
    )
    return out


def apply_residual_model_to_grid(
    era5: xr.Dataset,
    residual_model: Any,
    times: Iterable[str] | None = None,
    layers: Iterable[int] | None = None,
    context_samples: pd.DataFrame | None = None,
    surface_pressure: xr.DataArray | None = None,
) -> xr.Dataset:
    ds = era5
    if times:
        ds = ds.sel(time=pd.to_datetime(list(times)))
    if layers:
        ds = ds.sel(level_hpa=[int(v) for v in layers])
    needed = ["n_era5"]
    missing = [name for name in needed if name not in ds]
    if missing:
        raise ValueError(f"ERA5 grid is missing variables: {missing}")
    frame = era5_grid_to_feature_frame(ds)
    if hasattr(residual_model, "predict"):
        prediction = residual_model.predict(frame, context_frame=context_samples)
    else:
        prediction = MappingPrediction(mean=residual_model.predict_residual(frame))
    frame["residual_n"] = prediction.mean
    grid_shape = (ds.sizes["time"], ds.sizes["level_hpa"], ds.sizes["lat"], ds.sizes["lon"])
    corrected_values = (
        frame["n_era5"].to_numpy(dtype=np.float32) + frame["residual_n"].to_numpy(dtype=np.float32)
    ).reshape(grid_shape)
    corrected = xr.DataArray(
        corrected_values,
        coords={
            "time": ds["time"],
            "level_hpa": ds["level_hpa"],
            "lat": ds["lat"],
            "lon": ds["lon"],
        },
        dims=("time", "level_hpa", "lat", "lon"),
        name="n_corrected",
    )
    additions: dict[str, xr.DataArray] = {"n_corrected": corrected}
    for name, values in [
        ("n_correction_std", prediction.std),
        ("n_correction_gate", prediction.gate),
        ("nearest_context_distance_km", prediction.nearest_context_distance_km),
        ("context_station_count", prediction.context_station_count),
    ]:
        if values is None:
            continue
        array = np.asarray(values, dtype=np.float32)
        if array.size != int(np.prod(grid_shape)):
            raise ValueError(f"{name} has {array.size} values; expected {int(np.prod(grid_shape))}.")
        additions[name] = xr.DataArray(
            array.reshape(grid_shape),
            coords={"time": ds["time"], "level_hpa": ds["level_hpa"], "lat": ds["lat"], "lon": ds["lon"]},
            dims=ERA5_GRID_DIMS,
            name=name,
        )
    out = ds.assign(**additions)
    if surface_pressure is not None:
        surface_pressure = _align_surface_pressure(surface_pressure, out)
        below_ground_mask = surface_pressure < (out["level_hpa"] * 100.0)
        below_ground_mask = below_ground_mask.transpose(*ERA5_GRID_DIMS).astype(bool)
        # NetCDF does not define a portable boolean variable type. Persist the
        # QC flag as uint8, while retaining a boolean mask for calculations.
        out = out.assign(
            surface_pressure_pa=surface_pressure.astype("float32"),
            below_ground_level=below_ground_mask.astype(np.uint8).rename("below_ground_level"),
        )
        # A standard pressure surface below the ERA5 land surface has no direct
        # atmospheric interpretation.  Preserve the background variables but
        # mask all derived correction diagnostics and gradients.
        for name in [
            "n_corrected", "n_correction_std", "n_correction_gate",
            "nearest_context_distance_km", "context_station_count",
        ]:
            if name in out:
                out[name] = out[name].where(~below_ground_mask)
    height = out["height_m"] if "height_m" in out else pressure_level_height_approx(out["level_hpa"])
    out = out.assign(m_corrected=(out["n_corrected"] + 0.157 * height).astype("float32"))
    out = add_vertical_gradients(out)
    return set_refractivity_variable_attrs(out)


def open_surface_pressure_dataset(path: str | Path) -> xr.DataArray:
    """Open ERA5 single-level surface pressure and standardize it to Pa.

    The source may use the CDS short name ``sp`` or a descriptive pressure
    variable.  The returned field must have time/lat/lon dimensions.
    """
    ds = xr.open_dataset(path)
    name = _find_name(ds, ["sp", "surface_pressure", "surface_air_pressure"])
    if name is None:
        ds.close()
        raise ValueError(f"Surface-pressure file has no supported variable: {path}")
    pressure = ds[name]
    rename = {}
    for source, target in [("latitude", "lat"), ("longitude", "lon"), ("valid_time", "time")]:
        if source in pressure.dims or source in pressure.coords:
            rename[source] = target
    if rename:
        pressure = pressure.rename(rename)
    required = {"time", "lat", "lon"}
    if not required.issubset(pressure.dims):
        ds.close()
        raise ValueError(f"Surface pressure must have dimensions {sorted(required)}; got {pressure.dims}")
    units = str(pressure.attrs.get("units", "Pa")).lower().replace(" ", "")
    if units in {"hpa", "mbar", "millibar"}:
        pressure = pressure * 100.0
    elif units not in {"pa", "pascal", "pascals", ""}:
        ds.close()
        raise ValueError(f"Unsupported surface-pressure units {units!r} in {path}")
    pressure = pressure.transpose("time", "lat", "lon").astype("float32")
    pressure.name = "surface_pressure_pa"
    pressure.attrs.update({"long_name": "ERA5 surface pressure", "units": "Pa"})
    return pressure


def _align_surface_pressure(surface_pressure: xr.DataArray, grid: xr.Dataset) -> xr.DataArray:
    source = surface_pressure
    if not np.array_equal(source["time"].values, grid["time"].values):
        source = source.sel(time=grid["time"])
    if not np.array_equal(source["lat"].values, grid["lat"].values) or not np.array_equal(source["lon"].values, grid["lon"].values):
        source = source.interp(lat=grid["lat"], lon=grid["lon"], method="linear")
    return source


def load_mapping_model(path: str | Path, device: str | None = None) -> Any:
    """Load a legacy HGB joblib file or an HGB-SRNP checkpoint directory."""
    path = Path(path)
    if path.is_dir():
        from igra_forecast.mapping_neural import NeuralContextResidualModel

        return NeuralContextResidualModel.load(path, device=device)
    return ResidualMappingModel.load(path)


def era5_grid_to_feature_frame(ds: xr.Dataset) -> pd.DataFrame:
    coords = xr.broadcast(ds["time"], ds["level_hpa"], ds["lat"], ds["lon"])
    time_da, level_da, lat_da, lon_da = coords
    data = {
        "time": time_da.values.reshape(-1),
        "layer_hpa": level_da.values.reshape(-1).astype(float),
        "lat": lat_da.values.reshape(-1).astype(float),
        "lon": lon_da.values.reshape(-1).astype(float),
        "n_era5": ds["n_era5"].values.reshape(-1).astype(float),
    }
    for src, dst in [
        ("height_m", "era5_height_m"),
        ("t", "era5_temperature_k"),
        ("temperature", "era5_temperature_k"),
        ("q", "era5_specific_humidity"),
        ("specific_humidity", "era5_specific_humidity"),
    ]:
        if src in ds:
            data[dst] = ds[src].values.reshape(-1).astype(float)
    frame = add_mapping_features(pd.DataFrame(data))
    return frame


def add_vertical_gradients(ds: xr.Dataset) -> xr.Dataset:
    if "height_m" in ds:
        height = ds["height_m"]
    else:
        height = pressure_level_height_approx(ds["level_hpa"])
    dheight_km = height.diff("level_hpa") / 1000.0
    dn = ds["n_corrected"].diff("level_hpa") / dheight_km
    dm = ds["m_corrected"].diff("level_hpa") / dheight_km
    mid_levels = [
        f"{int(a)}-{int(b)}"
        for a, b in zip(ds["level_hpa"].values[:-1].tolist(), ds["level_hpa"].values[1:].tolist())
    ]
    dn = dn.assign_coords(level_hpa_mid=("level_hpa", mid_levels)).swap_dims({"level_hpa": "level_hpa_mid"})
    dm = dm.assign_coords(level_hpa_mid=("level_hpa", mid_levels)).swap_dims({"level_hpa": "level_hpa_mid"})
    return ds.assign(dn_dz=dn.astype("float32"), dm_dz=dm.astype("float32"))


def ensure_era5_dimension_order(ds: xr.Dataset) -> xr.Dataset:
    out = ds
    for name in list(out.data_vars):
        da = out[name]
        if all(dim in da.dims for dim in ERA5_GRID_DIMS):
            out[name] = da.transpose(*ERA5_GRID_DIMS)
    return out


def pressure_level_height_approx(level_hpa: xr.DataArray) -> xr.DataArray:
    return 44330.0 * (1.0 - (level_hpa / 1013.25) ** 0.1903)


def write_grid_product(ds: xr.Dataset, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = set_refractivity_variable_attrs(ds)
    encoding = {
        name: {"zlib": True, "complevel": 4}
        for name in [
            "n_era5", "height_m", "m_era5", "n_corrected", "m_corrected", "dn_dz", "dm_dz",
            "n_correction_std", "n_correction_gate", "nearest_context_distance_km", "context_station_count",
            "surface_pressure_pa", "below_ground_level",
        ]
        if name in ds
    }
    ds.to_netcdf(path, encoding=encoding)


def set_refractivity_variable_attrs(ds: xr.Dataset) -> xr.Dataset:
    out = ds.copy()
    attrs = {
        "n_era5": {
            "long_name": "ERA5 background atmospheric radio refractivity",
            "units": "N-units",
        },
        "n_corrected": {
            "long_name": "IGRA-corrected atmospheric radio refractivity",
            "units": "N-units",
        },
        "n_correction_std": {
            "long_name": "Calibrated standard deviation of the refractivity correction",
            "units": "N-units",
        },
        "n_correction_gate": {
            "long_name": "Observation-conditioned spatial correction gate",
            "units": "1",
        },
        "nearest_context_distance_km": {
            "long_name": "Distance to nearest same-time IGRA context station",
            "units": "km",
        },
        "context_station_count": {
            "long_name": "Number of IGRA context stations used by the correction model",
            "units": "1",
        },
        "height_m": {
            "long_name": "Geopotential height",
            "units": "m",
        },
        "surface_pressure_pa": {
            "long_name": "ERA5 surface pressure used for below-ground pressure-level masking",
            "units": "Pa",
        },
        "below_ground_level": {
            "long_name": "Standard pressure level located below the ERA5 surface",
            "units": "1",
            "flag_values": np.asarray([0, 1], dtype=np.uint8),
            "flag_meanings": "atmospheric_level below_ground_level",
        },
        "m_era5": {
            "long_name": "ERA5 background modified refractivity",
            "units": "M-units",
        },
        "m_corrected": {
            "long_name": "IGRA-corrected modified refractivity",
            "units": "M-units",
        },
        "dn_dz": {
            "long_name": "Vertical gradient of corrected radio refractivity",
            "units": "N-units km-1",
        },
        "dm_dz": {
            "long_name": "Vertical gradient of corrected modified refractivity",
            "units": "M-units km-1",
        },
    }
    for name, values in attrs.items():
        if name in out:
            out[name].attrs.update(values)
    return out


def _metric_rows(group: pd.DataFrame, split: str, layer: int | str, group_name: str) -> list[dict[str, Any]]:
    y_true = group["n_igra"].to_numpy(dtype=float)
    baseline = group["n_era5"].to_numpy(dtype=float)
    corrected = group["n_corrected"].to_numpy(dtype=float)
    return [
        _one_metric(split, layer, group_name, "era5", y_true, baseline),
        _one_metric(split, layer, group_name, "corrected", y_true, corrected),
    ]


def _one_metric(split: str, layer: int | str, group_name: str, model: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    err = y_pred - y_true
    return {
        "split": split,
        "layer_hpa": layer,
        "group": group_name,
        "model": model,
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(err)),
        "n": int(len(y_true)),
    }


def normalize_longitude(values: Any) -> Any:
    return ((values + 180.0) % 360.0) - 180.0


def _find_name(ds: xr.Dataset, candidates: Iterable[str]) -> str | None:
    names = set(ds.variables) | set(ds.coords) | set(ds.dims)
    for name in candidates:
        if name in names:
            return name
    lower = {name.lower(): name for name in names}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def is_zarr_path(path: Path) -> bool:
    return path.suffix.lower() == ".zarr" or (path.is_dir() and (path / ".zgroup").exists())
