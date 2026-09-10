from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from _bootstrap import PROJECT_ROOT  # noqa: F401
from igra_forecast.revision2_data import ensure_data_directories, load_data_config, refractivity_components, write_json


ALIASES = {
    "pressure": ["p", "pressure", "pres"],
    "temperature": ["ta", "temperature", "air_temperature"],
    "specific_humidity": ["q", "specific_humidity"],
    "height": ["height", "geopotential_height", "alt", "altitude"],
    "latitude": ["launch_lat", "lat", "latitude"],
    "longitude": ["launch_lon", "lon", "longitude"],
    "platform": ["platform", "platform_id", "site"],
    "time": ["launch_time", "time", "datetime", "date"],
}


def choose(dataset: xr.Dataset, logical: str, required: bool = True) -> str | None:
    name = next((candidate for candidate in ALIASES[logical] if candidate in dataset.variables), None)
    if required and name is None:
        raise KeyError(f"Cannot locate {logical}; tried {ALIASES[logical]}. Check the schema manifest.")
    return name


def unit_convert(values: np.ndarray, unit: str, kind: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    unit = unit.lower().replace(" ", "")
    if kind == "pressure" and ("pa" in unit and "hpa" not in unit or np.nanmedian(result) > 2000):
        result = result / 100.0
    if kind == "temperature" and (unit in {"c", "degc", "degree_celsius"} or np.nanmedian(result) < 100):
        result = result + 273.15
    if kind == "specific_humidity" and ("g/kg" in unit or np.nanpercentile(result, 95) > 0.2):
        result = result / 1000.0
    return result


def interp_log_pressure(pressure_hpa: np.ndarray, values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    valid = np.isfinite(pressure_hpa) & np.isfinite(values) & (pressure_hpa > 0)
    if valid.sum() < 2:
        return np.full(targets.shape, np.nan)
    x = np.log(pressure_hpa[valid])
    y = values[valid]
    order = np.argsort(x)
    x, unique = np.unique(x[order], return_index=True)
    y = y[order][unique]
    return np.interp(np.log(targets), x, y, left=np.nan, right=np.nan)


def profile_scalar(dataset: xr.Dataset, name: str | None, profile_dim: str, index: int) -> object:
    if name is None:
        return np.nan
    value = dataset[name]
    if profile_dim in value.dims:
        value = value.isel({profile_dim: index})
    array = np.asarray(value.values).reshape(-1)
    finite = array[pd.notna(array)]
    return finite[0] if len(finite) else np.nan


def profile_vector(dataset: xr.Dataset, name: str, profile_dim: str, index: int) -> np.ndarray:
    """Read either a per-profile vector or a vertical coordinate shared by all profiles."""
    value = dataset[name]
    if profile_dim in value.dims:
        value = value.isel({profile_dim: index})
    result = np.asarray(value.values).reshape(-1)
    if result.ndim != 1:
        raise ValueError(f"Expected a one-dimensional profile vector for {name}; found {value.dims}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert RAPSODI Level-2 profiles to the revision-2 six-level data contract.")
    parser.add_argument("--config", default="configs/revision2/data_pipeline.yaml")
    parser.add_argument("--input")
    parser.add_argument("--output", default="data/revision2/processed/rapsodi_profiles.parquet")
    parser.add_argument("--max-profiles", type=int, help="Optional local parsing smoke-test limit.")
    args = parser.parse_args()
    config = load_data_config(args.config)
    paths = ensure_data_directories(config)
    source = Path(args.input or config["rapsodi"]["local_store"])
    dataset = xr.open_zarr(source, consolidated=None)
    names = {logical: choose(dataset, logical, required=logical not in {"platform", "time"}) for logical in ALIASES}
    pressure = dataset[names["pressure"]]
    if len(pressure.dims) < 2:
        raise ValueError(f"Pressure must have profile and vertical dimensions; found {pressure.dims}")
    vertical_dim = pressure.dims[-1]
    profile_dim = pressure.dims[0]
    targets = np.asarray(config["study"]["pressure_levels_hpa"], dtype=float)
    rows = []
    profile_count = dataset.sizes[profile_dim]
    if args.max_profiles is not None:
        if args.max_profiles < 1:
            raise ValueError("--max-profiles must be positive")
        profile_count = min(profile_count, int(args.max_profiles))
    for index in tqdm(range(profile_count), desc="Prepare RAPSODI profiles"):
        p_raw = profile_vector(dataset, names["pressure"], profile_dim, index)
        t_raw = profile_vector(dataset, names["temperature"], profile_dim, index)
        q_raw = profile_vector(dataset, names["specific_humidity"], profile_dim, index)
        z_raw = profile_vector(dataset, names["height"], profile_dim, index)
        lengths = {"pressure": len(p_raw), "temperature": len(t_raw), "specific_humidity": len(q_raw), "height": len(z_raw)}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"RAPSODI profile vectors have inconsistent lengths at index {index}: {lengths}")
        p = unit_convert(p_raw, dataset[names["pressure"]].attrs.get("units", ""), "pressure")
        t = unit_convert(t_raw, dataset[names["temperature"]].attrs.get("units", ""), "temperature")
        q = unit_convert(q_raw, dataset[names["specific_humidity"]].attrs.get("units", ""), "specific_humidity")
        t_target = interp_log_pressure(p, t, targets)
        q_target = interp_log_pressure(p, q, targets)
        z_target = interp_log_pressure(p, np.asarray(z_raw, float), targets)
        mask = np.isfinite(t_target) & np.isfinite(q_target) & np.isfinite(z_target)
        if mask.sum() < config["study"]["min_valid_levels_per_profile"]:
            continue
        dry, wet, total, _ = refractivity_components(targets, t_target, q_target)
        time_name = names["time"] or (profile_dim if profile_dim in dataset.coords else None)
        time_value = profile_scalar(dataset, time_name, profile_dim, index)
        latitude = profile_scalar(dataset, names["latitude"], profile_dim, index)
        longitude = profile_scalar(dataset, names["longitude"], profile_dim, index)
        platform = profile_scalar(dataset, names["platform"], profile_dim, index)
        for level_index, level in enumerate(targets.astype(int)):
            rows.append(
                {"profile_id": f"RAPSODI_{index:04d}", "time": pd.to_datetime(time_value), "latitude": latitude,
                 "longitude": longitude, "platform": platform, "pressure_hpa": level,
                 "level_mask": int(mask[level_index]), "temperature_k": t_target[level_index],
                 "specific_humidity": q_target[level_index], "height_m": z_target[level_index],
                 "observed_n_dry": dry[level_index] if mask[level_index] else np.nan,
                 "observed_n_wet": wet[level_index] if mask[level_index] else np.nan,
                 "observed_n": total[level_index] if mask[level_index] else np.nan, "source": "RAPSODI_Level2"}
            )
    result = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False, compression="zstd")
    write_json(paths["manifests"] / "rapsodi_prepare_manifest.json", {
        "input": str(source.resolve()), "output": str(output.resolve()), "variable_mapping": names,
        "profiles": int(result["profile_id"].nunique()) if len(result) else 0, "rows": len(result),
        "source_profiles": int(dataset.sizes[profile_dim]), "profiles_requested": int(profile_count),
        "note": "Platform independence must be audited before selecting INMG as the non-GTS subset.",
    })
    dataset.close()
    print(f"RAPSODI profiles: {output.resolve()}")


if __name__ == "__main__":
    main()
