from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
from scipy.spatial import cKDTree

from _bootstrap import PROJECT_ROOT  # noqa: F401
from igra_forecast.revision2_data import (
    ensure_data_directories,
    load_data_config,
    macro_region,
    require_collocation_manifest,
    sha256_file,
    write_json,
)


def metrics(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column].dropna().to_numpy(float)
    return pd.Series(
        {"n": len(values), "bias": np.mean(values), "mae": np.mean(np.abs(values)),
         "rmse": np.sqrt(np.mean(np.square(values))), "std": np.std(values, ddof=1) if len(values) > 1 else np.nan,
         "variance": np.var(values, ddof=1) if len(values) > 1 else np.nan}
    )


def variance_identity(frame: pd.DataFrame) -> dict[str, float]:
    """Quantify Var(total)=Var(dry)+Var(wet)+2Cov(dry,wet)."""
    values = frame[["residual_dry", "residual_wet", "residual_total"]].dropna().to_numpy(float)
    if len(values) < 2:
        return {key: np.nan for key in [
            "variance_dry", "variance_wet", "covariance_dry_wet", "variance_total",
            "variance_total_from_components", "identity_error",
        ]}
    dry, wet, total = values.T
    covariance = float(np.cov(dry, wet, ddof=1)[0, 1])
    variance_dry = float(np.var(dry, ddof=1))
    variance_wet = float(np.var(wet, ddof=1))
    variance_total = float(np.var(total, ddof=1))
    reconstructed = variance_dry + variance_wet + 2.0 * covariance
    return {
        "variance_dry": variance_dry,
        "variance_wet": variance_wet,
        "covariance_dry_wet": covariance,
        "variance_total": variance_total,
        "variance_total_from_components": reconstructed,
        "identity_error": variance_total - reconstructed,
    }


def modified_refractivity_residual(frame: pd.DataFrame) -> pd.DataFrame:
    """Return total-N and geopotential-height contributions to M residual."""
    result = pd.DataFrame(index=frame.index)
    result["residual_m_n_contribution"] = frame["residual_total"].astype(float)
    result["residual_m_height_contribution"] = 0.157 * (
        frame["igra_height_m"].astype(float) - frame["era5_height_m"].astype(float)
    )
    result["residual_m"] = result["residual_m_n_contribution"] + result["residual_m_height_contribution"]
    return result


def station_support(stations: pd.DataFrame) -> pd.DataFrame:
    lat = np.deg2rad(stations["latitude"].to_numpy(float))
    lon = np.deg2rad(stations["longitude"].to_numpy(float))
    xyz = np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    distance, _ = cKDTree(xyz).query(xyz, k=2)
    chord = np.clip(distance[:, 1], 0, 2)
    stations = stations.copy()
    stations["nearest_station_distance_km"] = 6371.0088 * 2 * np.arcsin(chord / 2)
    return stations


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize dry/wet ERA5 refractivity errors before any ML correction.")
    parser.add_argument("--config", default="configs/revision2/data_pipeline.yaml")
    parser.add_argument("--input", default="data/revision2/processed/era5_igra_profiles")
    parser.add_argument("--collocation-manifest", default="data/revision2/manifests/era5_igra_profile_manifest.json")
    args = parser.parse_args()
    config = load_data_config(args.config)
    paths = ensure_data_directories(config)
    collocation = require_collocation_manifest(
        config, args.collocation_manifest, args.input, require_formal=True
    )
    columns = ["station_id", "time", "latitude", "longitude", "pressure_hpa", "level_mask",
               "below_ground_level", "igra_height_m", "era5_height_m",
               "residual_dry", "residual_wet", "residual_total"]
    frame = pads.dataset(args.input, format="parquet", partitioning="hive").to_table(columns=columns).to_pandas()
    frame = frame.loc[frame["level_mask"].astype(bool) & ~frame["below_ground_level"].astype(bool)].copy()
    frame["month"] = pd.DatetimeIndex(frame["time"]).month
    frame["season"] = frame["month"].map({12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                                            6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"})
    frame["macro_region"] = frame["latitude"].map(macro_region)
    frame = pd.concat([frame, modified_refractivity_residual(frame)], axis=1)
    stations = station_support(frame.groupby("station_id", as_index=False).agg(latitude=("latitude", "first"), longitude=("longitude", "first")))
    frame = frame.merge(stations[["station_id", "nearest_station_distance_km"]], on="station_id", how="left")
    frame["station_support_quartile"] = pd.qcut(frame["nearest_station_distance_km"], 4, duplicates="drop").astype(str)
    rows = []
    components = ["residual_dry", "residual_wet", "residual_total"]
    group_specs = {
        "overall": [], "pressure_level": ["pressure_hpa"], "season": ["season"],
        "macro_region": ["macro_region"], "level_region": ["pressure_hpa", "macro_region"],
        "level_season": ["pressure_hpa", "season"], "station_support": ["station_support_quartile"],
    }
    for grouping, keys in group_specs.items():
        iterator = [((), frame)] if not keys else frame.groupby(keys, observed=True)
        for key, group in iterator:
            key = key if isinstance(key, tuple) else (key,)
            labels = dict(zip(keys, key))
            for component in components:
                row = {"grouping": grouping, **labels, "component": component.removeprefix("residual_")}
                row.update(metrics(group, component).to_dict())
                rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(paths["manifests"] / "era5_physical_error_summary.csv", index=False)
    m_rows = []
    for grouping, keys in group_specs.items():
        iterator = [((), frame)] if not keys else frame.groupby(keys, observed=True)
        for key, group in iterator:
            key = key if isinstance(key, tuple) else (key,)
            labels = dict(zip(keys, key))
            for component in ["residual_m_n_contribution", "residual_m_height_contribution", "residual_m"]:
                row = {"grouping": grouping, **labels, "component": component.removeprefix("residual_m_")}
                if component == "residual_m":
                    row["component"] = "total_m"
                row.update(metrics(group, component).to_dict())
                m_rows.append(row)
    pd.DataFrame(m_rows).to_csv(paths["manifests"] / "era5_modified_refractivity_error_summary.csv", index=False)
    station_means = frame.groupby(["station_id", "pressure_hpa"])[components].mean().reset_index()
    seasonal_means = frame.groupby(["station_id", "pressure_hpa", "season"])[components].mean().reset_index()
    variance_rows = []
    variance_groups = [("overall", np.nan, frame)] + [
        ("pressure_level", level, group) for level, group in frame.groupby("pressure_hpa")
    ]
    for grouping, level, group in variance_groups:
        identity = variance_identity(group)
        variance_rows.append(
            {
                "grouping": grouping, "pressure_hpa": level, "component": "dry_wet_covariance_identity",
                **identity,
                "interpretation": "exact descriptive covariance decomposition; unexplained variation is not labelled observational noise",
            }
        )
        for component in components:
            total_variance = float(group[component].var())
            station_subset = station_means if grouping == "overall" else station_means.loc[station_means["pressure_hpa"] == level]
            seasonal_subset = seasonal_means if grouping == "overall" else seasonal_means.loc[seasonal_means["pressure_hpa"] == level]
            between_station = float(station_subset[component].var())
            station_season = float(seasonal_subset[component].var())
            keys = ["station_id", "pressure_hpa", "season"]
            group_means = group.groupby(keys, observed=True)[component].transform("mean")
            within_station_season = float((group[component] - group_means).var())
            variance_rows.append(
                {"grouping": grouping, "pressure_hpa": level, "component": component.removeprefix("residual_"),
                 "row_level_variance": total_variance, "between_station_mean_variance": between_station,
                 "station_season_mean_variance": station_season,
                 "within_station_season_variance": within_station_season,
                 "interpretation": "descriptive systematic/remaining-variation diagnostics; components are not assumed orthogonal and remaining variation is not labelled observational noise"}
            )
    pd.DataFrame(variance_rows).to_csv(paths["manifests"] / "era5_error_variance_diagnostics.csv", index=False)
    stations.to_csv(paths["manifests"] / "station_observational_support.csv", index=False)
    write_json(paths["manifests"] / "physical_error_diagnostics_manifest.json", {
        "dataset_version": config["study"].get("dataset_version"),
        "input": str(Path(args.input).resolve()), "usable_level_rows": len(frame), "stations": frame["station_id"].nunique(),
        "collocation_manifest": str(Path(args.collocation_manifest).resolve()),
        "collocation_manifest_sha256": sha256_file(args.collocation_manifest),
        "igra_profile_fingerprint": collocation["igra_profile_fingerprint"],
        "outputs": ["era5_physical_error_summary.csv", "era5_modified_refractivity_error_summary.csv",
                    "era5_error_variance_diagnostics.csv", "station_observational_support.csv"],
    })
    print(summary.loc[summary["grouping"] == "overall"].to_string(index=False))


if __name__ == "__main__":
    main()
