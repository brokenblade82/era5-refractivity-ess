from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.dataset as pads

from _bootstrap import PROJECT_ROOT  # noqa: F401
from igra_forecast.revision2_data import (
    ensure_data_directories,
    load_data_config,
    require_igra_profile_manifest,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize station/profile retention and missing-level selection effects.")
    parser.add_argument("--config", default="configs/revision2/data_pipeline.yaml")
    parser.add_argument("--station-retention", default="data/revision2/manifests/igra_station_retention.csv")
    parser.add_argument("--profiles", default="data/revision2/processed/igra_profiles")
    parser.add_argument("--profile-manifest", default="data/revision2/manifests/igra_profile_manifest.json")
    args = parser.parse_args()
    config = load_data_config(args.config)
    paths = ensure_data_directories(config)
    profile_manifest = require_igra_profile_manifest(
        config, args.profile_manifest, args.profiles, require_formal=True
    )
    stations = pd.read_csv(args.station_retention)
    stations["elevation_band"] = pd.cut(
        stations["station_elevation_m"], [-float("inf"), 0, 500, 1500, 3000, float("inf")],
        labels=["below_sea_level", "0-500", "500-1500", "1500-3000", ">3000"],
    ).astype(object)
    stations.loc[stations["elevation_band"].isna(), "elevation_band"] = "missing"
    station_summary = (
        stations.groupby(["macro_region", "elevation_band"], observed=True)
        .agg(candidate_stations=("station_id", "nunique"), eligible_stations=("eligible", "sum"),
             median_retained_profiles=("retained_profiles", "median"))
        .reset_index()
    )
    station_summary["retention_fraction"] = station_summary["eligible_stations"] / station_summary["candidate_stations"]
    station_summary.to_csv(paths["manifests"] / "station_retention_summary.csv", index=False)
    dataset = pads.dataset(args.profiles, format="parquet", partitioning="hive")
    frame = dataset.to_table(columns=["station_id", "pressure_hpa", "level_mask", "latitude", "station_elevation_m"]).to_pandas()
    frame["elevation_band"] = pd.cut(
        frame["station_elevation_m"], [-float("inf"), 0, 500, 1500, 3000, float("inf")],
        labels=["below_sea_level", "0-500", "500-1500", "1500-3000", ">3000"],
    ).astype(object)
    frame.loc[frame["elevation_band"].isna(), "elevation_band"] = "missing"
    missing_summary = (
        frame.groupby(["pressure_hpa", "elevation_band"], observed=True)["level_mask"]
        .agg(level_rows="size", valid_rows="sum").reset_index()
    )
    missing_summary["missing_fraction"] = 1 - missing_summary["valid_rows"] / missing_summary["level_rows"]
    missing_summary.to_csv(paths["manifests"] / "missing_level_summary.csv", index=False)
    counter_columns = [name for name in ["raw_soundings", "outside_period", "outside_synoptic_hours",
                                         "insufficient_valid_levels", "retained_profiles"] if name in stations]
    stepwise = pd.DataFrame({"stage_counter": counter_columns, "count": [stations[name].sum() for name in counter_columns]})
    stepwise.to_csv(paths["manifests"] / "igra_stepwise_counts.csv", index=False)
    write_json(paths["manifests"] / "data_retention_manifest.json", {
        "dataset_version": config["study"].get("dataset_version"),
        "station_retention": str(Path(args.station_retention).resolve()), "profiles": str(Path(args.profiles).resolve()),
        "igra_profile_fingerprint": profile_manifest["data_fingerprint"],
        "outputs": ["station_retention_summary.csv", "missing_level_summary.csv", "igra_stepwise_counts.csv"]})
    print(station_summary.to_string(index=False))


if __name__ == "__main__":
    main()
