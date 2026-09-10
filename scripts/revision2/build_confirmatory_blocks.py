from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from _bootstrap import PROJECT_ROOT  # noqa: F401
from igra_forecast.revision2_data import sha256_file, write_json


def assign_block(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["lat_bin"] = np.clip(np.floor((result["latitude"].astype(float) + 90.0) / 30.0).astype(int), 0, 5)
    result["lon_bin"] = np.clip(np.floor(np.mod(result["longitude"].astype(float), 360.0) / 60.0).astype(int), 0, 5)
    result["block"] = "lat" + result["lat_bin"].astype(str) + "_lon" + result["lon_bin"].astype(str)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the data-only, pre-performance Revision 2.1 spatial-block list.")
    parser.add_argument("--config", default="configs/revision2/confirmatory.yaml")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    split_path = Path(config["station_split"])
    station = assign_block(pd.read_csv(split_path, dtype={"station_id": str}))
    summary = station.groupby(["block", "lat_bin", "lon_bin"], as_index=False).agg(
        held_out_stations=("station_id", "nunique"),
        latitude_mean=("latitude", "mean"), longitude_mean=("longitude", "mean"),
    )
    settings = config["spatial_blocks"]
    eligible = summary.loc[summary["held_out_stations"] >= int(settings["minimum_stations"])].copy()
    eligible = eligible.sort_values(["held_out_stations", "block"], ascending=[False, True])
    southern_required = int(settings.get("require_southern_blocks", 0))
    southern = eligible.loc[eligible["lat_bin"] <= 1].head(southern_required)
    remaining = eligible.loc[~eligible["block"].isin(southern["block"])]
    target = int(settings["target_blocks"])
    selected = pd.concat([southern, remaining.head(target - len(southern))], ignore_index=True)
    if len(selected) != target:
        raise ValueError(f"Only {len(selected)} eligible blocks are available; expected {target}")
    selected["selection_rank"] = np.arange(1, len(selected) + 1)
    selected["latitude_min"] = -90.0 + selected["lat_bin"] * 30.0
    selected["latitude_max"] = selected["latitude_min"] + 30.0
    selected["longitude_min_0_360"] = selected["lon_bin"] * 60.0
    selected["longitude_max_0_360"] = selected["longitude_min_0_360"] + 60.0
    selected["selection_rule"] = "minimum station gate; reserve required southern blocks; then descending station count and block name"
    output_root = Path(config["documentation_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / "spatial_block_manifest.csv"
    selected.to_csv(output, index=False)
    write_json(output.with_suffix(".json"), {
        "station_split": str(split_path.resolve()), "station_split_sha256": sha256_file(split_path),
        "target_blocks": target, "minimum_stations":int(settings["minimum_stations"]),
        "require_southern_blocks": southern_required, "blocks": selected["block"].tolist(),
        "selection_uses_model_performance": False, "output": str(output.resolve()),
    })
    print(selected[["selection_rank", "block", "held_out_stations", "latitude_min", "latitude_max", "longitude_min_0_360", "longitude_max_0_360"]].to_string(index=False))
    print(f"Block manifest: {output.resolve()}")


if __name__ == "__main__":
    main()
