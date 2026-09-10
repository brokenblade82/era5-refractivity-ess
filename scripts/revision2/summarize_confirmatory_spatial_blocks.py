from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from _bootstrap import PROJECT_ROOT  # noqa: F401
from igra_forecast.revision2_data import write_json


def metric(path: Path, filename: str, component: str = "total", column: str = "rmse") -> float:
    table = pd.read_csv(path / filename)
    if "component" in table:
        table = table.loc[table["component"] == component]
    return float(table.iloc[0][column])


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize paired concurrent/future Revision 2.1 spatial blocks.")
    parser.add_argument("--config", default="configs/revision2/confirmatory.yaml")
    parser.add_argument("--block-manifest", default="docs/revision2_confirmatory/spatial_block_manifest.csv")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    blocks = pd.read_csv(args.block_manifest)
    primary = str(config["evaluation"]["primary_probability_variant"])
    rows = []
    root = Path(config["output_root"]) / "spatial_leave_block_out"
    for period in config["spatial_blocks"]["periods"]:
        for block in tqdm(blocks["block"], desc=f"Summarize blocks {period}"):
            directory = root / period / block
            era5 = directory / "era5_run42"
            hgb = directory / "hgb_run42"
            if not (era5 / "protocol_manifest.json").is_file() or not (hgb / "protocol_manifest.json").is_file():
                raise FileNotFoundError(f"Incomplete block result: {period}/{block}")
            hgb_manifest = json.loads((hgb / "protocol_manifest.json").read_text(encoding="utf-8"))
            era5_rmse = metric(era5, "point_metrics.csv")
            hgb_rmse = metric(hgb, "point_metrics.csv")
            base = {
                "period": period, "block": block,
                "held_out_stations": int(blocks.set_index("block").loc[block, "held_out_stations"]),
                "level_samples": int(pd.read_csv(hgb / "point_metrics.csv").query("component == 'total'").iloc[0]["n"]),
                "era5_rmse": era5_rmse, "hgb_rmse": hgb_rmse, "hgb_minus_era5_rmse": hgb_rmse - era5_rmse,
                "station_ci_lower": float(pd.read_csv(hgb / "station_bootstrap.csv").iloc[0]["ci_lower"]),
                "station_ci_upper": float(pd.read_csv(hgb / "station_bootstrap.csv").iloc[0]["ci_upper"]),
            }
            for seed in config["evaluation"]["seeds"]:
                run = directory / f"{primary}_run{seed}"
                if not (run / "protocol_manifest.json").is_file():
                    raise FileNotFoundError(f"Incomplete probability block result: {run}")
                row = dict(base)
                row.update({
                    "model": primary, "seed": int(seed),
                    "crps": metric(run, "probabilistic_metrics.csv", "total", "crps"),
                    "energy_score": metric(run, "joint_probability_metrics.csv", "total_profile", "profile_energy_score"),
                })
                rows.append(row)
    result = pd.DataFrame(rows)
    output = root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "spatial_block_runs.csv", index=False)
    unique = result.drop_duplicates(["period", "block"])
    rng = np.random.default_rng(42)
    summary_rows = []
    replicates = int(config["evaluation"]["bootstrap_replicates"])
    for period, group in unique.groupby("period"):
        values = group["hgb_minus_era5_rmse"].to_numpy(float)
        weights = group["level_samples"].to_numpy(float)
        samples = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(replicates)])
        summary_rows.append({
            "period": period, "blocks": len(group), "mean_delta": float(values.mean()), "median_delta": float(np.median(values)),
            "sample_weighted_delta": float(np.average(values, weights=weights)), "improved_fraction": float(np.mean(values < 0)),
            "block_bootstrap_ci_lower": float(np.quantile(samples, 0.025)), "block_bootstrap_ci_upper": float(np.quantile(samples, 0.975)),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "spatial_block_summary.csv", index=False)
    write_json(output / "spatial_block_summary_manifest.json", {
        "blocks": blocks["block"].tolist(), "periods": list(config["spatial_blocks"]["periods"]),
        "interpretation": "descriptive regional pressure test; improvement counts do not replace block-bootstrap inference",
        "output": str(output.resolve()),
    })
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
