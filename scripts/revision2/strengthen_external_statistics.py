from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from _bootstrap import PROJECT_ROOT  # noqa: F401
from igra_forecast.revision2_publication import cluster_bootstrap_mean, paired_rmse_units


def read_config(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def paired_metrics(frame: pd.DataFrame, grouping: str) -> pd.DataFrame:
    rows = []
    for group, selected in frame.groupby(grouping, observed=True):
        era5 = selected["era5_error"].to_numpy(float)
        model = selected["model_error"].to_numpy(float)
        rows.append({
            "grouping": grouping, "group": group, "n": len(selected),
            "era5_bias": era5.mean(), "model_bias": model.mean(), "model_minus_era5_bias": model.mean() - era5.mean(),
            "era5_mae": np.abs(era5).mean(), "model_mae": np.abs(model).mean(), "model_minus_era5_mae": np.abs(model).mean() - np.abs(era5).mean(),
            "era5_rmse": np.sqrt(np.square(era5).mean()), "model_rmse": np.sqrt(np.square(model).mean()),
            "model_minus_era5_rmse": np.sqrt(np.square(model).mean()) - np.sqrt(np.square(era5).mean()),
        })
    return pd.DataFrame(rows)


def bootstrap(frame: pd.DataFrame, unit: list[str], method: str, replicates: int, seed: int) -> dict[str, object]:
    values = paired_rmse_units(frame, unit)["rmse_difference"].to_numpy(float)
    return {"method": method, **cluster_bootstrap_mean(values, replicates, seed)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Add day-block and influence statistics to frozen external evaluations.")
    parser.add_argument("--config", default="configs/revision2/publication.yaml")
    parser.add_argument("--source", required=True, choices=["rapsodi", "cosmic2_dry_pressure"])
    parser.add_argument("--predictions")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = read_config(args.config)
    settings = config["external_statistics"]
    if args.source == "rapsodi":
        default_input = settings["rapsodi_predictions"]
        default_output = Path(config["paper_output_root"]) / "external" / "rapsodi"
    else:
        default_input = Path(settings["cosmic2_dry_pressure_root"]) / "predictions.parquet"
        default_output = Path(config["paper_output_root"]) / "external" / "cosmic2_dry_pressure_sensitivity"
    input_path = Path(args.predictions or default_input)
    output = Path(args.output_dir or default_output)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(input_path)
    valid = frame.loc[frame["evaluation_mask"]].copy()
    valid["era5_error"] = -valid["residual_total"]
    valid["model_error"] = valid["prediction_total"] - valid["residual_total"]
    valid["profile_id"] = valid["station_id"].astype(str)
    valid["date"] = pd.to_datetime(valid["time"], utc=True).dt.strftime("%Y-%m-%d")
    valid["month"] = pd.to_datetime(valid["time"], utc=True).dt.strftime("%Y-%m")
    valid["overall"] = "all"
    overall = paired_metrics(valid, "overall")
    by_level = paired_metrics(valid, "pressure_hpa")
    overall.to_csv(output / "overall_paired_metrics.csv", index=False)
    by_level.to_csv(output / "pressure_level_paired_metrics.csv", index=False)
    pd.concat([overall, by_level], ignore_index=True).to_csv(output / "paired_metrics.csv", index=False)
    replicates, seed = int(settings["bootstrap_replicates"]), int(settings["seed"])
    inference = pd.DataFrame([
        bootstrap(valid, ["profile_id", "time"], "external_profile_cluster", replicates, seed),
        bootstrap(valid, ["date"], "launch_or_sampling_day_block", replicates, seed),
        bootstrap(valid, ["month"], "month_block", replicates, seed),
    ])
    inference.to_csv(output / "external_cluster_bootstrap.csv", index=False)
    day_units = paired_rmse_units(valid, ["date"])
    leave_one_out = []
    for date in day_units["date"]:
        selected = day_units.loc[day_units["date"] != date, "rmse_difference"].to_numpy(float)
        leave_one_out.append({"omitted_date": date, "remaining_days": len(selected), "mean_rmse_difference": selected.mean() if len(selected) else np.nan})
    pd.DataFrame(leave_one_out).to_csv(output / "leave_one_day_out.csv", index=False)
    print(inference.to_string(index=False))
    print(f"Strengthened external statistics: {output.resolve()}")


if __name__ == "__main__":
    main()
