from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from _bootstrap import PROJECT_ROOT  # noqa: F401
from igra_forecast.revision2_data import macro_region, write_json
from igra_forecast.revision2_publication import cluster_bootstrap_mean, component_error_budget


def read_config(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def budget_rows(frame: pd.DataFrame, grouping: str, column: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group, selected in frame.groupby(column, observed=True, dropna=False):
        budgets = {}
        for model in ("era5", "hgb"):
            if model == "era5":
                dry = -selected["residual_dry"].to_numpy(float)
                wet = -selected["residual_wet"].to_numpy(float)
            else:
                dry = (selected["prediction_dry"] - selected["residual_dry"]).to_numpy(float)
                wet = (selected["prediction_wet"] - selected["residual_wet"]).to_numpy(float)
            budgets[model] = component_error_budget(dry, wet)
        row: dict[str, object] = {"grouping": grouping, "group": group, "samples": budgets["era5"].samples}
        for model, budget in budgets.items():
            for key, value in budget.__dict__.items():
                if key != "samples":
                    row[f"{model}_{key}"] = value
        for term in ["dry_mse", "wet_mse", "cross_term", "total_mse", "total_bias_squared", "total_centered_variance"]:
            row[f"hgb_minus_era5_{term}"] = getattr(budgets["hgb"], term) - getattr(budgets["era5"], term)
        row["era5_reconstruction_error"] = budgets["era5"].reconstructed_total_mse - budgets["era5"].total_mse
        row["hgb_reconstruction_error"] = budgets["hgb"].reconstructed_total_mse - budgets["hgb"].total_mse
        rows.append(row)
    return rows


def station_bootstrap(frame: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    station_rows = []
    for station, selected in frame.groupby("station_id", observed=True):
        era5 = component_error_budget(-selected["residual_dry"], -selected["residual_wet"])
        hgb = component_error_budget(selected["prediction_dry"] - selected["residual_dry"], selected["prediction_wet"] - selected["residual_wet"])
        row = {"station_id": station}
        for term in ["dry_mse", "wet_mse", "cross_term", "total_mse", "total_bias_squared", "total_centered_variance"]:
            row[term] = getattr(hgb, term) - getattr(era5, term)
        station_rows.append(row)
    station = pd.DataFrame(station_rows)
    rows = []
    for term in station.columns.drop("station_id"):
        rows.append({"term": term, **cluster_bootstrap_mean(station[term].to_numpy(float), replicates, seed)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dry/wet/cross-term MSE budgets for the frozen HGB evaluation.")
    parser.add_argument("--config", default="configs/revision2/publication.yaml")
    parser.add_argument("--protocol", default="space_time_holdout", choices=["spatial_station_disjoint", "temporal_holdout", "space_time_holdout"])
    parser.add_argument("--model-run")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = read_config(args.config)
    settings = config["physics"]
    model_run = Path(args.model_run or Path(config["confirmatory_root"]) / args.protocol / "hgb_run42")
    manifest = json.loads((model_run / "protocol_manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("smoke"):
        raise ValueError("Publication physics requires a complete formal frozen HGB run")
    frame = pd.read_parquet(model_run / "predictions.parquet")
    frame = frame.loc[frame["evaluation_mask"]].copy()
    timestamp = pd.to_datetime(frame["time"], utc=True)
    frame["season"] = np.select(
        [timestamp.dt.month.isin([12, 1, 2]), timestamp.dt.month.isin([3, 4, 5]), timestamp.dt.month.isin([6, 7, 8])],
        ["DJF", "MAM", "JJA"], default="SON",
    )
    frame["macro_region"] = frame["latitude"].map(macro_region)
    frame["elevation_band"] = pd.cut(frame["station_elevation_m"], [-np.inf, 0, 500, 1500, 3000, np.inf], labels=["below_sea_level", "0-500", "500-1500", "1500-3000", ">3000"])
    wet_edges = np.maximum.accumulate(frame["era5_n_wet"].quantile([0, .25, .5, .75, 1]).to_numpy(float) + np.arange(5) * 1e-12)
    frame["wet_refractivity_quartile"] = pd.cut(frame["era5_n_wet"], wet_edges, labels=["Q1", "Q2", "Q3", "Q4"], include_lowest=True)
    frame["overall"] = "all"
    rows = []
    for grouping, column in [
        ("overall", "overall"), ("pressure_level", "pressure_hpa"), ("season", "season"),
        ("macro_region", "macro_region"), ("elevation", "elevation_band"),
        ("wet_refractivity", "wet_refractivity_quartile"),
    ]:
        rows.extend(budget_rows(frame, grouping, column))
    output = Path(args.output_dir or Path(config["paper_output_root"]) / "physics" / args.protocol)
    output.mkdir(parents=True, exist_ok=True)
    budget = pd.DataFrame(rows)
    budget.to_csv(output / "physical_error_budget.csv", index=False)
    station_bootstrap(frame, int(settings["bootstrap_replicates"]), int(settings["seed"])).to_csv(output / "physical_error_budget_station_bootstrap.csv", index=False)
    maximum_reconstruction_error = float(np.nanmax(np.abs(budget[["era5_reconstruction_error", "hgb_reconstruction_error"]].to_numpy(float))))
    if maximum_reconstruction_error > 1e-8:
        raise ValueError(f"Dry/wet/cross-term MSE reconstruction failed: {maximum_reconstruction_error}")
    write_json(output / "physical_error_budget_manifest.json", {
        "complete": True, "protocol": args.protocol, "model_run": str(model_run.resolve()),
        "evaluation_rows": len(frame), "maximum_reconstruction_error": maximum_reconstruction_error,
        "remaining_variation_interpretation": "includes measurement, representativeness, collocation, and unmodelled variability; not observational noise",
    })
    print(f"Publication physical error budget: {output.resolve()}")


if __name__ == "__main__":
    main()
