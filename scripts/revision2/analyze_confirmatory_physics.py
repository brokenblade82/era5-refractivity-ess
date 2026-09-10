from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import yaml
from scipy.stats import spearmanr
from sklearn.neighbors import BallTree

from _bootstrap import PROJECT_ROOT  # noqa: F401
from run_revision2_baselines import load_protocol_data
from igra_forecast.revision2_data import macro_region, write_json


EARTH_RADIUS_KM = 6371.0088


def error_metrics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {
        "n": int(len(values)), "bias": float(values.mean()), "mae": float(np.abs(values).mean()),
        "rmse": float(np.sqrt(np.square(values).mean())), "variance": float(np.var(values)),
    }


def observational_support(train: pd.DataFrame, evaluation: pd.DataFrame) -> pd.DataFrame:
    training = train.groupby("station_id", as_index=False).agg(latitude=("latitude", "first"), longitude=("longitude", "first"))
    target = evaluation.groupby("station_id", as_index=False).agg(latitude=("latitude", "first"), longitude=("longitude", "first"))
    tree = BallTree(np.deg2rad(training[["latitude", "longitude"]].to_numpy(float)), metric="haversine")
    coordinates = np.deg2rad(target[["latitude", "longitude"]].to_numpy(float))
    distance, _ = tree.query(coordinates, k=1)
    target["nearest_training_station_km"] = distance[:, 0] * EARTH_RADIUS_KM
    for radius in (500, 1000):
        counts = tree.query_radius(coordinates, r=radius / EARTH_RADIUS_KM, count_only=True)
        target[f"training_stations_within_{radius}km"] = counts.astype(int)
    return target


def grouped_metrics(frame: pd.DataFrame, grouping: str, column: str) -> list[dict[str, object]]:
    rows = []
    for group, selected in frame.groupby(column, observed=True, dropna=False):
        for component in ("dry", "wet", "total"):
            era5_error = -selected[f"residual_{component}"].to_numpy(float)
            corrected_error = selected[f"prediction_{component}"].to_numpy(float) - selected[f"residual_{component}"].to_numpy(float)
            era5 = error_metrics(era5_error)
            corrected = error_metrics(corrected_error)
            rows.append({
                "grouping": grouping, "group": group, "component": component,
                **{f"era5_{key}": value for key, value in era5.items()},
                **{f"hgb_{key}": value for key, value in corrected.items()},
                "hgb_minus_era5_rmse": corrected["rmse"] - era5["rmse"],
            })
    return rows


def systematic_variance_decomposition(frame: pd.DataFrame) -> pd.DataFrame:
    """Separate persistent station, station-season, and remaining variability.

    These are descriptive hierarchical components.  The remaining component
    includes representativeness, measurement, collocation, and model errors and
    is deliberately not labelled observational noise.
    """
    rows = []
    for component in ("dry", "wet", "total"):
        work = frame[["station_id", "season", f"residual_{component}"]].dropna().copy()
        value = work[f"residual_{component}"].to_numpy(float)
        overall = float(value.mean())
        station_mean = work.groupby("station_id", observed=True)[f"residual_{component}"].transform("mean").to_numpy(float)
        after_station = value - station_mean
        work["after_station"] = after_station
        station_season = work.groupby(["station_id", "season"], observed=True)["after_station"].transform("mean").to_numpy(float)
        remaining = after_station - station_season
        components = {
            "persistent_station_mean": station_mean - overall,
            "station_season_departure": station_season,
            "remaining_variation": remaining,
        }
        total_centered_variance = float(np.mean(np.square(value - overall)))
        for name, values in components.items():
            mean_square = float(np.mean(np.square(values)))
            rows.append({
                "component": component,
                "term": name,
                "samples": len(values),
                "mean_square": mean_square,
                "fraction_of_centered_mean_square": mean_square / total_centered_variance if total_centered_variance > 0 else np.nan,
            })
        reconstructed = overall + components["persistent_station_mean"] + components["station_season_departure"] + components["remaining_variation"]
        rows.append({
            "component": component,
            "term": "reconstruction_check",
            "samples": len(value),
            "mean_square": float(np.max(np.abs(reconstructed - value))),
            "fraction_of_centered_mean_square": np.nan,
        })
    return pd.DataFrame(rows)


def support_association_summary(station: pd.DataFrame) -> pd.DataFrame:
    rows = []
    outcome = station["hgb_minus_era5_rmse"].to_numpy(float)
    for column in [
        "nearest_training_station_km", "training_stations_within_500km",
        "training_stations_within_1000km", "mean_era5_n_wet", "station_elevation_m", "latitude",
    ]:
        values = station[column].to_numpy(float)
        valid = np.isfinite(values) & np.isfinite(outcome)
        coefficient, p_value = spearmanr(values[valid], outcome[valid]) if valid.sum() >= 3 else (np.nan, np.nan)
        rows.append({
            "predictor": column, "stations": int(valid.sum()), "spearman_rho": float(coefficient),
            "two_sided_p_value_descriptive": float(p_value),
            "interpretation": "descriptive association; no causal attribution",
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantify physical error sources and observational-support associations for a frozen protocol.")
    parser.add_argument("--config", default="configs/revision2/confirmatory.yaml")
    parser.add_argument("--protocol", default="space_time_holdout", choices=["spatial_station_disjoint", "temporal_holdout", "space_time_holdout"])
    parser.add_argument("--model-run", help="HGB run directory; defaults to the protocol hgb_run42.")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    model_run = Path(args.model_run or Path(config["output_root"]) / args.protocol / "hgb_run42")
    run_manifest = json.loads((model_run / "protocol_manifest.json").read_text(encoding="utf-8"))
    predictions = pd.read_parquet(model_run / "predictions.parquet")
    dataset = pads.dataset(config["input"], format="parquet", partitioning="hive")
    phases = load_protocol_data(
        config["input"], config["station_split"], config["protocol_license_matrix"],
        args.protocol, None, None,
    )
    support = observational_support(phases["train"], phases["evaluation"])
    predictions = predictions.merge(support, on=["station_id", "latitude", "longitude"], how="left", validate="many_to_one")
    predictions = predictions.loc[predictions["evaluation_mask"]].copy()
    timestamp = pd.to_datetime(predictions["time"], utc=True)
    predictions["season"] = pd.Categorical(
        np.select(
            [timestamp.dt.month.isin([12, 1, 2]), timestamp.dt.month.isin([3, 4, 5]), timestamp.dt.month.isin([6, 7, 8])],
            ["DJF", "MAM", "JJA"], default="SON",
        ), categories=["DJF", "MAM", "JJA", "SON"], ordered=True,
    )
    predictions["macro_region"] = predictions["latitude"].map(macro_region)
    predictions["elevation_band"] = pd.cut(
        predictions["station_elevation_m"], [-np.inf, 0, 500, 1500, 3000, np.inf],
        labels=["below_sea_level", "0-500", "500-1500", "1500-3000", ">3000"],
    )
    predictions["support_distance_bin"] = pd.cut(
        predictions["nearest_training_station_km"], [0, 250, 500, 1000, np.inf],
        labels=["0-250", "250-500", "500-1000", ">1000"], include_lowest=True,
    )
    wet_quantiles = predictions["era5_n_wet"].quantile([0, .25, .5, .75, 1]).to_numpy(float)
    wet_quantiles = np.maximum.accumulate(wet_quantiles + np.arange(len(wet_quantiles)) * 1e-12)
    predictions["wet_refractivity_quartile"] = pd.cut(predictions["era5_n_wet"], wet_quantiles, labels=["Q1", "Q2", "Q3", "Q4"], include_lowest=True)

    rows: list[dict[str, object]] = []
    predictions["overall"] = "all"
    for grouping, column in [
        ("overall", "overall"), ("pressure_level", "pressure_hpa"), ("season", "season"),
        ("macro_region", "macro_region"), ("elevation", "elevation_band"),
        ("observational_support", "support_distance_bin"), ("wet_refractivity", "wet_refractivity_quartile"),
    ]:
        rows.extend(grouped_metrics(predictions, grouping, column))
    output = Path(config["paper_output_root"]) / ("smoke" if run_manifest.get("smoke") else "") / "physics" / args.protocol
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "physical_component_performance.csv", index=False)
    systematic_variance_decomposition(predictions).to_csv(
        output / "systematic_variance_decomposition.csv", index=False
    )

    station = predictions.groupby("station_id", as_index=False).agg(
        latitude=("latitude", "first"), longitude=("longitude", "first"), station_elevation_m=("station_elevation_m", "first"),
        nearest_training_station_km=("nearest_training_station_km", "first"),
        training_stations_within_500km=("training_stations_within_500km", "first"),
        training_stations_within_1000km=("training_stations_within_1000km", "first"),
        mean_era5_n_wet=("era5_n_wet", "mean"), samples=("evaluation_mask", "size"),
    )
    delta = predictions.assign(
        model_sq=np.square(predictions["prediction_total"] - predictions["residual_total"]),
        era5_sq=np.square(predictions["residual_total"]),
    ).groupby("station_id").agg(model_rmse=("model_sq", lambda value: np.sqrt(value.mean())), era5_rmse=("era5_sq", lambda value: np.sqrt(value.mean())))
    delta["hgb_minus_era5_rmse"] = delta["model_rmse"] - delta["era5_rmse"]
    station = station.merge(delta.reset_index(), on="station_id", validate="one_to_one")
    station.to_csv(output / "station_support_effects.csv", index=False)
    support_association_summary(station).to_csv(output / "support_association_summary.csv", index=False)

    height_contribution = 0.157 * (predictions["igra_height_m"] - predictions["era5_height_m"])
    m_rows = []
    for name, values in {
        "era5_n_error": -predictions["residual_total"],
        "height_contribution_observed_minus_era5": height_contribution,
        "era5_m_error": -(predictions["residual_total"] + height_contribution),
        "hgb_m_error": predictions["prediction_total"] - predictions["residual_total"] - height_contribution,
    }.items():
        m_rows.append({"quantity": name, **error_metrics(values.to_numpy(float))})
    pd.DataFrame(m_rows).to_csv(output / "modified_refractivity_contributions.csv", index=False)
    write_json(output / "physical_analysis_manifest.json", {
        "protocol": args.protocol, "model_run": str(model_run.resolve()), "evaluation_rows": len(predictions),
        "interpretation": "observational-support groupings are descriptive associations, not causal effects",
        "outputs": [
            "physical_component_performance.csv", "systematic_variance_decomposition.csv",
            "station_support_effects.csv", "support_association_summary.csv",
            "modified_refractivity_contributions.csv",
        ],
    })
    print(f"Physical diagnostics: {output.resolve()}")


if __name__ == "__main__":
    main()
