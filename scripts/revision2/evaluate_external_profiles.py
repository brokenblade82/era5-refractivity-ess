from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xarray as xr
import yaml

from _bootstrap import PROJECT_ROOT  # noqa: F401
from igra_forecast.revision2_baselines import metric_tables, prediction_to_rows, rows_to_profiles
from igra_forecast.revision2_confirmatory import predict_confirmatory_probability_hgb, validate_covariance_masks
from igra_forecast.revision2_data import macro_region, sha256_file, write_json


def require_external_collocation_manifest(
    input_path: Path,
    source: str,
    manifest_path: Path,
) -> tuple[dict[str, object], str]:
    """Bind external evaluation to the exact audited collocation parquet."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"External evaluation requires the upstream collocation manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_name") != source:
        raise ValueError(f"External source/manifest mismatch: requested={source}, manifest={manifest.get('source_name')}")
    if Path(manifest.get("output", "")).resolve() != input_path.resolve():
        raise ValueError("External input path does not match the audited collocation output")
    input_sha256 = sha256_file(input_path)
    if manifest.get("output_sha256") != input_sha256:
        raise ValueError("External collocation parquet hash does not match its manifest")
    output_audit = manifest.get("output_audit", {})
    if not output_audit or any(output_audit.get("nonfinite_core_on_evaluable_rows", {}).values()):
        raise ValueError("External collocation manifest lacks a passing output audit")
    return manifest, input_sha256


def land_sea_classification(frame: pd.DataFrame, static_path: Path) -> np.ndarray:
    with xr.open_dataset(static_path) as dataset:
        if "lsm" not in dataset:
            raise KeyError(f"ERA5 static file does not contain lsm: {static_path}")
        field = dataset["lsm"].squeeze(drop=True)
        latitudes = np.asarray(field["latitude"], dtype=float)
        longitudes = np.asarray(field["longitude"], dtype=float)
        target_lat = frame["latitude"].to_numpy(float)
        target_lon = np.mod(frame["longitude"].to_numpy(float), 360.0)
        if len(latitudes) < 2 or len(longitudes) < 2:
            raise ValueError(f"ERA5 static coordinates are incomplete: {static_path}")

        # The static archive is a regular global grid.  Direct index rounding
        # avoids constructing latitude/longitude-by-profile distance matrices,
        # which can otherwise require several GB for COSMIC-2.
        latitude_step = float(np.median(np.abs(np.diff(latitudes))))
        longitude_step = float(np.median(np.abs(np.diff(longitudes))))
        if not np.allclose(np.abs(np.diff(latitudes)), latitude_step, atol=1e-8):
            raise ValueError("ERA5 latitude coordinate is not regular")
        if not np.allclose(np.abs(np.diff(longitudes)), longitude_step, atol=1e-8):
            raise ValueError("ERA5 longitude coordinate is not regular")
        if latitudes[0] > latitudes[-1]:
            lat_index = np.rint((latitudes[0] - target_lat) / latitude_step).astype(int)
        else:
            lat_index = np.rint((target_lat - latitudes[0]) / latitude_step).astype(int)
        lon_index = np.rint(np.mod(target_lon - longitudes[0], 360.0) / longitude_step).astype(int)
        lat_index = np.clip(lat_index, 0, len(latitudes) - 1)
        lon_index = np.mod(lon_index, len(longitudes))
        values = np.asarray(field.values)[lat_index, lon_index]
    return np.where(values >= 0.5, "land", "sea")


def prepare_external_rows(frame: pd.DataFrame, source: str, platform_audit: Path | None) -> tuple[pd.DataFrame, dict[str, object]]:
    result = frame.copy()
    audit: dict[str, object] = {"source": source}
    if source == "rapsodi":
        if platform_audit is None or not platform_audit.is_file():
            raise FileNotFoundError("RAPSODI main validation requires --platform-audit with include_main and real_time_gts fields")
        platforms = pd.read_csv(platform_audit)
        required = {"platform", "real_time_gts", "include_main", "evidence"}
        if not required.issubset(platforms.columns):
            raise ValueError(f"RAPSODI platform audit lacks columns: {sorted(required - set(platforms.columns))}")
        def parse_bool(series: pd.Series) -> pd.Series:
            if pd.api.types.is_bool_dtype(series):
                return series.fillna(False)
            normalized = series.astype(str).str.strip().str.lower()
            if not normalized.isin(["true", "false", "1", "0", "yes", "no"]).all():
                raise ValueError("Platform audit booleans must use true/false, 1/0, or yes/no")
            return normalized.isin(["true", "1", "yes"])

        def normalize_platform(value: object) -> str:
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            text = str(value).strip().upper()
            if len(text) >= 3 and text.startswith("B'") and text.endswith("'"):
                text = text[2:-1]
            return text

        platforms["platform_normalized"] = platforms["platform"].map(normalize_platform)
        result["platform_normalized"] = result["platform"].map(normalize_platform)
        included = platforms.loc[parse_bool(platforms["include_main"]) & ~parse_bool(platforms["real_time_gts"]), "platform_normalized"]
        result = result.loc[result["platform_normalized"].isin(set(included))].copy()
        if result.empty:
            raise ValueError("No RAPSODI profiles remain after the non-real-time-GTS platform audit")
        result["station_id"] = result["platform_normalized"]
        audit.update({"platform_audit": str(platform_audit.resolve()), "platform_audit_sha256": sha256_file(platform_audit), "included_platforms": sorted(included.tolist())})
    else:
        result["station_id"] = result["profile_id"].astype(str)
        result["residual_dry"] = np.nan
        result["residual_wet"] = np.nan
    result["time"] = pd.to_datetime(result["time"], utc=True)
    result["residual_total"] = result["observed_n"] - result["era5_n"]
    if {"observed_n_dry", "observed_n_wet"}.issubset(result.columns):
        result["residual_dry"] = result["observed_n_dry"] - result["era5_n_dry"]
        result["residual_wet"] = result["observed_n_wet"] - result["era5_n_wet"]
    result["latitude"] = result["latitude"].astype(float)
    result["longitude"] = result["longitude"].astype(float)
    result["level_mask"] = result["level_mask"].astype(bool)
    result["below_ground_level"] = result["below_ground_level"].astype(bool)
    return result, audit


def grouped_total_metrics(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for group, selected in frame.loc[frame["evaluation_mask"]].groupby(column, observed=True, dropna=False):
        era5_error = -selected["residual_total"].to_numpy(float)
        model_error = selected["prediction_total"].to_numpy(float) - selected["residual_total"].to_numpy(float)
        rows.append({
            "grouping": column, "group": group, "n": len(selected),
            "era5_rmse": float(np.sqrt(np.square(era5_error).mean())),
            "model_rmse": float(np.sqrt(np.square(model_error).mean())),
            "model_minus_era5_rmse": float(np.sqrt(np.square(model_error).mean()) - np.sqrt(np.square(era5_error).mean())),
            "era5_bias": float(era5_error.mean()), "model_bias": float(model_error.mean()),
        })
    return pd.DataFrame(rows)


def external_profile_bootstrap(frame: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    """Bootstrap external profiles, rather than pretending platforms are stations."""
    valid = frame.loc[frame["evaluation_mask"]].copy()
    valid["model_sq"] = np.square(valid["prediction_total"] - valid["residual_total"])
    valid["era5_sq"] = np.square(valid["residual_total"])
    profile = valid.groupby(["station_id", "time"], observed=True).agg(
        model_mse=("model_sq", "mean"), era5_mse=("era5_sq", "mean")
    )
    values = (np.sqrt(profile["model_mse"]) - np.sqrt(profile["era5_mse"])).to_numpy(float)
    if not len(values):
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    samples = np.asarray([
        rng.choice(values, size=len(values), replace=True).mean() for _ in range(int(replicates))
    ])
    return pd.DataFrame([{
        "comparison": "model_minus_era5_rmse",
        "method": "external_profile_cluster",
        "profiles": len(values),
        "mean_difference": float(values.mean()),
        "ci_lower": float(np.quantile(samples, 0.025)),
        "ci_upper": float(np.quantile(samples, 0.975)),
        "replicates": int(replicates),
        "seed": int(seed),
    }])


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply one frozen Revision 2.1 model to external RAPSODI or COSMIC-2 profiles.")
    parser.add_argument("--config", default="configs/revision2/confirmatory.yaml")
    parser.add_argument("--source", required=True, choices=["rapsodi", "cosmic2"])
    parser.add_argument("--input", required=True, help="ERA5-collocated external profile parquet.")
    parser.add_argument("--model-run", required=True, help="Frozen structured-probability run directory.")
    parser.add_argument("--platform-audit", help="Required for the RAPSODI main subset.")
    parser.add_argument("--collocation-manifest", help="Optional upstream collocation manifest override.")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    input_path = Path(args.input)
    model_run = Path(args.model_run)
    collocation_manifest_path = Path(
        args.collocation_manifest or f"data/revision2/manifests/{args.source}_era5_collocation_manifest.json"
    )
    collocation_manifest, input_sha256 = require_external_collocation_manifest(
        input_path, args.source, collocation_manifest_path
    )
    collocation_audit = collocation_manifest.get("output_audit", {})
    manifest = json.loads((model_run / "protocol_manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("smoke"):
        raise ValueError("External evaluation requires one complete formal frozen model")
    frame, audit = prepare_external_rows(pd.read_parquet(input_path), args.source, Path(args.platform_audit) if args.platform_audit else None)
    arrays = rows_to_profiles(frame)
    fitted = joblib.load(model_run / "model.joblib")
    prediction = predict_confirmatory_probability_hgb(fitted, arrays)
    covariance_audit = validate_covariance_masks(prediction)
    predictions = prediction_to_rows(arrays, prediction, manifest["model"], int(manifest["seed"]))
    static_manifest = json.loads(Path("data/revision2/manifests/era5_archive_manifest.json").read_text(encoding="utf-8"))
    static_path = Path(static_manifest["static"]["path"])
    profile_locations = predictions[["station_id", "time", "latitude", "longitude"]].drop_duplicates()
    profile_locations["land_sea"] = land_sea_classification(profile_locations, static_path)
    predictions = predictions.merge(profile_locations, on=["station_id", "time", "latitude", "longitude"], how="left", validate="many_to_one")
    predictions["macro_region"] = predictions["latitude"].map(macro_region)
    predictions["month"] = pd.to_datetime(predictions["time"], utc=True).dt.strftime("%Y-%m")
    output = Path(args.output_dir or Path(config["paper_output_root"]) / "external" / args.source)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output / "predictions.parquet", index=False, compression="zstd")
    tables = metric_tables(
        predictions,
        list(map(float, config["evaluation"]["confidence_levels"])),
        int(config["evaluation"]["bootstrap_replicates"]),
        int(manifest["seed"]),
        int(config["evaluation"]["energy_score_draws"]),
        arrays=arrays,
        prediction=prediction,
        components=("total",) if args.source == "cosmic2" else ("dry", "wet", "total"),
    )
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    external_profile_bootstrap(
        predictions, int(config["evaluation"]["bootstrap_replicates"]), int(manifest["seed"])
    ).to_csv(output / "external_profile_bootstrap.csv", index=False)
    grouped = pd.concat([
        grouped_total_metrics(predictions, "pressure_hpa"), grouped_total_metrics(predictions, "macro_region"),
        grouped_total_metrics(predictions, "land_sea"), grouped_total_metrics(predictions, "month"),
    ], ignore_index=True)
    grouped.to_csv(output / "external_grouped_metrics.csv", index=False)
    overall = pd.read_csv(output / "point_metrics.csv").query("component == 'total'").iloc[0]
    probability_total = tables["probabilistic_metrics"].query("component == 'total'").iloc[0]
    coverage_total_90 = tables["coverage_curve"].loc[
        (tables["coverage_curve"]["component"] == "total")
        & np.isclose(tables["coverage_curve"]["nominal_coverage"], 0.90)
    ].iloc[0]
    era5_rmse = float(np.sqrt(np.square(predictions.loc[predictions["evaluation_mask"], "residual_total"]).mean()))
    model_rmse = float(overall["rmse"])
    limit = float(config["external"]["external_degradation_fraction_limit"])
    gate = {
        "source": args.source, "era5_rmse": era5_rmse, "model_rmse": model_rmse,
        "relative_degradation": (model_rmse - era5_rmse) / era5_rmse,
        "degradation_limit": limit, "passes_no_material_degradation": bool(model_rmse <= era5_rmse * (1 + limit)),
        "total_gaussian_nll": float(probability_total["gaussian_nll"]),
        "total_crps": float(probability_total["crps"]),
        "external_marginal_coverage_90": float(coverage_total_90["empirical_coverage"]),
        "external_probability_is_diagnostic_not_preregistered_gate": True,
    }
    pd.DataFrame([gate]).to_csv(output / "external_gate.csv", index=False)
    write_json(output / "external_manifest.json", {
        **audit, "input": str(input_path.resolve()), "input_sha256": input_sha256,
        "collocation_manifest": str(collocation_manifest_path.resolve()),
        "collocation_output_audit": collocation_audit,
        "model_run": str(model_run.resolve()), "model_protocol": manifest["protocol"],
        "external_profiles_never_used_for_training_selection_or_calibration": True,
        "bootstrap_unit": "external profile",
        "covariance_audit": covariance_audit, "gate": gate,
        "interpretation": config["external"]["cosmic2_interpretation"] if args.source == "cosmic2" else "platform-audited external regional validation",
    })
    print(pd.DataFrame([gate]).to_string(index=False))
    print(f"External evaluation: {output.resolve()}")


if __name__ == "__main__":
    main()
