from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from _bootstrap import PROJECT_ROOT  # noqa: F401
from igra_forecast.revision2_confirmatory import PROBABILITY_VARIANTS
from igra_forecast.revision2_data import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate formal Revision 2.1 protocol outputs and frozen-mean equality.")
    parser.add_argument("--config", default="configs/revision2/confirmatory.yaml")
    parser.add_argument("--protocol", required=True, choices=["spatial_station_disjoint", "temporal_holdout", "space_time_holdout"])
    parser.add_argument("--require-neural-comparators", action="store_true")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    root = Path(config["output_root"]) / args.protocol
    seeds = list(map(int, config["evaluation"]["seeds"]))
    expected = [(name, 42) for name in ["era5", "seasonal_mean", "ridge", "hgb"]]
    expected.extend((name, seed) for name in PROBABILITY_VARIANTS for seed in seeds)
    if args.require_neural_comparators:
        expected.extend((name, seed) for name in ["hgb_mlp", "rbf_mlp"] for seed in seeds)
    failures = []
    manifests = []
    for model, seed in expected:
        run = root / f"{model}_run{seed}"
        manifest_path = run / "protocol_manifest.json"
        if not manifest_path.is_file():
            failures.append(f"missing manifest: {run}")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests.append(manifest)
        if not manifest.get("complete") or manifest.get("smoke") or manifest.get("model") != model or int(manifest.get("seed", -1)) != seed:
            failures.append(f"invalid formal manifest: {manifest_path}")
        for name in ["predictions.parquet", "point_metrics.csv", "station_bootstrap.csv", "month_bootstrap.csv", "hierarchical_bootstrap.csv"]:
            if not (run / name).is_file():
                failures.append(f"missing {name}: {run}")
        for name in ["point_metrics.csv", "station_bootstrap.csv", "month_bootstrap.csv", "hierarchical_bootstrap.csv"]:
            path = run / name
            if path.is_file():
                numeric = pd.read_csv(path).select_dtypes(include=[np.number])
                if numeric.size and not np.isfinite(numeric.to_numpy()).all():
                    failures.append(f"non-finite metric: {path}")
        if model in PROBABILITY_VARIANTS:
            for name in ["probabilistic_metrics.csv", "coverage_curve.csv", "joint_probability_metrics.csv", "joint_coverage_curve.csv", "profile_covariance.parquet", "crossfit_station_audit.csv", "probability_license_audit.json"]:
                if not (run / name).is_file():
                    failures.append(f"missing probability artifact {name}: {run}")
            audit = manifest.get("covariance_audit", {})
            if int(audit.get("profiles_checked", 0)) <= 0 or float(audit.get("minimum_eigenvalue", -1)) <= 0:
                failures.append(f"invalid covariance audit: {manifest_path}")
            license_path = run / "probability_license_audit.json"
            if license_path.is_file():
                license_audit = json.loads(license_path.read_text(encoding="utf-8"))
                if int(license_audit.get("training_calibration_station_time_overlap", -1)) != 0:
                    failures.append(f"training/calibration station-time overlap: {license_path}")
                if license_audit.get("scale_models_use_calibration_rows") or license_audit.get("mean_models_use_calibration_rows"):
                    failures.append(f"calibration leakage flag: {license_path}")

    hgb_path = root / "hgb_run42" / "predictions.parquet"
    if hgb_path.is_file():
        hgb = pd.read_parquet(hgb_path, columns=["station_id", "time", "pressure_hpa", "prediction_total"])
        hgb = hgb.rename(columns={"prediction_total": "hgb_prediction"})
        for variant in PROBABILITY_VARIANTS:
            for seed in seeds:
                path = root / f"{variant}_run{seed}" / "predictions.parquet"
                if not path.is_file():
                    continue
                probability = pd.read_parquet(path, columns=["station_id", "time", "pressure_hpa", "prediction_total"])
                paired = hgb.merge(probability, on=["station_id", "time", "pressure_hpa"], validate="one_to_one")
                maximum = float(np.max(np.abs(paired["prediction_total"] - paired["hgb_prediction"])))
                if maximum != 0.0:
                    failures.append(f"frozen HGB mean mismatch {variant} run={seed}: {maximum}")
    report = {
        "protocol": args.protocol, "expected_runs": len(expected), "manifests_found": len(manifests),
        "frozen_hgb_mean_exactly_equal": not any("mean mismatch" in item for item in failures),
        "passed": not failures, "failures": failures,
    }
    output = root / "validation_report.json"
    write_json(output, report)
    if failures:
        raise RuntimeError("Revision 2.1 validation failed:\n- " + "\n- ".join(failures))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
