from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import PROJECT_ROOT  # noqa: F401
from igra_forecast.revision2_data import sha256_file, write_json
from igra_forecast.revision2_ess import ESS_MODELS, load_yaml


WINDOWS_ABSOLUTE = re.compile(r"[A-Za-z]:\\")


def _require(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty ESS artifact: {path}")


def _validate_release(root: Path) -> dict[str, object]:
    required = ["README.md", "CITATION.cff", "LICENSE", "DERIVED_DATA_LICENSE.txt", "environment.yml", "SHA256SUMS", "release_manifest.json"]
    for name in required:
        _require(root / name)
    failures = []
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            failures.append(relative)
    if failures:
        raise ValueError(f"Release checksum failures: {failures[:10]}")
    absolute_hits = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".md", ".json", ".yaml", ".yml", ".csv", ".txt", ".cff", ".py"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if WINDOWS_ABSOLUTE.search(text):
                absolute_hits.append(str(path.relative_to(root)))
    if absolute_hits:
        raise ValueError(f"Release contains Windows absolute paths: {absolute_hits[:10]}")
    forbidden = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".nc", ".tar", ".gz", ".zip", ".joblib", ".pt"}]
    if forbidden:
        raise ValueError(f"Release contains raw data or model binaries: {forbidden[:10]}")
    return {"release_files": sum(path.is_file() for path in root.rglob("*")), "checksum_failures": 0, "absolute_path_hits": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ESS evidence and optional minimal reproducibility release.")
    parser.add_argument("--config", default="configs/revision2/ess.yaml")
    parser.add_argument("--release-mode", action="store_true")
    parser.add_argument("--release-root")
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.release_mode:
        release_root = Path(args.release_root or config["outputs"]["release_root"])
        report = {"passed": True, "release": _validate_release(release_root)}
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    external = Path(config["outputs"]["external_root"])
    diagnostics = Path(config["outputs"]["diagnostics_root"])
    stats = Path(config["outputs"]["paper_root"]) / "statistics"
    required = [
        external / "rapsodi_model_suite/predictions.parquet",
        external / "rapsodi_model_suite/model_metrics.csv",
        external / "cosmic2_height_model_suite/model_metrics.csv",
        external / "cosmic2_height_model_suite/manifest.json",
        diagnostics / "correction_mse_decomposition.csv",
        diagnostics / "model_error_budget_comparison.csv",
        diagnostics / "probability_transfer_diagnostics.csv",
        diagnostics / "ess_diagnostic_manifest.json",
        stats / "ess_evidence_gate.json",
        stats / "ess_final_manifest.json",
        stats / "cross_model_external_transfer.csv",
    ]
    for path in required:
        _require(path)
    for name, item in config["frozen_runs"].items():
        if name == "seasonal_mean":
            path = Path(item["path"]) / "predictions.parquet"
            expected = item["frozen_prediction_sha256"]
        else:
            path = Path(item["path"]) / "model.joblib"
            expected = item["model_sha256"]
        if sha256_file(path) != expected:
            raise ValueError(f"Frozen run changed during ESS work: {name}")
    decomposition = pd.read_csv(diagnostics / "correction_mse_decomposition.csv")
    budget = pd.read_csv(diagnostics / "model_error_budget_comparison.csv")
    tolerance = float(config["statistics"]["decomposition_tolerance"])
    valid_decomposition = decomposition.loc[decomposition["comparison_valid"].astype(bool)]
    if valid_decomposition["closure_error"].abs().max() > tolerance or budget["closure_error"].abs().max() > tolerance:
        raise ValueError("ESS algebraic decomposition tolerance failed")
    probability = pd.read_csv(diagnostics / "probability_transfer_diagnostics.csv")
    if set(probability["source"]) != {"igra", "rapsodi", "cosmic2"}:
        raise ValueError("Probability-transfer diagnostics do not contain all observing systems")
    cross_model = pd.read_csv(stats / "cross_model_external_transfer.csv")
    for source in ["rapsodi", "cosmic2"]:
        if set(cross_model.loc[cross_model["source"] == source, "model"]) != set(ESS_MODELS):
            raise ValueError(f"External suite is incomplete for {source}")
    cosmic_manifest = json.loads((external / "cosmic2_height_model_suite/manifest.json").read_text(encoding="utf-8"))
    if cosmic_manifest.get("dry_pressure_used") is not False or cosmic_manifest.get("external_data_used_for_training_selection_or_calibration") is not False:
        raise ValueError("COSMIC manifest violates the frozen ESS interpretation")
    identity_max = 0.0
    mask_counts = set()
    for path in sorted((external / "cosmic2_height_model_suite/predictions").glob("*.parquet")):
        frame = pd.read_parquet(path, columns=[
            "era5_n_height", "era5_n_dry_height", "era5_n_wet_height", "evaluation_mask",
            *[f"{name}_prediction_n" for name in ESS_MODELS],
        ])
        valid = frame["evaluation_mask"].astype(bool)
        identity = np.abs(frame.loc[valid, "era5_n_height"] - frame.loc[valid, "era5_n_dry_height"] - frame.loc[valid, "era5_n_wet_height"])
        identity_max = max(identity_max, float(identity.max()) if len(identity) else 0.0)
        mask_counts.add(int(valid.sum()))
        for model in ESS_MODELS:
            if not np.isfinite(frame.loc[valid, f"{model}_prediction_n"]).all():
                raise ValueError(f"Non-finite {model} COSMIC predictions on shared evaluation rows")
    if identity_max > 1e-10:
        raise ValueError(f"COSMIC ERA5 dry+wet identity failed: {identity_max}")
    gate = json.loads((stats / "ess_evidence_gate.json").read_text(encoding="utf-8"))
    if gate.get("status") not in {"ready_for_writing", "expand_cosmic_sampling", "manual_review"}:
        raise ValueError("Unknown ESS evidence-gate status")
    report = {
        "passed": True, "gate_status": gate["status"],
        "decomposition_max_abs_closure": float(valid_decomposition["closure_error"].abs().max()),
        "decomposition_raw_max_abs_closure": float(decomposition["closure_error"].abs().max()),
        "physical_budget_max_abs_closure": float(budget["closure_error"].abs().max()),
        "cosmic_era5_component_identity_max_abs": identity_max, "probability_sources": sorted(probability["source"].unique()),
        "external_models": list(ESS_MODELS), "release": None,
    }
    output = Path(config["outputs"]["documentation_root"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "ess_validation_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
