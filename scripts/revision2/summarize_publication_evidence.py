from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from _bootstrap import PROJECT_ROOT  # noqa: F401
from igra_forecast.revision2_data import sha256_file, write_json


def read_config(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def require(path: str | Path) -> Path:
    result = Path(path)
    if not result.is_file():
        raise FileNotFoundError(f"Publication evidence input is missing: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the Revision 2.2 publication evidence package.")
    parser.add_argument("--config", default="configs/revision2/publication.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--cosmic-height-root")
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    config = read_config(args.config)
    confirmatory = Path(config["confirmatory_root"])
    confirmatory_paper = Path(config["confirmatory_paper_root"])
    output = Path(args.output_dir or config["paper_output_root"])
    output.mkdir(parents=True, exist_ok=True)

    internal_rows = []
    for protocol, evidence in [
        ("spatial_station_disjoint", "development"),
        ("temporal_holdout", "supporting"),
        ("space_time_holdout", "primary_confirmatory"),
    ]:
        summary_path = require(confirmatory / protocol / "summary" / "repeated_run_summary.csv")
        gate_path = require(confirmatory / protocol / "summary" / "point_gate.csv")
        summary = pd.read_csv(summary_path)
        gate = pd.read_csv(gate_path).iloc[0]
        for model in ["era5", "hgb", "hgb_mlp", "rbf_mlp"]:
            selected = summary.loc[summary["model"] == model]
            if selected.empty:
                continue
            row = selected.iloc[0]
            internal_rows.append({
                "protocol": protocol, "evidence_level": evidence, "model": model,
                "runs": row["runs"], "rmse": row["point_rmse_mean"], "mae": row["point_mae_mean"], "bias": row["point_bias_mean"],
                "station_mean_rmse_difference_vs_era5": row["station_mean_difference_mean"],
                "station_ci_lower": row["station_ci_lower_mean"], "station_ci_upper": row["station_ci_upper_mean"],
                "hgb_point_gate_passed": bool(gate["passes"]),
            })
    pd.DataFrame(internal_rows).to_csv(output / "internal_generalization_summary.csv", index=False)

    physics_path = require(Path(config["paper_output_root"]) / "physics" / "space_time_holdout" / "physical_error_budget.csv")
    physics = pd.read_csv(physics_path)
    physics.to_csv(output / "physical_error_budget.csv", index=False)

    probability_rows = []
    internal_probability = pd.read_csv(require(confirmatory / "space_time_holdout" / "summary" / "repeated_run_summary.csv"))
    for model in ["hgb_prob_global", "hgb_prob_hetero_diag", "hgb_prob_hetero_structured"]:
        row = internal_probability.loc[internal_probability["model"] == model].iloc[0]
        probability_rows.append({
            "domain": "IGRA_space_time_holdout", "coordinate": "pressure_levels", "model": model,
            "crps": row["prob_crps_mean"], "energy_score": row["prob_profile_energy_score_mean"],
            "marginal_coverage_90": row["marginal_coverage_90_mean"], "evidence_level": "primary_confirmatory_internal",
        })
    rapsodi_probability = pd.read_csv(require(confirmatory_paper / "external" / "rapsodi" / "probabilistic_metrics.csv")).query("component == 'total'").iloc[0]
    rapsodi_coverage = pd.read_csv(require(confirmatory_paper / "external" / "rapsodi" / "coverage_curve.csv")).query("component == 'total' and nominal_coverage == 0.9").iloc[0]
    probability_rows.append({"domain": "RAPSODI_INMG", "coordinate": "measured_pressure", "model": "frozen_structured_HGB", "crps": rapsodi_probability["crps"], "energy_score": rapsodi_probability.get("profile_energy_score"), "marginal_coverage_90": rapsodi_coverage["empirical_coverage"], "evidence_level": "external_regional_diagnostic"})
    old_cosmic = pd.read_csv(require(confirmatory_paper / "external" / "cosmic2" / "probabilistic_metrics.csv")).query("component == 'total'").iloc[0]
    old_coverage = pd.read_csv(require(confirmatory_paper / "external" / "cosmic2" / "coverage_curve.csv")).query("component == 'total' and nominal_coverage == 0.9").iloc[0]
    probability_rows.append({"domain": "COSMIC2", "coordinate": "dry_pressure_sensitivity", "model": "frozen_structured_HGB", "crps": old_cosmic["crps"], "energy_score": old_cosmic.get("profile_energy_score"), "marginal_coverage_90": old_coverage["empirical_coverage"], "evidence_level": "coordinate_sensitivity"})
    height_root = Path(args.cosmic_height_root or Path(config["output_root"]) / "cosmic2_height")
    height_probability = pd.read_csv(require(height_root / "probabilistic_metrics.csv")).iloc[0]
    probability_rows.append({"domain": "COSMIC2", "coordinate": "MSL_geometric_height", "model": "frozen_structured_HGB", "crps": height_probability["crps"], "energy_score": None, "marginal_coverage_90": height_probability["marginal_coverage_90"], "evidence_level": "primary_cross_platform_diagnostic"})
    pd.DataFrame(probability_rows).to_csv(output / "probability_transfer_summary.csv", index=False)

    external_rows = []
    rapsodi_gate = pd.read_csv(require(confirmatory_paper / "external" / "rapsodi" / "external_gate.csv")).iloc[0]
    external_rows.append({"source": "RAPSODI_INMG", "coordinate": "measured_pressure", "era5_rmse": rapsodi_gate["era5_rmse"], "model_rmse": rapsodi_gate["model_rmse"], "model_minus_era5_rmse": rapsodi_gate["model_rmse"] - rapsodi_gate["era5_rmse"], "evidence_level": "external_regional"})
    old_gate = pd.read_csv(require(confirmatory_paper / "external" / "cosmic2" / "external_gate.csv")).iloc[0]
    external_rows.append({"source": "COSMIC2", "coordinate": "dry_pressure_sensitivity", "era5_rmse": old_gate["era5_rmse"], "model_rmse": old_gate["model_rmse"], "model_minus_era5_rmse": old_gate["model_rmse"] - old_gate["era5_rmse"], "evidence_level": "coordinate_sensitivity"})
    height_point = pd.read_csv(require(height_root / "point_metrics.csv")).set_index("model")
    external_rows.append({"source": "COSMIC2", "coordinate": "MSL_geometric_height", "era5_rmse": height_point.loc["era5", "rmse"], "model_rmse": height_point.loc["hgb", "rmse"], "model_minus_era5_rmse": height_point.loc["hgb", "rmse"] - height_point.loc["era5", "rmse"], "evidence_level": "primary_cross_platform_diagnostic"})
    pd.DataFrame(external_rows).to_csv(output / "external_validation_summary.csv", index=False)

    application = pd.read_csv(require(confirmatory_paper / "application" / "space_time_holdout" / "application_summary.csv"))
    application["evidence_level"] = "supplementary_diagnostic"
    application.to_csv(output / "application_evidence_summary.csv", index=False)

    matrix = pd.DataFrame([
        ["methodological_novelty", "attention/context unsupported", "remove architecture claim; use HGB as analysis tool", "resolved_by_repositioning", "Introduction; Methods"],
        ["physical_interpretability", "no dry/wet attribution", "six-level dry/wet/cross-term MSE budget", "implemented", "Results"],
        ["temporal_generalization", "no future holdout", "temporal and space-time holdouts", "implemented", "Results"],
        ["spatial_heterogeneity", "block CI crossed zero", "20 concurrent and future blocks with block bootstrap", "implemented_limit_confirmed", "Results; Discussion"],
        ["external_validation", "global product unsupported", "RAPSODI plus COSMIC-2 geometric-height comparison", "implemented_after_height_run", "Results"],
        ["probability_transfer", "internal coverage overinterpreted", "internal/external coverage and CRPS separated", "implemented", "Results; Discussion"],
        ["application_value", "duct/propagation claims unsupported", "retain only integral/gradient sensitivity in supplement", "claim_removed", "Supplementary"],
    ], columns=["review_issue", "original_problem", "modification_action", "status", "affected_sections"])
    matrix.to_csv(output / "reviewer_response_evidence_matrix.csv", index=False)

    model_path = Path(config["frozen_model_run"]) / "model.joblib"
    height_manifest_path = require(height_root / "manifest.json")
    height_manifest = json.loads(height_manifest_path.read_text(encoding="utf-8"))
    if height_manifest.get("smoke") and not args.allow_smoke:
        raise ValueError("A smoke COSMIC-height result cannot enter the formal publication evidence package")
    input_paths = [
        Path("data/revision2/manifests/revision2_data_audit.json"), model_path, height_manifest_path,
        confirmatory / "space_time_holdout" / "summary" / "point_gate.csv",
        confirmatory / "spatial_leave_block_out" / "summary" / "spatial_block_summary.csv",
    ]
    write_json(output / "publication_evidence_manifest.json", {
        "complete": True, "smoke": bool(height_manifest.get("smoke")),
        "paper_positioning": "ERA5 refractivity physical error diagnosis and cross-domain generalization",
        "frozen_model": str(model_path.resolve()), "frozen_model_sha256": sha256_file(model_path),
        "cosmic2_primary_coordinate": "MSL_geometric_height",
        "cosmic2_qc": "official bad == 0",
        "cosmic2_height_manifest": str(height_manifest_path.resolve()),
        "cosmic2_dry_pressure_result": str((confirmatory_paper / "external" / "cosmic2").resolve()),
        "cosmic2_dry_pressure_interpretation": "dry-pressure-coordinate sensitivity result",
        "bootstrap_units": ["station", "spatial block", "external profile", "sampling day", "month"],
        "evidence_levels": {"space_time_holdout": "primary confirmatory", "RBF_MLP": "exploratory comparator", "COSMIC2_dry_pressure": "sensitivity"},
        "global_product_allowed": False,
        "global_product_prohibition_reason": "Primary space-time point gate did not pass and external transfer is domain dependent.",
        "input_hashes": {str(path.resolve()): sha256_file(path) for path in input_paths},
        "outputs": [
            "internal_generalization_summary.csv", "physical_error_budget.csv", "probability_transfer_summary.csv",
            "external_validation_summary.csv", "application_evidence_summary.csv", "reviewer_response_evidence_matrix.csv",
        ],
    })
    print(f"Publication evidence package: {output.resolve()}")


if __name__ == "__main__":
    main()
