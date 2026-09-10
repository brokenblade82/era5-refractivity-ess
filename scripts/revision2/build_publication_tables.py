from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
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
        raise FileNotFoundError(f"Required final-statistics input is missing: {result}")
    return result


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in frame.fillna("").astype(str).itertuples(index=False, name=None):
        rows.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
    return "\n".join(rows) + "\n"


def latex_escape(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    for source, target in [("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")]:
        text = text.replace(source, target)
    return text


def latex_table(frame: pd.DataFrame, caption: str, label: str) -> str:
    alignment = "l" + "r" * max(0, len(frame.columns) - 1)
    lines = [
        r"\begin{table*}[t]", r"\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        r"\small", f"\\begin{{tabular}}{{{alignment}}}", r"\toprule",
        " & ".join(latex_escape(column) for column in frame.columns) + r" \\", r"\midrule",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(" & ".join(latex_escape(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def formatted(frame: pd.DataFrame, formats: dict[str, str]) -> pd.DataFrame:
    result = frame.copy()
    for column, pattern in formats.items():
        if column in result:
            result[column] = result[column].map(lambda value: "" if pd.isna(value) else format(float(value), pattern))
    return result


def write_bundle(root: Path, name: str, frame: pd.DataFrame, caption: str) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / f"{name}.csv"
    md_path = root / f"{name}.md"
    tex_path = root / f"{name}.tex"
    frame.to_csv(csv_path, index=False)
    md_path.write_text(f"# {caption}\n\n{markdown_table(frame)}", encoding="utf-8")
    tex_path.write_text(latex_table(frame, caption, f"tab:{name}"), encoding="utf-8")
    return [csv_path, md_path, tex_path]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build journal-neutral Revision 2.3 publication tables.")
    parser.add_argument("--config", default="configs/revision2/publication.yaml")
    parser.add_argument("--statistics-dir")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = read_config(args.config)
    final_root = Path(config["final_output_root"])
    statistics = Path(args.statistics_dir or final_root / "statistics")
    manifest = json.loads(require(statistics / "final_statistics_manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("smoke"):
        raise ValueError("Tables require complete formal final statistics")
    output = Path(args.output_dir or final_root / "tables")
    main_root, supplementary_root = output / "main", output / "supplementary"
    written: list[Path] = []

    audit = json.loads(require("data/revision2/manifests/revision2_data_audit.json").read_text(encoding="utf-8"))
    cosmic = json.loads(require("results_mapping/revision2_publication/cosmic2_height/manifest.json").read_text(encoding="utf-8"))
    table1 = pd.DataFrame([
        ["IGRA", "Radiosonde", "Standard pressure", audit["stations"], audit["profiles"], audit["rows"], "Training and internal evaluation"],
        ["RAPSODI/INMG", "Radiosonde", "Measured pressure", 1, 237, 1299, "Regional external diagnostic"],
        ["COSMIC-2 atmPrf", "GNSS radio occultation", "MSL geometric height", "--", cosmic["audit"]["profiles"], cosmic["audit"]["evaluable_rows"], "Cross-platform diagnostic"],
    ], columns=["Dataset", "Observing system", "Vertical coordinate", "Stations/platforms", "Profiles", "Level/height samples", "Role"])
    written += write_bundle(main_root, "table01_data_and_observing_systems", table1, "Datasets, observing systems, and evidential roles.")

    budget = pd.read_csv(require(Path(config["paper_output_root"]) / "physical_error_budget.csv"))
    budget = budget.loc[budget["grouping"].isin(["overall", "pressure_level"])].copy()
    budget["Level"] = np.where(budget["grouping"] == "overall", "Overall", budget["group"].astype(str) + " hPa")
    table2 = budget[["Level", "hgb_minus_era5_dry_mse", "hgb_minus_era5_wet_mse", "hgb_minus_era5_cross_term", "hgb_minus_era5_total_mse"]]
    table2.columns = ["Level", "Delta dry MSE", "Delta wet MSE", "Delta cross term", "Delta total MSE"]
    table2 = formatted(table2, {column: ".3f" for column in table2.columns[1:]})
    written += write_bundle(main_root, "table02_physical_error_budget", table2, "Changes in the dry, wet, cross, and total mean-squared-error terms (HGB minus ERA5).")

    internal = pd.read_csv(require(Path(config["paper_output_root"]) / "internal_generalization_summary.csv"))
    table3 = internal.loc[internal["model"].isin(["era5", "hgb", "hgb_mlp", "rbf_mlp"]), [
        "protocol", "evidence_level", "model", "runs", "rmse", "mae", "bias",
        "station_mean_rmse_difference_vs_era5", "station_ci_lower", "station_ci_upper",
    ]]
    table3 = formatted(table3, {column: ".3f" for column in ["rmse", "mae", "bias", "station_mean_rmse_difference_vs_era5", "station_ci_lower", "station_ci_upper"]})
    written += write_bundle(main_root, "table03_internal_generalization", table3, "Internal spatial, temporal, and space-time generalization results.")

    probability = pd.read_csv(require(Path(config["paper_output_root"]) / "probability_transfer_summary.csv"))
    table4 = probability[["domain", "coordinate", "model", "crps", "energy_score", "marginal_coverage_90", "evidence_level"]]
    table4 = formatted(table4, {"crps": ".3f", "energy_score": ".3f", "marginal_coverage_90": ".3f"})
    written += write_bundle(main_root, "table04_probability_transfer", table4, "Internal calibration and cross-platform predictive-interval transfer.")

    cross = pd.read_csv(require(statistics / "cross_platform_comparison.csv"))
    table5 = formatted(cross, {column: ".3f" for column in ["era5_rmse", "hgb_rmse", "hgb_minus_era5_rmse", "ci_lower", "ci_upper", "marginal_coverage_90", "crps"]})
    written += write_bundle(main_root, "table05_cross_platform_comparison", table5, "Cross-platform comparisons with inference units matched to each observing system.")

    cosmic_group = pd.read_csv(require(statistics / "cosmic_height_band_region_metrics.csv"))
    table6 = cosmic_group.loc[cosmic_group["grouping"].isin(["height_band", "macro_region", "land_sea"]), [
        "grouping", "height_band", "macro_region", "land_sea", "n", "era5_rmse", "hgb_rmse", "hgb_minus_era5_rmse", "coverage_90",
    ]]
    table6 = formatted(table6, {"era5_rmse": ".3f", "hgb_rmse": ".3f", "hgb_minus_era5_rmse": ".3f", "coverage_90": ".3f"})
    written += write_bundle(main_root, "table06_cosmic_height_and_region", table6, "COSMIC-2 geometric-height results by altitude, latitude region, and surface type.")

    supplementary = [
        ("tableS01_cosmic_height_levels", statistics / "cosmic_height_level_metrics.csv", "COSMIC-2 metrics at all 18 fixed MSL heights."),
        ("tableS02_cosmic_stratified", statistics / "cosmic_height_band_region_metrics.csv", "Complete stratified COSMIC-2 diagnostics."),
        ("tableS03_cosmic_block_bootstrap", statistics / "cosmic_height_block_bootstrap.csv", "Archive-month bootstrap results."),
        ("tableS04_sampling_archive_audit", statistics / "cosmic_sampling_archive_audit.csv", "Audit of preregistered CDAAC archive days and UTC date spans."),
        ("tableS05_physical_level_bootstrap", statistics / "physical_budget_level_bootstrap.csv", "Pressure-level station-cluster bootstrap for the physical error budget."),
        ("tableS06_physical_region_bootstrap", statistics / "physical_budget_region_bootstrap.csv", "Regional station-cluster bootstrap for the physical error budget."),
        ("tableS07_spatial_blocks", Path(config["confirmatory_root"]) / "spatial_leave_block_out" / "summary" / "spatial_block_runs.csv", "Complete spatial-block pressure tests."),
        ("tableS08_application_diagnostics", Path(config["paper_output_root"]) / "application_evidence_summary.csv", "Integral and adjacent-level gradient diagnostics."),
    ]
    for name, path, caption in supplementary:
        written += write_bundle(supplementary_root, name, pd.read_csv(require(path)), caption)

    write_json(output / "tables_manifest.json", {
        "complete": True, "source_statistics_manifest_sha256": sha256_file(statistics / "final_statistics_manifest.json"),
        "files": {str(path.relative_to(output)): sha256_file(path) for path in written},
        "formatting": {"rmse_mae_bias": "3 decimals", "small_differences": "3 decimals", "coverage": "proportion"},
    })
    print(f"Publication tables: {output.resolve()}")


if __name__ == "__main__":
    main()
