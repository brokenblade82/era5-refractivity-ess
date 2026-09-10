from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from _bootstrap import PROJECT_ROOT  # noqa: F401
from igra_forecast.revision2_data import sha256_file, write_json


def read_config(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Revision 2.3 statistics, tables, figures, and evidence boundaries.")
    parser.add_argument("--config", default="configs/revision2/publication.yaml")
    parser.add_argument("--root")
    args = parser.parse_args()
    config = read_config(args.config)
    root = Path(args.root or config["final_output_root"])
    statistics = root / "statistics"
    failures: list[str] = []
    stats_manifest_path = statistics / "final_statistics_manifest.json"
    check(stats_manifest_path.is_file(), "Missing final_statistics_manifest.json", failures)
    manifest = json.loads(stats_manifest_path.read_text(encoding="utf-8")) if stats_manifest_path.is_file() else {}
    check(manifest.get("complete") is True and manifest.get("smoke") is False, "Final statistics are not formal and complete", failures)
    check(manifest.get("training_performed") is False, "Finalization must not train a model", failures)
    check(manifest.get("global_product_allowed") is False, "Global product boundary changed", failures)
    check(manifest.get("frozen_counts_observed") == manifest.get("frozen_counts_expected"), "Frozen COSMIC-2 counts changed", failures)
    check(float(manifest.get("maximum_mse_reconstruction_error", np.inf)) < 1e-10, "MSE reconstruction tolerance failed", failures)
    for path, expected_rows in [
        (statistics / "cosmic_height_level_metrics.csv", 18),
        (statistics / "cosmic_sampling_archive_audit.csv", 24),
    ]:
        check(path.is_file(), f"Missing {path.name}", failures)
        if path.is_file():
            check(len(pd.read_csv(path)) == expected_rows, f"Unexpected row count in {path.name}", failures)

    level_path = statistics / "cosmic_height_level_metrics.csv"
    if level_path.is_file():
        level = pd.read_csv(level_path)
        check(level["height_km"].tolist() == list(np.arange(0.5, 9.01, 0.5)), "Height grid changed", failures)
        numeric = level.select_dtypes(include=[np.number]).to_numpy(float)
        check(np.isfinite(numeric).all(), "Non-finite values in height metrics", failures)
        frozen = pd.read_csv("results_mapping/revision2_publication/cosmic2_height/point_metrics.csv").set_index("model")
        pooled_era5 = np.sqrt(np.average(np.square(level["era5_rmse"]), weights=level["n"]))
        pooled_hgb = np.sqrt(np.average(np.square(level["hgb_rmse"]), weights=level["n"]))
        check(abs(pooled_era5 - frozen.loc["era5", "rmse"]) < 1e-10, "ERA5 pooled RMSE does not match frozen output", failures)
        check(abs(pooled_hgb - frozen.loc["hgb", "rmse"]) < 1e-10, "HGB pooled RMSE does not match frozen output", failures)

    cross_path = statistics / "cross_platform_comparison.csv"
    if cross_path.is_file():
        cross = pd.read_csv(cross_path)
        required = ["era5_rmse", "hgb_rmse", "hgb_minus_era5_rmse", "ci_lower", "ci_upper", "marginal_coverage_90", "crps"]
        check(np.isfinite(cross[required].to_numpy(float)).all(), "Non-finite values in cross-platform comparison", failures)
        igra = cross.query("source == 'IGRA'")
        internal = pd.read_csv("paper_outputs/revision2_publication/internal_generalization_summary.csv")
        expected_era5 = float(internal.query("protocol == 'space_time_holdout' and model == 'era5'").iloc[0]["rmse"])
        check(len(igra) == 1 and abs(float(igra.iloc[0]["era5_rmse"]) - expected_era5) < 1e-12,
              "IGRA ERA5 RMSE is missing or inconsistent with the frozen space-time result", failures)

    for input_path, expected_hash in manifest.get("input_hashes", {}).items():
        path = Path(input_path)
        check(path.is_file(), f"Frozen input disappeared: {path}", failures)
        if path.is_file():
            check(sha256_file(path) == expected_hash, f"Frozen input hash changed: {path}", failures)

    table_manifest = root / "tables" / "tables_manifest.json"
    figure_manifest = root / "figures_manifest.json"
    check(table_manifest.is_file(), "Missing tables manifest", failures)
    check(figure_manifest.is_file(), "Missing figures manifest", failures)
    if table_manifest.is_file():
        tables = json.loads(table_manifest.read_text(encoding="utf-8"))
        check(len(tables.get("files", {})) == 42, "Expected 14 table bundles in CSV/Markdown/LaTeX", failures)
        for relative, digest in tables.get("files", {}).items():
            path = root / "tables" / relative
            check(path.is_file() and sha256_file(path) == digest, f"Table missing or changed: {relative}", failures)
    if figure_manifest.is_file():
        figures = json.loads(figure_manifest.read_text(encoding="utf-8"))
        check(len(figures.get("figures", {})) == 16, "Expected eight figures in PNG and PDF", failures)
        for relative, digest in figures.get("figures", {}).items():
            path = root / relative
            check(path.is_file() and path.stat().st_size > 1000, f"Figure missing or empty: {relative}", failures)
            if path.is_file():
                check(sha256_file(path) == digest, f"Figure hash changed: {relative}", failures)
                if path.suffix.lower() == ".png":
                    with Image.open(path) as image:
                        dpi = image.info.get("dpi", (0, 0))
                        check(min(dpi) >= 295, f"PNG is not 300 dpi: {relative} ({dpi})", failures)
                        check(image.width >= 1000 and image.height >= 600, f"PNG resolution too small: {relative}", failures)

    documentation = Path(config["final_documentation_root"])
    forbidden = ["global correction product", "well calibrated across platforms", "universal generalization advantage"]
    for path in documentation.glob("*.md") if documentation.is_dir() else []:
        text = path.read_text(encoding="utf-8").lower()
        for phrase in forbidden:
            check(phrase not in text, f"Forbidden claim in {path.name}: {phrase}", failures)
    report = {
        "passed": not failures, "failures": failures,
        "checked_counts": manifest.get("frozen_counts_observed"),
        "statistics_manifest": str(stats_manifest_path.resolve()),
        "tables_manifest": str(table_manifest.resolve()), "figures_manifest": str(figure_manifest.resolve()),
    }
    write_json(root / "validation_report.json", report)
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Revision 2.3 publication artifacts passed validation: {root.resolve()}")


if __name__ == "__main__":
    main()
