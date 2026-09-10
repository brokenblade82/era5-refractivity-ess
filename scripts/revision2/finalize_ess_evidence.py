from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from _bootstrap import PROJECT_ROOT  # noqa: F401
from igra_forecast.revision2_data import sha256_file, write_json
from igra_forecast.revision2_ess import ESS_MODELS, cosmic_sampling_checks, load_yaml


def _load_cosmic(root: Path) -> pd.DataFrame:
    files = sorted((root / "predictions").glob("predictions_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No ESS COSMIC prediction partitions found below {root}")
    return pd.concat([pd.read_parquet(path) for path in tqdm(files, desc="Load COSMIC gate partitions")], ignore_index=True)


def _rmse_delta(frame: pd.DataFrame, model: str = "hgb") -> float:
    valid = frame.loc[frame["evaluation_mask"]]
    if valid.empty:
        return float("nan")
    base = valid["era5_n_height"].to_numpy(float) - valid["observed_n"].to_numpy(float)
    corrected = valid[f"{model}_prediction_n"].to_numpy(float) - valid["observed_n"].to_numpy(float)
    return float(np.sqrt(np.square(corrected).mean()) - np.sqrt(np.square(base).mean()))


def _archive_audit(frame: pd.DataFrame, full_delta: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = frame.loc[frame["evaluation_mask"]].copy()
    profile = valid[["source_archive", "profile_id", "time"]].drop_duplicates()
    rows = []
    for archive, part in valid.groupby("source_archive", observed=True):
        omitted = valid.loc[valid["source_archive"] != archive]
        rows.append({
            "source_archive": archive,
            "evaluable_rows": int(len(part)),
            "row_fraction": float(len(part) / len(valid)),
            "profiles": int(profile.loc[profile["source_archive"] == archive, "profile_id"].nunique()),
            "leave_one_archive_delta_rmse": _rmse_delta(omitted),
            "leave_one_archive_departure": _rmse_delta(omitted) - full_delta,
        })
    archive_table = pd.DataFrame(rows)
    dates = profile.copy()
    dates["time"] = pd.to_datetime(dates["time"], utc=True)
    date_table = dates.groupby("source_archive", observed=True).agg(
        earliest_utc=("time", "min"), latest_utc=("time", "max"),
        utc_dates=("time", lambda value: int(value.dt.date.nunique())), profiles=("profile_id", "nunique"),
    ).reset_index()
    return archive_table, date_table


def _band_bootstrap(frame: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    valid = frame.loc[frame["evaluation_mask"]].copy()
    valid["height_km"] = valid["height_m"] / 1000.0
    valid["height_band"] = pd.cut(
        valid["height_km"], [0.5, 2.0, 5.0, 9.0], labels=["low", "middle", "upper"], include_lowest=True
    )
    rows = []
    rng = np.random.default_rng(seed)
    for band, part in valid.groupby("height_band", observed=True):
        part = part.copy()
        part["base_sq"] = np.square(part["era5_n_height"] - part["observed_n"])
        part["model_sq"] = np.square(part["hgb_prediction_n"] - part["observed_n"])
        aggregates = part.groupby("source_archive", observed=True).agg(
            n=("evaluation_mask", "size"), base_sq=("base_sq", "sum"), model_sq=("model_sq", "sum")
        ).to_numpy(float)
        values = np.empty(replicates, dtype=float)
        for index in range(replicates):
            selected = aggregates[rng.integers(0, len(aggregates), size=len(aggregates))].sum(axis=0)
            n, base_sq, model_sq = selected
            values[index] = math.sqrt(model_sq / n) - math.sqrt(base_sq / n)
        point = _rmse_delta(part)
        rows.append({
            "height_band": str(band), "n": int(len(part)), "archive_months": int(len(aggregates)),
            "hgb_minus_era5_rmse": point, "ci_lower": float(np.quantile(values, 0.025)),
            "ci_upper": float(np.quantile(values, 0.975)), "replicates": replicates, "seed": seed,
        })
    return pd.DataFrame(rows)


def _year_table(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame.loc[frame["evaluation_mask"]].copy()
    valid["year"] = pd.to_datetime(valid["time"], utc=True).dt.year
    return pd.DataFrame([
        {"year": int(year), "n": int(len(part)), "hgb_minus_era5_rmse": _rmse_delta(part)}
        for year, part in valid.groupby("year", observed=True)
    ])


def _cross_model_summary(rapsodi: pd.DataFrame, cosmic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, table in [("rapsodi", rapsodi), ("cosmic2", cosmic)]:
        for model in ESS_MODELS:
            selected = table.loc[table["model"] == model]
            if selected.empty:
                raise ValueError(f"Missing {source}/{model} external metric")
            delta = float(selected["model_minus_era5_rmse"].iloc[0])
            rows.append({
                "source": source, "model": model, "model_minus_era5_rmse": delta,
                "direction": "improves" if delta < 0 else "degrades" if delta > 0 else "neutral",
                "comparison_valid": bool(selected.get("comparison_valid", pd.Series([True])).iloc[0]),
                "status": str(selected.get("status", pd.Series(["valid"])).iloc[0]),
            })
    result = pd.DataFrame(rows)
    valid_result = result.loc[result["comparison_valid"]]
    pivot = valid_result.pivot(index="model", columns="source", values="direction")
    pivot = pivot.dropna(subset=["rapsodi", "cosmic2"], how="any")
    result.attrs["opposite_platform_count"] = int(((pivot["rapsodi"] == "improves") & (pivot["cosmic2"] == "degrades")).sum())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize ESS evidence and apply the preregistered COSMIC sampling gate.")
    parser.add_argument("--config", default="configs/revision2/ess.yaml")
    parser.add_argument("--cosmic-root")
    parser.add_argument("--diagnostics-root")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = load_yaml(args.config)
    external_root = Path(config["outputs"]["external_root"])
    cosmic_root = Path(args.cosmic_root or external_root / "cosmic2_height_model_suite")
    rapsodi_root = external_root / "rapsodi_model_suite"
    diagnostics_root = Path(args.diagnostics_root or config["outputs"]["diagnostics_root"])
    required_diagnostics = [
        "correction_mse_decomposition.csv", "model_error_budget_comparison.csv",
        "probability_transfer_diagnostics.csv", "probability_coverage_transfer.csv",
        "ess_diagnostic_manifest.json",
    ]
    missing = [name for name in required_diagnostics if not (diagnostics_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Run diagnose_ess_transfer.py first; missing={missing}")
    cosmic = _load_cosmic(cosmic_root)
    full_delta = _rmse_delta(cosmic)
    archive, dates = _archive_audit(cosmic, full_delta)
    replicates = int(config["statistics"]["bootstrap_replicates"])
    seed = int(config["statistics"]["seed"])
    bands = _band_bootstrap(cosmic, replicates, seed)
    years = _year_table(cosmic)
    output = Path(args.output_dir or Path(config["outputs"]["paper_root"]) / "statistics")
    output.mkdir(parents=True, exist_ok=True)
    archive.to_csv(output / "cosmic_archive_influence.csv", index=False)
    dates.to_csv(output / "cosmic_sampling_archive_audit.csv", index=False)
    bands.to_csv(output / "cosmic_height_band_block_bootstrap.csv", index=False)
    years.to_csv(output / "cosmic_year_sensitivity.csv", index=False)
    rapsodi_metrics = pd.read_csv(rapsodi_root / "model_metrics.csv")
    cosmic_metrics = pd.read_csv(cosmic_root / "model_metrics.csv")
    cross_model = _cross_model_summary(rapsodi_metrics, cosmic_metrics)
    cross_model.to_csv(output / "cross_model_external_transfer.csv", index=False)
    for name in required_diagnostics[:-1]:
        pd.read_csv(diagnostics_root / name).to_csv(output / name, index=False)
    thresholds = config["sampling_gate"]
    year_difference = float(years["hgb_minus_era5_rmse"].max() - years["hgb_minus_era5_rmse"].min())
    finite_leave_one = archive["leave_one_archive_delta_rmse"].replace([np.inf, -np.inf], np.nan).dropna()
    checks = cosmic_sampling_checks(
        full_delta,
        archive["leave_one_archive_delta_rmse"].to_numpy(float),
        archive["leave_one_archive_departure"].to_numpy(float),
        bands[["ci_lower", "ci_upper"]].to_numpy(float),
        archive["row_fraction"].to_numpy(float),
        years["hgb_minus_era5_rmse"].to_numpy(float),
        thresholds,
    )
    diagnostics_manifest = json.loads((diagnostics_root / "ess_diagnostic_manifest.json").read_text(encoding="utf-8"))
    tolerance = float(config["statistics"]["decomposition_tolerance"])
    integrity = {
        "correction_decomposition_closes": bool(diagnostics_manifest["decomposition_max_abs_closure"] <= tolerance),
        "physical_budget_closes": bool(diagnostics_manifest["physical_budget_max_abs_closure"] <= tolerance),
        "all_required_models_present": bool(set(cross_model["model"]) == set(ESS_MODELS)),
        "cosmic_archives_match_manifest": bool(cosmic["source_archive"].nunique() in {
            int(thresholds["expected_one_day_archives"]), int(thresholds["expected_expanded_archives"])
        }),
    }
    if not all(integrity.values()):
        status = "manual_review"
    elif any(checks.values()) and cosmic["source_archive"].nunique() == int(thresholds["expected_one_day_archives"]):
        status = "expand_cosmic_sampling"
    elif any(checks.values()):
        status = "manual_review"
    else:
        status = "ready_for_writing"
    opposite_count = int(cross_model.attrs["opposite_platform_count"])
    claim_scope = (
        "cross-method transfer limits supported by at least two valid prespecified correction forms"
        if opposite_count >= 2 else
        "method-specific case study; do not generalize the HGB result to statistical correction as a class"
    )
    gate = {
        "status": status,
        "cosmic_archives": int(cosmic["source_archive"].nunique()),
        "cosmic_full_hgb_minus_era5_rmse": full_delta,
        "leave_one_archive_range": (
            [float(finite_leave_one.min()), float(finite_leave_one.max())]
            if not finite_leave_one.empty else [None, None]
        ),
        "maximum_archive_row_fraction": float(archive["row_fraction"].max()),
        "year_difference": year_difference,
        "sampling_checks": checks,
        "integrity_checks": integrity,
        "opposite_platform_pattern_model_count": opposite_count,
        "recommended_claim_scope": claim_scope,
        "external_results_used_for_model_selection_or_recalibration": False,
        "next_action": {
            "ready_for_writing": "Build the reproducibility release and begin the ESS manuscript.",
            "expand_cosmic_sampling": "Run the fixed 5/15/25-day COSMIC expansion without changing the model or analysis groups.",
            "manual_review": "Stop automatic expansion and audit the failed integrity or sampling condition.",
        }[status],
    }
    write_json(output / "ess_evidence_gate.json", gate)
    manifest_inputs = [
        *(diagnostics_root / name for name in required_diagnostics),
        rapsodi_root / "manifest.json", cosmic_root / "manifest.json",
    ]
    write_json(output / "ess_final_manifest.json", {
        "complete": status != "manual_review", "status": status, "config": str(Path(args.config).resolve()),
        "gate": gate, "inputs": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in manifest_inputs],
        "outputs": [
            "cosmic_archive_influence.csv", "cosmic_sampling_archive_audit.csv",
            "cosmic_height_band_block_bootstrap.csv", "cosmic_year_sensitivity.csv",
            "cross_model_external_transfer.csv", "ess_evidence_gate.json",
        ],
        "frozen_results_overwritten": False,
    })
    print(json.dumps(gate, indent=2, ensure_ascii=False))
    print(f"ESS final evidence: {output.resolve()}")


if __name__ == "__main__":
    main()
