from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm

from _bootstrap import PROJECT_ROOT  # noqa: F401
from igra_forecast.revision2_data import sha256_file, write_json
from igra_forecast.revision2_ess import (
    ESS_MODELS,
    cluster_bootstrap_decomposition,
    correction_decomposition,
    load_yaml,
    load_protocol_training_rows,
    pit_histogram,
    probability_diagnostics,
    training_wet_edges,
)


def _cosmic_frame(root: Path) -> pd.DataFrame:
    files = sorted((root / "predictions").glob("predictions_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No COSMIC ESS prediction partitions found below {root}")
    return pd.concat([pd.read_parquet(path) for path in tqdm(files, desc="Load COSMIC ESS partitions")], ignore_index=True)


def _model_columns(source: str, model: str) -> tuple[str, str]:
    if source == "cosmic2":
        return "residual_total", f"{model}_correction_total"
    return "residual_total", f"{model}_correction_total"


def _add_source_residual(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    result = frame.copy()
    if source == "cosmic2":
        result["residual_total"] = result["observed_n"] - result["era5_n_height"]
        result["vertical_coordinate"] = result["height_m"] / 1000.0
        result["vertical_kind"] = "height_km_msl"
        result["height_band"] = pd.cut(
            result["vertical_coordinate"], [0.5, 2.0, 5.0, 9.0],
            labels=["0.5-2", "2-5", "5-9"], include_lowest=True,
        )
    else:
        result["vertical_coordinate"] = result["pressure_hpa"].astype(float)
        result["vertical_kind"] = "pressure_hpa"
    return result


def _decomposition_table(frame: pd.DataFrame, source: str, grouping: str | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected = frame.loc[frame["evaluation_mask"]].copy()
    groups: Iterable[tuple[Any, pd.DataFrame]]
    groups = [("overall", selected)] if grouping is None else selected.groupby(grouping, observed=True, dropna=False)
    for group, part in groups:
        for model in ESS_MODELS:
            values = correction_decomposition(part["residual_total"], part[f"{model}_correction_total"])
            support_fraction = (
                float(part["ridge_feature_support"].mean())
                if model == "ridge" and "ridge_feature_support" in part else 1.0
            )
            rows.append({
                "source": source, "model": model, "grouping": grouping or "overall", "group": str(group),
                "feature_support_fraction": support_fraction,
                "comparison_valid": bool(support_fraction == 1.0),
                "status": (
                    "valid" if support_fraction == 1.0 else "out_of_training_constant_feature_support"
                ),
                **values,
            })
    return pd.DataFrame(rows)


def _bootstrap_table(frame: pd.DataFrame, source: str, replicates: int, seed: int) -> pd.DataFrame:
    selected = frame.loc[frame["evaluation_mask"]].copy()
    clusters = {
        "igra": ["station_id"],
        "rapsodi": ["date"],
        "cosmic2": ["source_archive"],
    }[source]
    rows = []
    for model in ESS_MODELS:
        result = cluster_bootstrap_decomposition(
            selected, "residual_total", f"{model}_correction_total", clusters, replicates, seed
        )
        support_fraction = (
            float(selected["ridge_feature_support"].mean())
            if model == "ridge" and "ridge_feature_support" in selected else 1.0
        )
        rows.append({
            "source": source, "model": model, "bootstrap_unit": "+".join(clusters),
            "feature_support_fraction": support_fraction,
            "comparison_valid": bool(support_fraction == 1.0),
            "status": "valid" if support_fraction == 1.0 else "out_of_training_constant_feature_support",
            **result,
        })
    return pd.DataFrame(rows)


def _internal_suite(config: dict[str, Any]) -> pd.DataFrame:
    root = Path(config["confirmatory_config"])
    confirmatory = load_yaml(root)
    protocol_root = Path(confirmatory["output_root"]) / "space_time_holdout"
    tables: list[pd.DataFrame] = []
    for model in ESS_MODELS:
        run = protocol_root / f"{model}_run42"
        frame = pd.read_parquet(run / "predictions.parquet")
        result = frame.copy()
        result[f"{model}_correction_total"] = result["prediction_total"]
        result[f"{model}_correction_dry"] = result["prediction_dry"]
        result[f"{model}_correction_wet"] = result["prediction_wet"]
        result["date"] = pd.to_datetime(result["time"], utc=True).dt.strftime("%Y-%m-%d")
        result["macro_region"] = np.where(
            result["latitude"] < -30, "southern_extratropics",
            np.where(result["latitude"] > 30, "northern_extratropics", "tropics"),
        )
        keep = [
            "station_id", "time", "latitude", "pressure_hpa", "era5_n_wet", "residual_dry",
            "residual_wet", "residual_total", "evaluation_mask", "date", "macro_region",
            f"{model}_correction_dry", f"{model}_correction_wet", f"{model}_correction_total",
        ]
        tables.append(result[keep].assign(model_source=model))
    # Convert the three long model-specific frames to one common wide frame.
    base_columns = [
        "station_id", "time", "latitude", "pressure_hpa", "era5_n_wet", "residual_dry",
        "residual_wet", "residual_total", "evaluation_mask", "date", "macro_region",
    ]
    merged = tables[0][base_columns + [f"{ESS_MODELS[0]}_correction_dry", f"{ESS_MODELS[0]}_correction_wet", f"{ESS_MODELS[0]}_correction_total"]]
    for index, model in enumerate(ESS_MODELS[1:], start=1):
        columns = base_columns + [f"{model}_correction_dry", f"{model}_correction_wet", f"{model}_correction_total"]
        merged = merged.merge(tables[index][columns], on=base_columns, how="inner", validate="one_to_one")
    training = load_protocol_training_rows(config["input"], config["station_split"])
    edges = training_wet_edges(training)
    edges[0], edges[-1] = -np.inf, np.inf
    merged["wet_regime"] = pd.cut(
        merged["era5_n_wet"], edges, labels=["Q1", "Q2", "Q3", "Q4"], include_lowest=True
    )
    return merged


def _physical_budget(config: dict[str, Any]) -> pd.DataFrame:
    confirmatory = load_yaml(config["confirmatory_config"])
    root = Path(confirmatory["output_root"]) / "space_time_holdout"
    rows = []
    candidates = [
        "seasonal_mean_run42", "ridge_run42", "hgb_run42",
        *[f"hgb_mlp_run{seed}" for seed in [11, 23, 42, 71, 101]],
        *[f"rbf_mlp_run{seed}" for seed in [11, 23, 42, 71, 101]],
    ]
    for run_name in tqdm(candidates, desc="ESS internal model error budgets"):
        path = root / run_name / "predictions.parquet"
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        valid = frame["evaluation_mask"].astype(bool)
        for group_name, group_value, selected in [
            ("overall", "overall", valid),
            *[("pressure_hpa", str(level), valid & (frame["pressure_hpa"] == level)) for level in sorted(frame["pressure_hpa"].unique())],
        ]:
            r_d = frame.loc[selected, "residual_dry"].to_numpy(float)
            r_w = frame.loc[selected, "residual_wet"].to_numpy(float)
            c_d = frame.loc[selected, "prediction_dry"].to_numpy(float)
            c_w = frame.loc[selected, "prediction_wet"].to_numpy(float)
            base_cross = 2.0 * np.mean(r_d * r_w)
            corrected_cross = 2.0 * np.mean((r_d - c_d) * (r_w - c_w))
            dry_delta = np.mean(np.square(r_d - c_d)) - np.mean(np.square(r_d))
            wet_delta = np.mean(np.square(r_w - c_w)) - np.mean(np.square(r_w))
            total_delta = np.mean(np.square((r_d + r_w) - (c_d + c_w))) - np.mean(np.square(r_d + r_w))
            rows.append({
                "run": run_name, "model": run_name.split("_run")[0], "grouping": group_name,
                "group": group_value, "n": int(selected.sum()), "dry_mse_delta": float(dry_delta),
                "wet_mse_delta": float(wet_delta), "cross_term_delta": float(corrected_cross - base_cross),
                "total_mse_delta": float(total_delta),
                "closure_error": float(total_delta - dry_delta - wet_delta - (corrected_cross - base_cross)),
            })
    return pd.DataFrame(rows)


def _probability_tables(sources: dict[str, pd.DataFrame], confidence_levels: list[float], pit_bins: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    diagnostics, coverage, histograms = [], [], []
    for source, frame in sources.items():
        valid = frame.loc[frame["evaluation_mask"]].copy()
        if source == "cosmic2":
            error = valid["observed_n"].to_numpy(float) - valid["hgb_prediction_n"].to_numpy(float)
            std = valid["hgb_prediction_std"].to_numpy(float)
            groups = [("overall", "overall", valid), *[("height_m", str(key), part) for key, part in valid.groupby("height_m", observed=True)], *[("macro_region", str(key), part) for key, part in valid.groupby("macro_region", observed=True)]]
        else:
            error = valid["residual_total"].to_numpy(float) - valid["hgb_correction_total"].to_numpy(float)
            std = valid["hgb_prediction_std"].to_numpy(float)
            vertical = "pressure_hpa"
            groups = [("overall", "overall", valid), *[(vertical, str(key), part) for key, part in valid.groupby(vertical, observed=True)], *[("macro_region", str(key), part) for key, part in valid.groupby("macro_region", observed=True)]]
        overall = probability_diagnostics(error, std)
        diagnostics.append({"source": source, "grouping": "overall", "group": "overall", **overall})
        hist = pit_histogram(error, std, pit_bins)
        hist.insert(0, "source", source)
        histograms.append(hist)
        for grouping, group, part in groups[1:]:
            if source == "cosmic2":
                group_error = part["observed_n"].to_numpy(float) - part["hgb_prediction_n"].to_numpy(float)
            else:
                group_error = part["residual_total"].to_numpy(float) - part["hgb_correction_total"].to_numpy(float)
            group_std = part["hgb_prediction_std"].to_numpy(float)
            diagnostics.append({"source": source, "grouping": grouping, "group": group, **probability_diagnostics(group_error, group_std)})
        for nominal in confidence_levels:
            z_value = float(norm.ppf((1.0 + float(nominal)) / 2.0))
            coverage.append({
                "source": source, "nominal_coverage": nominal,
                "empirical_coverage": float((np.abs(error / std) <= z_value).mean()), "n": int(len(error)),
                "coverage_error": float((np.abs(error / std) <= z_value).mean() - float(nominal)),
            })
    return pd.DataFrame(diagnostics), pd.DataFrame(coverage), pd.concat(histograms, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose correction and probability transfer for the ESS evidence chain.")
    parser.add_argument("--config", default="configs/revision2/ess.yaml")
    parser.add_argument("--cosmic-root")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = load_yaml(args.config)
    output = Path(args.output_dir or config["outputs"]["diagnostics_root"])
    output.mkdir(parents=True, exist_ok=True)
    external_root = Path(config["outputs"]["external_root"])
    rapsodi_path = external_root / "rapsodi_model_suite" / "predictions.parquet"
    if not rapsodi_path.is_file():
        raise FileNotFoundError("Run evaluate_ess_external_suite.py --source rapsodi first")
    cosmic_root = Path(args.cosmic_root or external_root / "cosmic2_height_model_suite")
    rapsodi = _add_source_residual(pd.read_parquet(rapsodi_path), "rapsodi")
    cosmic2 = _add_source_residual(_cosmic_frame(cosmic_root), "cosmic2")
    internal = _add_source_residual(_internal_suite(config), "igra")
    decomposition = pd.concat([
        _decomposition_table(internal, "igra", None), _decomposition_table(rapsodi, "rapsodi", None),
        _decomposition_table(cosmic2, "cosmic2", None),
    ], ignore_index=True)
    decomposition.to_csv(output / "correction_mse_decomposition.csv", index=False)
    pd.concat([
        _decomposition_table(internal, "igra", "pressure_hpa"),
        _decomposition_table(rapsodi, "rapsodi", "pressure_hpa"),
        _decomposition_table(cosmic2, "cosmic2", "height_m"),
    ], ignore_index=True).to_csv(output / "correction_alignment_by_vertical_coordinate.csv", index=False)
    pd.concat([
        _decomposition_table(internal, "igra", "macro_region"),
        _decomposition_table(rapsodi, "rapsodi", "macro_region"),
        _decomposition_table(cosmic2, "cosmic2", "macro_region"),
        _decomposition_table(cosmic2, "cosmic2", "land_sea"),
    ], ignore_index=True).to_csv(output / "correction_alignment_by_region.csv", index=False)
    pd.concat([
        _decomposition_table(internal, "igra", "wet_regime"),
        _decomposition_table(rapsodi, "rapsodi", "wet_regime"),
        _decomposition_table(cosmic2, "cosmic2", "wet_regime"),
    ], ignore_index=True).to_csv(output / "correction_alignment_by_wet_regime.csv", index=False)
    budget = _physical_budget(config)
    tolerance = float(config["statistics"]["decomposition_tolerance"])
    if not budget.empty and float(budget["closure_error"].abs().max()) > tolerance:
        raise ValueError("Internal dry/wet/cross physical budget does not close")
    budget.to_csv(output / "model_error_budget_comparison.csv", index=False)
    replicates = int(config["statistics"]["bootstrap_replicates"])
    seed = int(config["statistics"]["seed"])
    pd.concat([
        _bootstrap_table(internal, "igra", replicates, seed),
        _bootstrap_table(rapsodi, "rapsodi", replicates, seed),
        _bootstrap_table(cosmic2, "cosmic2", replicates, seed),
    ], ignore_index=True).to_csv(output / "transfer_mechanism_bootstrap.csv", index=False)
    # The internal probability frame comes from the frozen structured HGB run,
    # rather than the deterministic HGB table above.
    prob_run = Path(config["frozen_runs"]["hgb_probability"]["path"]) / "predictions.parquet"
    igra_probability = pd.read_parquet(prob_run).rename(columns={
        "prediction_total": "hgb_correction_total", "std_total": "hgb_prediction_std",
    })
    igra_probability["macro_region"] = np.where(
        igra_probability["latitude"] < -30, "southern_extratropics",
        np.where(igra_probability["latitude"] > 30, "northern_extratropics", "tropics"),
    )
    diagnostics, coverage, histograms = _probability_tables(
        {"igra": igra_probability, "rapsodi": rapsodi, "cosmic2": cosmic2},
        list(map(float, config["statistics"]["confidence_levels"])),
        int(config["statistics"]["pit_bins"]),
    )
    diagnostics.to_csv(output / "probability_transfer_diagnostics.csv", index=False)
    coverage.to_csv(output / "probability_coverage_transfer.csv", index=False)
    histograms.to_csv(output / "pit_histograms.csv", index=False)
    valid_decomposition = decomposition.loc[decomposition["comparison_valid"].astype(bool)]
    invalid_decomposition = decomposition.loc[~decomposition["comparison_valid"].astype(bool)]
    write_json(output / "ess_diagnostic_manifest.json", {
        "complete": True, "config": str(Path(args.config).resolve()), "cosmic_root": str(cosmic_root.resolve()),
        "inputs": {
            "rapsodi": {"path": str(rapsodi_path.resolve()), "sha256": sha256_file(rapsodi_path)},
            "cosmic_manifest": json.loads((cosmic_root / "manifest.json").read_text(encoding="utf-8")),
            "igra_probability": {"path": str(prob_run.resolve()), "sha256": sha256_file(prob_run)},
        },
        "decomposition_max_abs_closure": float(valid_decomposition["closure_error"].abs().max()),
        "decomposition_raw_max_abs_closure": float(decomposition["closure_error"].abs().max()),
        "decomposition_invalid_comparisons_excluded_from_integrity_gate": int(len(invalid_decomposition)),
        "decomposition_integrity_scope": "comparison_valid == true; raw out-of-support Ridge diagnostics remain in CSV and manifest",
        "physical_budget_max_abs_closure": float(budget["closure_error"].abs().max()),
        "external_data_used_for_training_selection_or_calibration": False,
        "interpretation": "mechanism localization and scale/distribution transfer diagnostics; not causal attribution or external recalibration",
    })
    print(f"ESS diagnostics: {output.resolve()}")


if __name__ == "__main__":
    main()
