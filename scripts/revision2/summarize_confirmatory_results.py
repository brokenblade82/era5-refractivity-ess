from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from _bootstrap import PROJECT_ROOT  # noqa: F401
from summarize_revision2_baselines import read_optional_csv
from igra_forecast.revision2_data import write_json


def total_row(path: Path, filename: str, component: str = "total") -> dict[str, float]:
    table = read_optional_csv(path / filename)
    if table.empty:
        return {}
    if "component" in table:
        selected = table.loc[table["component"].astype(str) == component]
        if not selected.empty:
            table = selected
    return {str(key): value for key, value in table.iloc[0].items()}


def run_record(run: Path) -> dict[str, object] | None:
    manifest_path = run / "protocol_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("smoke"):
        return None
    record: dict[str, object] = {
        "model": manifest["model"], "seed": int(manifest["seed"]), "run_dir": str(run.resolve()),
        "protocol": manifest["protocol"], "block": manifest.get("block"),
        "block_evaluation_period": manifest.get("block_evaluation_period"),
    }
    for prefix, filename, component in [
        ("point", "point_metrics.csv", "total"),
        ("prob", "probabilistic_metrics.csv", "total"),
        ("joint", "joint_probability_metrics.csv", "total_profile"),
        ("station", "station_bootstrap.csv", ""),
        ("month", "month_bootstrap.csv", ""),
        ("hierarchical", "hierarchical_bootstrap.csv", ""),
    ]:
        values = total_row(run, filename, component)
        record.update({f"{prefix}_{key}": value for key, value in values.items()})
    coverage = read_optional_csv(run / "coverage_curve.csv")
    if not coverage.empty:
        selected = coverage.loc[(coverage["component"] == "total") & np.isclose(coverage["nominal_coverage"], 0.90)]
        if not selected.empty:
            record["marginal_coverage_90"] = float(selected.iloc[0]["empirical_coverage"])
    return record


def repeated_summary(runs: pd.DataFrame) -> pd.DataFrame:
    numeric = [column for column in runs.select_dtypes(include=[np.number]).columns if column != "seed"]
    rows = []
    for model, group in runs.groupby("model", sort=True):
        row: dict[str, object] = {"model": model, "runs": len(group)}
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if len(values):
                row[f"{column}_mean"] = float(values.mean())
                row[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_probability_comparisons(runs: pd.DataFrame, primary: str) -> pd.DataFrame:
    """Return paired run deltas for every probability baseline versus the primary model."""
    candidates = sorted(model for model in runs["model"].unique() if str(model).startswith("hgb_prob_"))
    indexed = runs.set_index(["model", "seed"])
    rows: list[dict[str, object]] = []
    for comparator in candidates:
        if comparator == primary:
            continue
        common = sorted(
            set(runs.loc[runs["model"] == primary, "seed"])
            & set(runs.loc[runs["model"] == comparator, "seed"])
        )
        for seed in common:
            primary_row = indexed.loc[(primary, seed)]
            comparator_row = indexed.loc[(comparator, seed)]
            rows.append({
                "primary_model": primary,
                "comparator_model": comparator,
                "seed": int(seed),
                "crps_delta_primary_minus_comparator": float(primary_row["prob_crps"] - comparator_row["prob_crps"]),
                "energy_delta_primary_minus_comparator": float(primary_row["joint_profile_energy_score"] - comparator_row["joint_profile_energy_score"]),
                "nll_delta_primary_minus_comparator": float(primary_row["joint_multivariate_gaussian_nll_per_valid_level"] - comparator_row["joint_multivariate_gaussian_nll_per_valid_level"]),
                "coverage_90_delta_primary_minus_comparator": float(primary_row["marginal_coverage_90"] - comparator_row["marginal_coverage_90"]),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize one frozen Revision 2.1 protocol and evaluate preregistered gates.")
    parser.add_argument("--config", default="configs/revision2/confirmatory.yaml")
    parser.add_argument("--protocol", required=True, choices=["spatial_station_disjoint", "temporal_holdout", "space_time_holdout"])
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    root = Path(config["output_root"]) / args.protocol
    records = [record for run in sorted(root.iterdir()) if run.is_dir() if (record := run_record(run)) is not None]
    if not records:
        raise FileNotFoundError(f"No complete formal runs found below {root}")
    runs = pd.DataFrame(records)
    output = root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output / "run_metrics.csv", index=False)
    repeated_summary(runs).to_csv(output / "repeated_run_summary.csv", index=False)

    evaluation = config["evaluation"]
    primary = str(evaluation["primary_probability_variant"])
    reference = str(evaluation["probability_reference_variant"])
    required = int(evaluation["selection_rule"]["minimum_winning_runs"])
    pairwise_probability_comparisons(runs, primary).to_csv(
        output / "probability_pairwise_comparison.csv", index=False
    )
    common = sorted(set(runs.loc[runs["model"] == primary, "seed"]) & set(runs.loc[runs["model"] == reference, "seed"]))
    crps_delta = []
    energy_delta = []
    if common:
        indexed = runs.set_index(["model", "seed"])
        for seed in common:
            crps_delta.append(float(indexed.loc[(primary, seed), "prob_crps"] - indexed.loc[(reference, seed), "prob_crps"]))
            energy_delta.append(float(indexed.loc[(primary, seed), "joint_profile_energy_score"] - indexed.loc[(reference, seed), "joint_profile_energy_score"]))
    primary_rows = runs.loc[runs["model"] == primary]
    coverage_min = float(evaluation["selection_rule"]["marginal_coverage_90_min"])
    coverage_max = float(evaluation["selection_rule"]["marginal_coverage_90_max"])
    coverage_mean = float(primary_rows["marginal_coverage_90"].mean()) if not primary_rows.empty else np.nan
    probability_gate = {
        "protocol": args.protocol,
        "primary_variant": primary,
        "reference_variant": reference,
        "paired_runs": len(common),
        "required_winning_runs": required,
        "crps_wins": int(sum(value < 0 for value in crps_delta)),
        "energy_score_wins": int(sum(value < 0 for value in energy_delta)),
        "mean_crps_delta": float(np.mean(crps_delta)) if crps_delta else np.nan,
        "mean_energy_score_delta": float(np.mean(energy_delta)) if energy_delta else np.nan,
        "mean_marginal_coverage_90": coverage_mean,
        "coverage_gate": bool(coverage_min <= coverage_mean <= coverage_max),
    }
    probability_gate["passes"] = bool(
        probability_gate["paired_runs"] == len(evaluation["seeds"])
        and probability_gate["crps_wins"] >= required
        and probability_gate["energy_score_wins"] >= required
        and probability_gate["coverage_gate"]
    )
    pd.DataFrame([probability_gate]).to_csv(output / "probability_gate.csv", index=False)

    hgb = runs.loc[runs["model"] == "hgb"]
    point_gate = {"protocol": args.protocol, "primary_confirmation": args.protocol == config["interpretation"]["primary_confirmatory_protocol"], "passes": False}
    if len(hgb) == 1:
        row = hgb.iloc[0]
        point_gate.update({
            "station_mean_rmse_difference": float(row.get("station_mean_difference", np.nan)),
            "station_ci_lower": float(row.get("station_ci_lower", np.nan)),
            "station_ci_upper": float(row.get("station_ci_upper", np.nan)),
            "passes": bool(float(row.get("station_ci_upper", np.inf)) < 0),
        })
    pd.DataFrame([point_gate]).to_csv(output / "point_gate.csv", index=False)
    write_json(output / "summary_manifest.json", {
        "protocol": args.protocol, "runs": len(runs), "root": str(root.resolve()),
        "interpretation": "primary confirmation" if point_gate["primary_confirmation"] else "supporting/development evidence",
        "probability_gate": probability_gate, "point_gate": point_gate,
    })
    print(pd.DataFrame([point_gate]).to_string(index=False))
    print(pd.DataFrame([probability_gate]).to_string(index=False))
    print(f"Summary: {output.resolve()}")


if __name__ == "__main__":
    main()
