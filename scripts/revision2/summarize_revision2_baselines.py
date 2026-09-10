from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from pandas.errors import EmptyDataError


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def read_optional_csv(path: str | Path) -> pd.DataFrame:
 """Read an optional table; deterministic baselines intentionally emit an empty CSV."""
    path = Path(path)
    if not path.is_file() or not path.read_text(encoding="utf-8-sig").strip():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate completed revision-2 baseline runs and apply the preregistered neural gate.")
    parser.add_argument("--config", default="configs/revision2/baselines.yaml")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--input-root")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = load_yaml(args.config)
    root = Path(args.input_root or Path(config["output_root"]) / args.protocol)
    output = Path(args.output_dir or root / "summary")
    output.mkdir(parents=True, exist_ok=True)
    rows, costs = [], []
    for manifest_path in sorted(root.glob("*/protocol_manifest.json")):
        manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("complete") or manifest.get("smoke"):
            continue
        run = manifest_path.parent
        point = pd.read_csv(run / "point_metrics.csv")
        probability = read_optional_csv(run / "probabilistic_metrics.csv")
        total = point.loc[point["component"] == "total"].iloc[0].to_dict()
        row = {"model": manifest["model"], "seed": int(manifest["seed"]), "run_dir": str(run.resolve()), **{f"point_{key}": value for key, value in total.items() if key != "component"}}
        if not probability.empty and (probability["component"] == "total").any():
            prob = probability.loc[probability["component"] == "total"].iloc[0].to_dict()
            row.update({f"prob_{key}": value for key, value in prob.items() if key not in {"component", "n"}})
        rows.append(row)
        cost = pd.read_csv(run / "compute_cost.csv")
        costs.append(cost)
    if not rows:
        raise FileNotFoundError(f"No completed formal runs found under {root}")
    runs = pd.DataFrame(rows).sort_values(["model", "seed"])
    runs.to_csv(output / "run_metrics.csv", index=False)
    numeric = [column for column in runs.select_dtypes("number") if column != "seed"]
    summary = runs.groupby("model")[numeric].agg(["mean", "std", "count"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(output / "repeated_run_summary.csv", index=False)
    if costs:
        pd.concat(costs, ignore_index=True).to_csv(output / "compute_cost.csv", index=False)

    rule = config["evaluation"]["selection_rule"]
    metric = f"prob_{rule['primary_probability_metric']}"
    gate = {"protocol": args.protocol, "primary_probability_metric": rule["primary_probability_metric"], "required_winning_runs": int(rule["minimum_winning_runs"]), "rmse_noninferiority_margin_n_units": float(rule["rmse_noninferiority_margin_n_units"]), "paired_runs": 0, "probability_wins": 0, "rmse_noninferior": False, "passes": False}
    if metric in runs and {"probabilistic_hgb", "hgb_mlp"}.issubset(set(runs["model"])):
        common = sorted(set(runs.loc[runs["model"] == "probabilistic_hgb", "seed"]) & set(runs.loc[runs["model"] == "hgb_mlp", "seed"]))
        ph = runs.set_index(["model", "seed"])
        probability_delta = [float(ph.loc[("hgb_mlp", seed), metric] - ph.loc[("probabilistic_hgb", seed), metric]) for seed in common]
        rmse_delta = [float(ph.loc[("hgb_mlp", seed), "point_rmse"] - ph.loc[("probabilistic_hgb", seed), "point_rmse"]) for seed in common]
        gate.update({
            "paired_runs": len(common), "probability_wins": int(sum(value < 0 for value in probability_delta)),
            "mean_probability_delta": sum(probability_delta) / len(probability_delta) if probability_delta else None,
            "mean_rmse_delta": sum(rmse_delta) / len(rmse_delta) if rmse_delta else None,
            "rmse_noninferior": bool(rmse_delta and (sum(rmse_delta) / len(rmse_delta)) <= float(rule["rmse_noninferiority_margin_n_units"])),
        })
        gate["passes"] = bool(gate["probability_wins"] >= gate["required_winning_runs"] and gate["rmse_noninferior"])
    pd.DataFrame([gate]).to_csv(output / "model_selection_gate.csv", index=False)
    print(pd.DataFrame([gate]).to_string(index=False))
    print(f"Summary: {output.resolve()}")


if __name__ == "__main__":
    main()
