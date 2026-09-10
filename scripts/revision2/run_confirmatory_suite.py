from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

from _bootstrap import PROJECT_ROOT  # noqa: F401
from igra_forecast.revision2_confirmatory import PROBABILITY_VARIANTS


def complete_manifest(path: Path, expected_model: str, expected_protocol: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        payload.get("complete")
        and not payload.get("smoke")
        and payload.get("model") == expected_model
        and payload.get("protocol") == expected_protocol
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen Revision 2.1 confirmatory comparison with visible progress.")
    parser.add_argument("--config", default="configs/revision2/confirmatory.yaml")
    parser.add_argument("--protocol", required=True, choices=["spatial_station_disjoint", "temporal_holdout", "space_time_holdout", "spatial_leave_block_out"])
    parser.add_argument("--block")
    parser.add_argument("--block-evaluation-period", choices=["concurrent", "future"], default="concurrent")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--include-neural-comparators", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-times", type=int)
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    seeds = list(map(int, config["evaluation"]["seeds"]))
    experiments: list[tuple[str, int]] = [(name, 42) for name in ["era5", "seasonal_mean", "ridge", "hgb"]]
    experiments.extend((variant, seed) for variant in PROBABILITY_VARIANTS for seed in seeds)
    if args.include_neural_comparators and args.protocol != "spatial_leave_block_out":
        experiments.extend((model, seed) for model in ["hgb_mlp", "rbf_mlp"] for seed in seeds)

    suffix = f"_{args.block}_{args.block_evaluation_period}" if args.block else ""
    protocol_root = Path(config["output_root"]) / ("smoke" if args.smoke else "") / f"{args.protocol}{suffix}"
    hgb_model = protocol_root / "hgb_run42" / "model.joblib"
    progress = tqdm(experiments, desc=f"Confirmatory {args.protocol}")
    for model, seed in progress:
        progress.set_postfix(model=model, run=seed)
        output = protocol_root / f"{model}_run{seed}"
        if complete_manifest(output / "protocol_manifest.json", model, args.protocol):
            tqdm.write(f"[SKIP] complete: {model}, run={seed}")
            continue
        command = [
            sys.executable,
            str(Path(__file__).with_name("run_revision2_baselines.py")),
            "--config", args.config,
            "--model", model,
            "--protocol", args.protocol,
            "--seed", str(seed),
            "--device", args.device,
            "--output-dir", str(output),
        ]
        if model in PROBABILITY_VARIANTS:
            if not hgb_model.is_file():
                raise FileNotFoundError(f"Frozen HGB mean must be completed first: {hgb_model}")
            command.extend(["--mean-model", str(hgb_model)])
        if model in {"hgb_mlp", "rbf_mlp"}:
            command.append("--resume")
        if args.epochs is not None:
            command.extend(["--epochs", str(args.epochs)])
        if args.block:
            command.extend(["--block", args.block, "--block-evaluation-period", args.block_evaluation_period])
        if args.smoke:
            command.append("--smoke")
        if args.max_times is not None:
            command.extend(["--max-times", str(args.max_times)])
        subprocess.run(command, check=True)
    print(f"Completed confirmatory suite: {protocol_root.resolve()}")


if __name__ == "__main__":
    main()
