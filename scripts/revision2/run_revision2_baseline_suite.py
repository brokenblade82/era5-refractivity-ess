from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the preregistered revision-2 baseline suite with resumable progress.")
    parser.add_argument("--config", default="configs/revision2/baselines.yaml")
    parser.add_argument("--protocol", required=True, choices=["spatial_station_disjoint", "temporal_holdout", "space_time_holdout", "spatial_leave_block_out"])
    parser.add_argument("--block")
    parser.add_argument("--block-evaluation-period", choices=["concurrent", "future"], default="concurrent")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--include-rbf", action="store_true")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    seeds = list(map(int, config["evaluation"]["seeds"]))
    experiments = [(name, 42) for name in ["era5", "seasonal_mean", "ridge", "hgb"]]
    experiments.extend(("probabilistic_hgb", seed) for seed in seeds)
    experiments.extend(("hgb_mlp", seed) for seed in seeds)
    if args.include_rbf:
        experiments.extend(("rbf_mlp", seed) for seed in seeds)
    suffix = f"_{args.block}_{args.block_evaluation_period}" if args.block else ""
    protocol_root = Path(config["output_root"]) / f"{args.protocol}{suffix}"
    progress = tqdm(experiments, desc=f"Revision2 {args.protocol}")
    for model, seed in progress:
        progress.set_postfix(model=model, seed=seed)
        output = protocol_root / f"{model}_run{seed}"
        manifest = output / "protocol_manifest.json"
        if manifest.is_file():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if payload.get("complete") and not payload.get("smoke"):
                tqdm.write(f"[SKIP] complete: {model}, run={seed}")
                continue
        command = [
            sys.executable, str(Path(__file__).with_name("run_revision2_baselines.py")),
            "--config", args.config, "--model", model, "--protocol", args.protocol,
            "--seed", str(seed), "--device", args.device, "--output-dir", str(output),
        ]
        if model in {"hgb_mlp", "rbf_mlp"}:
            command.append("--resume")
        if args.epochs is not None:
            command.extend(["--epochs", str(args.epochs)])
        if args.block:
            command.extend(["--block", args.block, "--block-evaluation-period", args.block_evaluation_period])
        subprocess.run(command, check=True)
    print(f"Completed suite: {protocol_root.resolve()}")


if __name__ == "__main__":
    main()
