from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from _bootstrap import PROJECT_ROOT  # noqa: F401


def is_complete(path: Path, model: str, period: str) -> bool:
    manifest = path / "protocol_manifest.json"
    if not manifest.is_file():
        return False
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(value.get("complete") and value.get("model") == model and value.get("block_evaluation_period") == period)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the non-neural Revision 2.1 30x60-degree spatial pressure test.")
    parser.add_argument("--config", default="configs/revision2/confirmatory.yaml")
    parser.add_argument("--block-manifest", default="docs/revision2_confirmatory/spatial_block_manifest.csv")
    parser.add_argument("--period", choices=["concurrent", "future", "all"], default="all")
    parser.add_argument("--fold", help="Optional single block such as lat3_lon1.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    blocks = pd.read_csv(args.block_manifest)
    if args.fold:
        blocks = blocks.loc[blocks["block"] == args.fold]
        if blocks.empty:
            raise ValueError(f"Block is not in the frozen manifest: {args.fold}")
    periods = list(config["spatial_blocks"]["periods"]) if args.period == "all" else [args.period]
    seeds = list(map(int, config["evaluation"]["seeds"]))
    primary = str(config["evaluation"]["primary_probability_variant"])
    experiments = [("era5", 42), ("hgb", 42), *[(primary, seed) for seed in seeds]]
    tasks = [(period, row.block, model, seed) for period in periods for row in blocks.itertuples() for model, seed in experiments]
    progress = tqdm(tasks, desc="Revision2.1 spatial blocks")
    for period, block, model, seed in progress:
        progress.set_postfix(period=period, block=block, model=model, run=seed)
        root = Path(config["output_root"]) / "spatial_leave_block_out" / period / block
        output = root / f"{model}_run{seed}"
        if is_complete(output, model, period):
            tqdm.write(f"[SKIP] {period} {block} {model} run={seed}")
      continue
        command = [
            sys.executable, str(Path(__file__).with_name("run_revision2_baselines.py")),
            "--config", args.config, "--model", model, "--protocol", "spatial_leave_block_out",
            "--block", block, "--block-evaluation-period", period,
            "--seed", str(seed), "--device", args.device, "--output-dir", str(output),
        ]
        if model == primary:
            mean_model = root / "hgb_run42" / "model.joblib"
            if not mean_model.is_file():
                raise FileNotFoundError(f"Frozen block HGB mean is missing: {mean_model}")
            command.extend(["--mean-model", str(mean_model)])
        subprocess.run(command, check=True)
    print("All requested spatial-block runs completed.")


if __name__ == "__main__":
    main()
