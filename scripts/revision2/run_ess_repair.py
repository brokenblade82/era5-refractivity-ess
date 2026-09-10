"""Non-training ESS repairs. Run from the activated torch environment."""
from __future__ import annotations

import argparse
import os

from _bootstrap import PROJECT_ROOT
from igra_forecast.revision2_ess_repair_pipeline import RepairPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/revision2/ess_repair.yaml")
    parser.add_argument("--stage", required=True, choices=["freeze", "fit-ridge", "cache-cosmic", "evaluate", "bootstrap", "diagnose", "summarize", "validate"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-profiles", type=int)
    parser.add_argument("--bootstrap-replicates", type=int)
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    pipeline = RepairPipeline(PROJECT_ROOT, args)
    pipeline.run(args.stage)


if __name__ == "__main__":
    main()
