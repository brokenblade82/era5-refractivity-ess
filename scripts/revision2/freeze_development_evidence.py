from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from _bootstrap import PROJECT_ROOT  # noqa: F401
from igra_forecast.revision2_data import sha256_file, stable_fingerprint, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the already-inspected 2024 station-disjoint evidence as development evidence.")
    parser.add_argument("--config", default="configs/revision2/confirmatory.yaml")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    documentation = Path(config["documentation_root"])
    documentation.mkdir(parents=True, exist_ok=True)
    audit_path = Path("data/revision2/manifests/revision2_data_audit.json")
    split_path = Path(config["station_split"])
    license_path = Path(config["protocol_license_matrix"])
    development = Path(config["development_results"])
    gate_path = development / "summary" / "model_selection_gate.csv"
    required = [audit_path, split_path, license_path, gate_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Development evidence is incomplete: {missing}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected = {"stations": 630, "profiles": 767337, "rows": 4604022}
    mismatches = {key: [audit.get(key), value] for key, value in expected.items() if int(audit.get(key, -1)) != value}
    if not audit.get("passed") or mismatches:
        raise ValueError(f"Revision-2 data audit does not match the frozen evidence: {mismatches}")
    split = pd.read_csv(split_path, dtype={"station_id": str})
    split["evidence_role"] = split["split"].replace({"test": "development_station_disjoint"})
    split_output = documentation / "development_station_roles.csv"
    split.to_csv(split_output, index=False)
    key_results = sorted(path for path in development.rglob("*.csv") if path.is_file())
    payload = {
        "status": "frozen_development_evidence",
        "interpretation": "The 2024 station-disjoint results informed the model-family decision and are not the untouched primary confirmation set.",
        "primary_confirmatory_protocol": config["interpretation"]["primary_confirmatory_protocol"],
        "dataset_version": audit["dataset_version"],
        "data_counts": expected,
        "station_roles": split["split"].value_counts().sort_index().to_dict(),
        "config": str(Path(args.config).resolve()),
        "config_fingerprint": stable_fingerprint(config),
        "files": {
            str(path.resolve()): sha256_file(path)
            for path in [audit_path, split_path, license_path, gate_path, *key_results]
        },
        "development_station_roles": str(split_output.resolve()),
    }
    output = documentation / "development_evidence_manifest.json"
    write_json(output, payload)
    print(f"Frozen development evidence: {output.resolve()}")


if __name__ == "__main__":
    main()
