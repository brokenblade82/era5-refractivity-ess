from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from _bootstrap import PROJECT_ROOT  # noqa: F401
from igra_forecast.revision2_data import write_json


def read_single(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    if len(table) != 1:
        raise ValueError(f"Expected one gate row: {path}")
    return table.iloc[0].to_dict()


def as_bool(value: object) -> bool:
    if isinstance(value, (bool,)):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine frozen point, probability, external, and application gates without changing the model.")
    parser.add_argument("--config", default="configs/revision2/confirmatory.yaml")
    parser.add_argument("--rapsodi-gate", default="paper_outputs/revision2_confirmatory/external/rapsodi/external_gate.csv")
    parser.add_argument("--cosmic2-gate", default="paper_outputs/revision2_confirmatory/external/cosmic2/external_gate.csv")
    parser.add_argument("--output", default="paper_outputs/revision2_confirmatory/final_gate_manifest.json")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    protocol = config["interpretation"]["primary_confirmatory_protocol"]
    summary = Path(config["output_root"]) / protocol / "summary"
    point = read_single(summary / "point_gate.csv")
    probability = read_single(summary / "probability_gate.csv")
    rapsodi = read_single(Path(args.rapsodi_gate))
    cosmic2 = read_single(Path(args.cosmic2_gate))
    point_pass = as_bool(point["passes"])
    rapsodi_pass = as_bool(rapsodi["passes_no_material_degradation"])
    cosmic_pass = as_bool(cosmic2["passes_no_material_degradation"])
    payload = {
        "primary_confirmatory_protocol": protocol,
        "point_gate": point,
        "probability_gate": probability,
        "rapsodi_gate": rapsodi,
        "cosmic2_gate": cosmic2,
        "deployment_point_product_allowed": bool(point_pass and rapsodi_pass and cosmic_pass),
        "structured_probability_product_allowed": bool(point_pass and rapsodi_pass and cosmic_pass and as_bool(probability["passes"])),
        "structured_probability_product_allowed_under_preregistered_internal_gate": bool(
            point_pass and rapsodi_pass and cosmic_pass and as_bool(probability["passes"])
        ),
        "external_probability_transfer_diagnostics": {
            "rapsodi_marginal_coverage_90": rapsodi.get("external_marginal_coverage_90"),
            "cosmic2_marginal_coverage_90": cosmic2.get("external_marginal_coverage_90"),
            "is_preregistered_gate": False,
            "reporting_requirement": "Report external coverage directly; internal calibration does not establish external-domain calibration.",
        },
        "failure_policy": {
            "point_or_external_failure": "Do not generate a global correction product; reposition as ERA5 refractivity error diagnosis and validation.",
            "probability_failure_only": "Retain the HGB point result if supported, but do not claim structured predictive uncertainty as validated.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, payload)
    print(pd.DataFrame([{
        "point": point_pass, "probability": as_bool(probability["passes"]), "rapsodi": rapsodi_pass,
        "cosmic2": cosmic_pass, "deployment_allowed": payload["deployment_point_product_allowed"],
    }]).to_string(index=False))
    print(f"Final gates: {output.resolve()}")


if __name__ == "__main__":
    main()
