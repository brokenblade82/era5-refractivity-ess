"""Read-only correction of ESS mask accounting; never invokes the frozen pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
KEYS = ["profile_id", "time", "height_m", "source_archive"]
MASKS = ["height_mask", "evaluation_mask", "common_mask", "below_surface",
         "within_era5_height_support", "within_correction_height_support", "nonfinite_probability"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def boolean_mask(series):
    if series.isna().any() or not series.isin([0, 1, False, True]).all():
        raise ValueError(f"Mask {series.name} must contain only nonmissing 0/1 or booleans")
    return series.astype(bool)


def audit_block(frame):
    """Count on boolean masks and independently check the selection predicate."""
    frame = frame.copy()
    for name in MASKS:
        frame[name] = boolean_mask(frame[name])
    if set(frame.variant) != {"A", "B", "C"} or frame.duplicated(["variant", *KEYS]).any():
        raise ValueError("Expected unique matched A/B/C target keys")
    groups = {v: g.set_index(KEYS).sort_index() for v, g in frame.groupby("variant")}
    a = groups["A"]
    for group in groups.values():
        if not group.index.equals(a.index):
            raise ValueError("Coordinate variants do not share input targets")
    common = np.logical_and.reduce([g.evaluation_mask.to_numpy() for g in groups.values()])
    rows = []
    for v, group in groups.items():
        expected = (group.height_mask & np.isfinite(group.observed_n) & ~group.below_surface
                    & group.within_era5_height_support & group.within_correction_height_support
                    & np.isfinite(group.era5_n_height) & np.isfinite(group.hgb_correction_total)
                    & ~group.nonfinite_probability)
        if not expected.equals(group.evaluation_mask):
            raise ValueError(f"Actual evaluation selection differs from physical predicate: {v}")
        if not np.array_equal(group.common_mask, common):
            raise ValueError("Stored common mask is not the A/B/C intersection")
        current = group.evaluation_mask
        added, removed = ~a.evaluation_mask & current, a.evaluation_mask & ~current
        row = {"variant": v, "n_input": len(group), "n_valid": int(current.sum()),
               "n_common": int(common.sum()), "input_masked": int((~group.height_mask).sum()),
               "added_vs_A": int(added.sum()), "removed_vs_A": int(removed.sum()),
               "below_surface": int(group.below_surface.sum()),
               "outside_era5_support": int((~group.within_era5_height_support).sum()),
               "outside_correction_support": int((~group.within_correction_height_support).sum()),
               "nonfinite_probability": int(group.nonfinite_probability.sum()),
               "removed_below_surface": int((removed & group.below_surface).sum()),
               "removed_outside_era5_support": int((removed & ~group.within_era5_height_support).sum()),
               "removed_outside_correction_support": int((removed & ~group.within_correction_height_support).sum()),
               "removed_nonfinite_probability": int((removed & group.nonfinite_probability).sum())}
        if any(not 0 <= value <= len(group) for k, value in row.items() if k != "variant"):
            raise ValueError("Mask count outside [0, n_input]")
        if row["n_valid"] != int(a.evaluation_mask.sum()) + row["added_vs_A"] - row["removed_vs_A"]:
            raise ValueError("Added/removed sample accounting does not close")
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Only first two blocks; separate outputs")
    args = parser.parse_args()
    results = ROOT / "results_mapping/revision2_ess_repair"
    paper = ROOT / "paper_outputs/revision2_ess_repair"
    output = ROOT / "paper_outputs/ess_submission_audit"
    if args.smoke:
        output = output / "smoke" / sha(Path(__file__))[:12]
    stage_path = results / "stages/evaluate.json"
    stage = read_json(stage_path)
    validation_path = paper / "validation_report.json"
    validation = read_json(validation_path)
    if stage.get("smoke") or not stage.get("complete") or validation.get("status") != "repair_evidence_complete":
        raise ValueError("Requires completed formal repair results")
    old_path = paper / "statistics/mask_audit_summary.csv"
    old_sha = sha(old_path)
    old_record = [item for item in stage["artifacts"] if (ROOT / item["path"]) == old_path]
    if len(old_record) != 1 or old_record[0]["sha256"] != old_sha:
        raise ValueError("Original mask audit differs from its frozen stage hash")
    evidence = read_json(paper / "repair_evidence_manifest.json")
    if evidence["fingerprint"] != stage["fingerprint"] or evidence.get("smoke"):
        raise ValueError("Repair evidence and prediction stage use different versions")
    identity = {"source_fingerprint": stage["fingerprint"], "evaluate_manifest_sha256": sha(stage_path),
                "validation_sha256": sha(validation_path), "audit_code_sha256": sha(Path(__file__)),
                "smoke": args.smoke}
    files = [item for item in stage["artifacts"] if "evaluation/cosmic2/" in item["path"].replace("\\", "/")
             and item["path"].endswith(".parquet")]
    actual_files = set((results / "evaluation/cosmic2").rglob("*.parquet"))
    if actual_files != {ROOT / item["path"] for item in files} or not files:
        raise ValueError("Prediction partitions differ from the frozen stage manifest")
    if args.smoke:
        files = files[:2]
    output.mkdir(parents=True, exist_ok=True)
    frozen_path = output / "audit_inputs.json"
    if frozen_path.exists() and read_json(frozen_path) != identity:
        raise ValueError("Existing audit uses different code or inputs; do not mix audit versions")
    write_json(frozen_path, identity)
    rows = []
    columns = [*KEYS, *MASKS, "variant", "observed_n", "era5_n_height", "hgb_correction_total"]
    for i, item in enumerate(tqdm(files, desc="Verify and recount frozen COSMIC blocks")):
        source = ROOT / item["path"]
        if sha(source) != item["sha256"]:
            raise ValueError(f"Frozen prediction changed: {source}")
        part = output / "blocks" / f"block_{i:05d}.json"
        key = {**identity, "source": item}
        if args.resume and part.exists():
            stored = read_json(part)
            if stored["identity"] != key:
                raise ValueError("Audit block provenance mismatch")
            records = stored["records"]
        else:
            frame = pq.read_table(source, columns=columns).to_pandas()
            records = audit_block(frame).assign(block=item["path"]).to_dict("records")
            write_json(part, {"identity": key, "records": records})
        rows.extend(records)
    detail = pd.DataFrame(rows)
    summary = detail.groupby("variant").sum(numeric_only=True).reset_index()
    comparison = []
    old = pd.read_csv(old_path).query("source == 'cosmic2'").set_index("variant")
    for _, row in summary.iterrows():
        for name in summary.columns.drop("variant"):
            comparison.append({"variant": row.variant, "field": name, "original": old.loc[row.variant, name],
                "corrected": row[name], "difference": row[name]-old.loc[row.variant, name],
                "comparison_valid": not args.smoke})
    comparison = pd.DataFrame(comparison)
    if not args.smoke:
        unaffected = comparison.loc[comparison.field.ne("input_masked")]
        if not unaffected.difference.eq(0).all():
            raise ValueError("Unexpected changes outside input_masked: investigate before manuscript use")
        if not summary.n_input.eq(2176344).all() or not summary.input_masked.eq(121335).all():
            raise ValueError("Unexpected input sample counts")
    for name, frame in (("mask_audit_corrected_by_block.csv", detail), ("mask_audit_corrected_summary.csv", summary),
                        ("mask_audit_changes.csv", comparison)):
        write_csv(output / name, frame)
    if sha(stage_path) != identity["evaluate_manifest_sha256"] or sha(old_path) != old_sha:
        raise ValueError("Source audit changed during execution")
    artifacts = {p.name: sha(p) for p in output.glob("*.csv")}
    write_json(output / "audit_report.json", {**identity, "status": "smoke_passed" if args.smoke else "accounting_repaired",
        "partitions_checked": len(files), "artifacts": artifacts,
        "source_audit_sha256": old_sha, "evaluation_predicate_verified": True,
        "prediction_files_match_frozen_hashes": True, "performance_recomputed": False,
        "impact": "Only input_masked accounting corrected; original predictions and statistics retained",
        "submission_ready": False})
    print(f"Audit complete: {output / 'audit_report.json'}")


if __name__ == "__main__":
    main()
