from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

from _bootstrap import PROJECT_ROOT  # noqa: F401
from igra_forecast.revision2_data import ensure_data_directories, load_data_config, macro_region, require_collocation_manifest, write_json


def allocate_counts(n: int, fractions: list[float]) -> np.ndarray:
    raw = np.asarray(fractions, dtype=float) * n
    counts = np.floor(raw).astype(int)
    order = np.argsort(-(raw - counts))
    counts[order[: int(n - counts.sum())]] += 1
    if n >= len(fractions):
        for index in np.flatnonzero(counts == 0):
            donor = int(np.argmax(counts))
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[index] += 1
    return counts


def enforce_minimum_counts(names: list[str], counts: np.ndarray, minimums: dict[str, int]) -> np.ndarray:
    counts = counts.copy()
    index = {name: i for i, name in enumerate(names)}
    for name, minimum in minimums.items():
        target = index[name]
        while counts[target] < int(minimum):
            donors = [i for i, donor_name in enumerate(names) if i != target and counts[i] > int(minimums.get(donor_name, 0))]
            if not donors:
                raise ValueError(f"Cannot satisfy minimum station count for {name}: {minimum}")
            donor = max(donors, key=lambda i: (counts[i] - int(minimums.get(names[i], 0)), names[i] == "train"))
            counts[donor] -= 1
            counts[target] += 1
    return counts


def balanced_assignment(group: pd.DataFrame, names: list[str], counts: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Distribute roles over longitude sectors while exactly preserving quotas."""
    group = group.copy()
    group["_jitter"] = rng.random(len(group))
    group = group.sort_values(["longitude_sector", "_jitter", "station_id"]).reset_index(drop=True)
    remaining = counts.astype(int).copy()
    target = np.maximum(counts.astype(float), 1.0)
    labels: list[str] = []
    for _ in range(len(group)):
        score = np.where(remaining > 0, remaining / target, -np.inf)
        chosen = int(np.argmax(score))
        labels.append(names[chosen])
        remaining[chosen] -= 1
    if np.any(remaining != 0):
        raise AssertionError(f"Station split quota was not exhausted: {remaining.tolist()}")
    group["split"] = labels
    return group.drop(columns="_jitter")


def protocol_license_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(protocol: str, phase: str, role: str, start: str, end: str, note: str) -> None:
        rows.append({"protocol": protocol, "phase": phase, "station_license": role, "time_start_utc": start, "time_end_utc_exclusive": end, "note": note})

    add("spatial_station_disjoint", "train", "train", "2024-01-01", "2025-01-01", "model fitting")
    add("spatial_station_disjoint", "early_stopping", "early_val", "2024-01-01", "2025-01-01", "neural early stopping only")
    add("spatial_station_disjoint", "calibration", "calibration", "2024-01-01", "2025-01-01", "post-hoc predictive-scale calibration")
    add("spatial_station_disjoint", "evaluation", "test", "2024-01-01", "2025-01-01", "final station-disjoint evaluation")
    add("temporal_holdout", "train", "train", "2024-01-01", "2025-01-01", "model fitting")
    add("temporal_holdout", "early_stopping", "early_val", "2024-01-01", "2025-01-01", "neural early stopping only")
    add("temporal_holdout", "calibration", "train", "2025-01-01", "2025-04-01", "future-time predictive-scale calibration")
    add("temporal_holdout", "evaluation", "train", "2025-04-01", "2026-01-01", "future-time evaluation")
    add("space_time_holdout", "train", "train", "2024-01-01", "2025-01-01", "model fitting")
    add("space_time_holdout", "early_stopping", "early_val", "2024-01-01", "2025-01-01", "neural early stopping only")
    add("space_time_holdout", "calibration", "calibration", "2025-01-01", "2025-04-01", "future station-disjoint calibration")
    add("space_time_holdout", "evaluation", "test", "2025-04-01", "2026-01-01", "future station-disjoint evaluation")
    add("spatial_leave_block_out", "train", "block_out", "2024-01-01", "2025-01-01", "block-specific model fitting")
    add("spatial_leave_block_out", "early_stopping", "block_out_early_val", "2024-01-01", "2025-01-01", "block-out early stopping")
    add("spatial_leave_block_out", "calibration_concurrent", "block_out_calibration", "2024-01-01", "2025-01-01", "block-out concurrent-period calibration")
    add("spatial_leave_block_out", "calibration_future", "block_out_calibration", "2025-01-01", "2025-04-01", "block-out future-period calibration")
    add("spatial_leave_block_out", "evaluation_concurrent", "block_in", "2024-01-01", "2025-01-01", "held-out block concurrent evaluation")
    add("spatial_leave_block_out", "evaluation_future", "block_in", "2025-04-01", "2026-01-01", "held-out block future evaluation")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fixed station roles and protocol licenses for revision 2.")
    parser.add_argument("--config", default="configs/revision2/data_pipeline.yaml")
    parser.add_argument("--input", default="data/revision2/processed/era5_igra_profiles")
    parser.add_argument("--output", default="data/revision2/manifests/station_split.csv")
    parser.add_argument("--collocation-manifest", default="data/revision2/manifests/era5_igra_profile_manifest.json")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_data_config(args.config)
    ensure_data_directories(config)
    collocation = require_collocation_manifest(config, args.collocation_manifest, args.input, require_formal=True)
    dataset = pads.dataset(args.input, format="parquet", partitioning="hive")
    columns = ["station_id", "latitude", "longitude", "station_elevation_m", "time", "level_mask"]
    frame = dataset.to_table(columns=columns).to_pandas()
    station = frame.groupby("station_id", as_index=False).agg(
        latitude=("latitude", "first"), longitude=("longitude", "first"), station_elevation_m=("station_elevation_m", "first"),
        valid_level_rows=("level_mask", "sum"), first_time=("time", "min"), last_time=("time", "max"),
    )
    profiles = frame[["station_id", "time"]].drop_duplicates()
    station["profile_count"] = station["station_id"].map(profiles.groupby("station_id").size()).astype(int)
    station["macro_region"] = station["latitude"].map(macro_region)
    station["longitude_sector"] = np.floor(np.mod(station["longitude"], 360.0) / 60.0).astype(int)
    settings = config["splits"]
    seed = int(args.seed if args.seed is not None else settings["seed"])
    rng = np.random.default_rng(seed)
    names = list(settings["fractions"])
    fractions = [float(settings["fractions"][name]) for name in names]
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Station split fractions must sum to one")
    assignments = []
    southern_minimums = {key: int(value) for key, value in settings.get("minimum_southern_extratropical_stations", {}).items()}
    for region, group in station.groupby("macro_region", sort=True):
        counts = allocate_counts(len(group), fractions)
        if region == "southern_extratropics":
            counts = enforce_minimum_counts(names, counts, southern_minimums)
        assignments.append(balanced_assignment(group, names, counts, rng))
    result = pd.concat(assignments, ignore_index=True).sort_values("station_id")
    if result["station_id"].duplicated().any() or set(result["station_id"]) != set(station["station_id"]):
        raise AssertionError("Station assignment is not a one-to-one partition")
    counts = result.groupby(["split", "macro_region"])["station_id"].nunique().unstack(fill_value=0)
    southern = counts.get("southern_extratropics", pd.Series(dtype=int)).to_dict()
    deficient = {role: required for role, required in southern_minimums.items() if int(southern.get(role, 0)) < required}
    if deficient:
        raise ValueError(f"Southern-extratropical split minimums were not met: {deficient}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    profile_index = profiles.merge(result[["station_id", "split", "macro_region", "longitude_sector"]], on="station_id", how="left", validate="many_to_one")
    timestamp = pd.DatetimeIndex(pd.to_datetime(profile_index["time"], utc=True))
    profile_index["year"], profile_index["month"] = timestamp.year, timestamp.month
    profile_index["temporal_role"] = np.select(
        [timestamp < pd.Timestamp("2025-01-01", tz="UTC"), timestamp < pd.Timestamp("2025-04-01", tz="UTC")],
        ["development_2024", "calibration_2025q1"], default="evaluation_2025q2_q4",
    )
    profile_index_path = output.parent / "profile_index.parquet"
    profile_index.to_parquet(profile_index_path, index=False, compression="zstd")
    protocol_path = output.parent / "protocol_license_matrix.csv"
    pd.DataFrame(protocol_license_rows()).to_csv(protocol_path, index=False)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    write_json(output.with_suffix(".json"), {
        "dataset_version": config["study"].get("dataset_version"), "seed": seed, "fractions": settings["fractions"],
        "stratification": "macro-region quotas with longitude-sector-balanced deterministic assignment",
        "station_count": len(result), "counts_by_split_and_region": counts.to_dict(orient="index"),
        "southern_minimums": southern_minimums, "sha256": digest,
        "collocation_fingerprint": collocation.get("igra_profile_fingerprint"),
        "profile_index": str(profile_index_path.resolve()), "protocol_license_matrix": str(protocol_path.resolve()),
    })
    print(counts.to_string())
    print(f"Split file: {output.resolve()}")
    print(f"Protocol license matrix: {protocol_path.resolve()}")


if __name__ == "__main__":
    main()
