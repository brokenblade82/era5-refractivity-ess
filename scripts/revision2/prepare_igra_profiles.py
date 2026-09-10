from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from _bootstrap import PROJECT_ROOT  # noqa: F401
from igra_forecast.revision2_data import (
    atomic_promote_directory,
    data_namespace,
    ensure_data_directories,
    load_data_config,
    macro_region,
    parse_int,
    refractivity_components,
    revision2_config_fingerprint,
    sha256_file,
    specific_humidity_from_vapor_pressure,
    stable_fingerprint,
    vapor_pressure_from_humidity,
    write_json,
)


def _number(text: str, scale: float = 1.0) -> float:
    value = parse_int(text)
    return np.nan if value is None else value / scale


def parse_level(line: str) -> dict[str, object] | None:
    if len(line) < 39:
        return None
    pressure_pa = _number(line[9:15])
    if not np.isfinite(pressure_pa):
        return None
    return {
        "level_type_1": line[0:1],
        "level_type_2": line[1:2],
        "elapsed_time_hhmm": line[3:8].strip(),
        "pressure_hpa": pressure_pa / 100.0,
        "pressure_flag": line[15:16].strip(),
        "height_m": _number(line[16:21]),
        "height_flag": line[21:22].strip(),
        "temperature_c": _number(line[22:27], 10.0),
        "temperature_flag": line[27:28].strip(),
        "relative_humidity_percent": _number(line[28:33], 10.0),
        "dewpoint_depression_c": _number(line[34:39], 10.0),
    }


def parse_header(line: str) -> tuple[str, pd.Timestamp, int] | None:
    if not line.startswith("#"):
        return None
    parts = line[1:].split()
    if len(parts) < 7:
        return None
    try:
        station_id = parts[0]
        year, month, day, hour = map(int, parts[1:5])
        n_levels = int(parts[6])
        time = pd.Timestamp(year=year, month=month, day=day, hour=hour, tz="UTC")
    except (TypeError, ValueError):
        return None
    return station_id, time, n_levels


def parse_station_metadata(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="ascii", errors="replace").splitlines():
        if len(line) < 40:
            continue
        try:
            rows.append(
                {
                    "station_id": line[:11].strip(),
                    "latitude": float(line[12:20]),
                    "longitude": float(line[21:30]),
                    "station_elevation_m": float(line[31:37]),
                    "station_name": line[41:71].strip(),
                }
            )
        except ValueError:
            continue
    frame = pd.DataFrame(rows).drop_duplicates("station_id")
    if frame.empty:
        raise ValueError(f"Station metadata is empty: {path}")
    return frame.set_index("station_id")


def finalize_profile(
    station_id: str,
    time: pd.Timestamp,
    raw_levels: list[dict[str, object]],
    pressure_levels: list[int],
    min_levels: int,
    station_meta: pd.Series,
) -> pd.DataFrame | None:
    selected = [row for row in raw_levels if row["level_type_1"] == "1" and row["pressure_hpa"] in pressure_levels]
    if not selected:
        return None
    frame = pd.DataFrame(selected)
    frame["valid_thermodynamics"] = (
        np.isfinite(frame["temperature_c"])
        & np.isfinite(frame["height_m"])
        & (frame["temperature_c"] > -120)
        & (frame["temperature_c"] < 60)
    )
    frame = frame.sort_values(["pressure_hpa", "valid_thermodynamics"], ascending=[False, False])
    frame = frame.drop_duplicates("pressure_hpa", keep="first")
    e, humidity_source = vapor_pressure_from_humidity(
        frame["temperature_c"].to_numpy(),
        frame["relative_humidity_percent"].to_numpy(),
        frame["dewpoint_depression_c"].to_numpy(),
    )
    frame["vapor_pressure_hpa"] = e
    frame["humidity_source"] = humidity_source
    frame["specific_humidity"] = specific_humidity_from_vapor_pressure(e, frame["pressure_hpa"].to_numpy())
    frame["valid_level"] = frame["valid_thermodynamics"] & np.isfinite(frame["specific_humidity"])
    if int(frame["valid_level"].sum()) < min_levels:
        return None
    frame["temperature_k"] = frame["temperature_c"] + 273.15
    dry, wet, total, _ = refractivity_components(
        frame["pressure_hpa"].to_numpy(), frame["temperature_k"].to_numpy(), frame["specific_humidity"].to_numpy()
    )
    frame["igra_n_dry"] = np.where(frame["valid_level"], dry, np.nan)
    frame["igra_n_wet"] = np.where(frame["valid_level"], wet, np.nan)
    frame["igra_n"] = np.where(frame["valid_level"], total, np.nan)
    frame = frame.set_index("pressure_hpa").reindex(pressure_levels).rename_axis("pressure_hpa").reset_index()
    frame["valid_level"] = frame["valid_level"].eq(True)
    frame["station_id"] = station_id
    frame["time"] = time.tz_convert(None)
    frame["latitude"] = float(station_meta["latitude"])
    frame["longitude"] = float(station_meta["longitude"])
    frame["station_elevation_m"] = float(station_meta["station_elevation_m"])
    frame["station_name"] = str(station_meta["station_name"])
    frame["year"] = time.year
    frame["month"] = time.month
    frame["level_mask"] = frame["valid_level"].astype("uint8")
    frame["qc_flags"] = (
        "P:" + frame["pressure_flag"].fillna("").astype(str)
        + "|Z:" + frame["height_flag"].fillna("").astype(str)
        + "|T:" + frame["temperature_flag"].fillna("").astype(str)
        + "|H:" + frame["humidity_source"].fillna("missing").astype(str)
    )
    frame = frame.rename(
        columns={"temperature_k": "igra_temperature_k", "specific_humidity": "igra_specific_humidity", "height_m": "igra_height_m"}
    )
    return frame[
        [
            "station_id", "time", "latitude", "longitude", "station_elevation_m", "station_name",
            "year", "month", "pressure_hpa", "level_mask", "igra_temperature_k", "igra_specific_humidity",
            "igra_height_m", "igra_n_dry", "igra_n_wet", "igra_n", "vapor_pressure_hpa", "humidity_source", "qc_flags",
            "pressure_flag", "height_flag", "temperature_flag", "level_type_1", "level_type_2",
        ]
    ]


def parse_archive(
    archive: Path,
    station_meta: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    hours: set[int],
    levels: list[int],
    min_levels: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    profile_frames: list[pd.DataFrame] = []
    counters: defaultdict[str, int] = defaultdict(int)
    with zipfile.ZipFile(archive) as bundle:
        members = [name for name in bundle.namelist() if name.lower().endswith(".txt")]
        if len(members) != 1:
            raise ValueError(f"Expected one text member in {archive}, found {len(members)}")
        with bundle.open(members[0]) as binary:
            stream = (line.decode("ascii", errors="replace").rstrip("\r\n") for line in binary)
            iterator = iter(stream)
            for line in iterator:
                header = parse_header(line)
                if header is None:
                    continue
                station_id, time, n_levels = header
                raw_levels = []
                for _ in range(n_levels):
                    try:
                        level_line = next(iterator)
                    except StopIteration:
                        counters["truncated_soundings"] += 1
                        break
                    parsed = parse_level(level_line)
                    if parsed is not None:
                        raw_levels.append(parsed)
                counters["raw_soundings"] += 1
                if time < start or time > end:
                    counters["outside_period"] += 1
                    continue
                if time.hour not in hours:
                    counters["outside_synoptic_hours"] += 1
                    continue
                if station_id not in station_meta.index:
                    counters["missing_station_metadata"] += 1
                    continue
                profile = finalize_profile(station_id, time, raw_levels, levels, min_levels, station_meta.loc[station_id])
                if profile is None:
                    counters["insufficient_valid_levels"] += 1
                    continue
                profile_frames.append(profile)
                counters["retained_profiles"] += 1
    return (pd.concat(profile_frames, ignore_index=True) if profile_frames else pd.DataFrame()), dict(counters)


def threshold_sensitivity(summary: pd.DataFrame, candidates: list[int], min_total: int, min_region: int) -> pd.DataFrame:
    """Report legacy two-year total thresholds without using them for selection."""
    attempts = []
    for threshold in candidates:
        selected = summary.loc[summary["retained_profiles"] >= threshold]
        counts = selected.groupby("macro_region")["station_id"].nunique().to_dict()
        accepted = len(selected) >= min_total and all(counts.get(region, 0) >= min_region for region in [
            "northern_extratropics", "tropics", "southern_extratropics"
        ])
        attempts.append({"threshold": threshold, "stations": len(selected), **counts, "passes_global_regional_audit": accepted})
    return pd.DataFrame(attempts)


def _temporal_profile_audit(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {
            "profiles_development_2024": 0, "months_development_2024": 0,
            "profiles_calibration_2025q1": 0, "months_calibration_2025q1": 0,
            "profiles_evaluation_2025q2_q4": 0, "months_evaluation_2025q2_q4": 0,
        }
    profiles = frame[["station_id", "time"]].drop_duplicates()
    time = pd.DatetimeIndex(pd.to_datetime(profiles["time"], utc=True))
    masks = {
        "development_2024": time < pd.Timestamp("2025-01-01", tz="UTC"),
        "calibration_2025q1": (time >= pd.Timestamp("2025-01-01", tz="UTC")) & (time < pd.Timestamp("2025-04-01", tz="UTC")),
        "evaluation_2025q2_q4": time >= pd.Timestamp("2025-04-01", tz="UTC"),
    }
    result: dict[str, int] = {}
    for name, mask in masks.items():
        selected = profiles.loc[np.asarray(mask)]
        result[f"profiles_{name}"] = int(len(selected))
        result[f"months_{name}"] = int(pd.DatetimeIndex(selected["time"]).to_period("M").nunique()) if len(selected) else 0
    return result


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def migrate_legacy_smoke_artifacts(paths: dict[str, Path], config: dict[str, Any]) -> bool:
    """Move the old three-station output out of the formal namespace without deleting it."""
    legacy_manifest = paths["manifests"] / "igra_profile_manifest.json"
    if not legacy_manifest.is_file():
        return False
    payload = _load_json(legacy_manifest)
    if payload.get("mode") == "formal":
        return False
    eligible = int(payload.get("eligible_stations", 0))
    threshold = payload.get("chosen_station_profile_threshold")
    if eligible >= int(config["study"]["min_total_stations"]) or threshold not in (0, "smoke", None):
        return False
    smoke_paths = data_namespace(paths, smoke=True)
    moves = [
        (paths["processed"] / "igra_profiles", smoke_paths["processed"] / "igra_profiles"),
        (paths["interim"] / "igra_by_station", smoke_paths["interim"] / "igra_by_station"),
    ]
    for source, destination in moves:
        if source.exists() and not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
    for name in [
        "igra_profile_manifest.json", "igra_station_retention.csv", "igra_eligibility_selection.csv", "igra_threshold_selection.csv",
        "igra_download_manifest.json", "igra_candidate_stations.csv", "igra_station_exclusions.csv",
        "igra_station_metadata_issues.csv",
    ]:
        source = paths["manifests"] / name
        destination = smoke_paths["manifests"] / name
        if source.exists() and not destination.exists():
            shutil.move(str(source), str(destination))
    smoke_download = smoke_paths["manifests"] / "igra_download_manifest.json"
    smoke_candidates = smoke_paths["manifests"] / "igra_candidate_stations.csv"
    station_list = paths["raw"] / "igra" / "metadata" / Path(config["igra"]["station_list"]).name
    if smoke_download.is_file():
        download_payload = _load_json(smoke_download)
        if download_payload.get("mode") is None:
            download_payload.update(
                {
                    "mode": "smoke",
                    "complete": int(download_payload.get("candidate_station_count", 0)) > 0,
                    "config_fingerprint": revision2_config_fingerprint(config),
                    "candidate_path": str(smoke_candidates.resolve()),
                    "station_metadata_path": str(station_list.resolve()),
                    "station_metadata_sha256": sha256_file(station_list),
                    "failed_sounding_files": 0,
                }
            )
            write_json(smoke_download, download_payload)
    return True


def _cache_paths(interim: Path, station_id: str) -> tuple[Path, Path]:
    return interim / f"{station_id}.parquet", interim / f"{station_id}.audit.json"


def _cache_reusable(target: Path, audit_path: Path, archive: Path, config_fingerprint: str) -> dict[str, Any] | None:
    if not audit_path.is_file():
        return None
    audit = _load_json(audit_path)
    if audit.get("config_fingerprint") != config_fingerprint:
        return None
    stat = archive.stat()
    if int(audit.get("archive_bytes", -1)) != stat.st_size or int(audit.get("archive_mtime_ns", -1)) != stat.st_mtime_ns:
        return None
    if int(audit.get("retained_profiles", 0)) > 0 and not target.is_file():
        return None
    audit["reused_cache"] = True
    return audit


def _parse_archive_worker(
    archive_text: str,
    target_text: str,
    audit_text: str,
    station_record: dict[str, Any],
    start_text: str,
    end_text: str,
    hours: list[int],
    levels: list[int],
    min_levels: int,
    config_fingerprint: str,
) -> dict[str, Any]:
    archive, target, audit_path = Path(archive_text), Path(target_text), Path(audit_text)
    station_id = str(station_record["station_id"])
    station_meta = pd.DataFrame([station_record]).set_index("station_id")
    frame, counters = parse_archive(
        archive,
        station_meta,
        pd.Timestamp(start_text),
        pd.Timestamp(end_text),
        set(map(int, hours)),
        list(map(int, levels)),
        int(min_levels),
    )
    stat = archive.stat()
    archive_hash = sha256_file(archive)
    audit: dict[str, Any] = {
        "station_id": station_id,
        "archive": str(archive),
        "archive_bytes": stat.st_size,
        "archive_mtime_ns": stat.st_mtime_ns,
        "archive_sha256": archive_hash,
        "config_fingerprint": config_fingerprint,
        **counters,
        "retained_profiles": int(counters.get("retained_profiles", 0)),
        "reused_cache": False,
        "latitude": float(station_record["latitude"]),
        "longitude": float(station_record["longitude"]),
        "station_elevation_m": float(station_record["station_elevation_m"]),
        "macro_region": macro_region(float(station_record["latitude"])),
        **_temporal_profile_audit(frame),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        target.unlink(missing_ok=True)
    else:
        temporary = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, target)
    write_json(audit_path, audit)
    return audit


def _station_metadata_is_valid(frame: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    settings = config["igra"]
    lat_min, lat_max = map(float, settings.get("valid_latitude_range", [-90.0, 90.0]))
    lon_min, lon_max = map(float, settings.get("valid_longitude_range", [-180.0, 180.0]))
    return (
        frame["latitude"].between(lat_min, lat_max)
        & frame["longitude"].between(lon_min, lon_max)
        & np.isfinite(frame[["latitude", "longitude"]]).all(axis=1)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse official IGRA soundings into six-level masked profiles.")
    parser.add_argument("--config", default="configs/revision2/data_pipeline.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse valid per-station caches and rebuild only missing stations.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Discard per-station parse caches; raw ZIP files are preserved.")
    parser.add_argument("--workers", type=int, help="Parallel station parsers; defaults to igra.workers.")
    parser.add_argument("--max-stations", type=int, help="Smoke-test limit; disables threshold selection.")
    args = parser.parse_args()
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume are mutually exclusive")
    config = load_data_config(args.config)
    paths = ensure_data_directories(config)
    migrated = migrate_legacy_smoke_artifacts(paths, config)
    smoke = args.max_stations is not None
    namespace = data_namespace(paths, smoke=smoke)
    raw_dir = paths["raw"] / "igra"
    station_list = raw_dir / "metadata" / Path(config["igra"]["station_list"]).name
    station_meta = parse_station_metadata(station_list)
    candidate_path = namespace["manifests"] / "igra_candidate_stations.csv"
    download_manifest_path = namespace["manifests"] / "igra_download_manifest.json"
    if not candidate_path.is_file() or not download_manifest_path.is_file():
        raise FileNotFoundError(f"IGRA download evidence is missing in {namespace['manifests']}; run download_igra.py first")
    download_manifest = _load_json(download_manifest_path)
    expected_mode = "smoke" if smoke else "formal"
    if download_manifest.get("mode") != expected_mode or not bool(download_manifest.get("complete")):
        raise ValueError(f"IGRA {expected_mode} download manifest is incomplete or has the wrong mode")
    candidates = pd.read_csv(candidate_path)
    if args.max_stations:
        candidates = candidates.head(args.max_stations)
    if not _station_metadata_is_valid(candidates, config).all():
        raise ValueError("Candidate station list contains invalid coordinates")
    candidate_lookup = candidates.set_index("station_id")
    archive_lookup = {path.name[:11]: path for path in (raw_dir / "data-por").glob("*-data.txt.zip")}
    missing_archives = sorted(set(candidates["station_id"]) - set(archive_lookup))
    if missing_archives:
        raise FileNotFoundError(f"Missing {len(missing_archives)} IGRA archives; first stations: {missing_archives[:10]}")
    archives = [archive_lookup[station_id] for station_id in candidates["station_id"]]
    interim = namespace["interim"] / "igra_by_station"
    output = namespace["processed"] / "igra_profiles"
    manifest_path = namespace["manifests"] / "igra_profile_manifest.json"
    if args.rebuild_cache:
        shutil.rmtree(interim, ignore_errors=True)
    if output.exists() and (output / "_SUCCESS").is_file() and args.resume and manifest_path.is_file():
        existing = _load_json(manifest_path)
        if existing.get("mode") == expected_mode and existing.get("config_fingerprint") == revision2_config_fingerprint(config):
            print(f"Complete {expected_mode} IGRA profile product already exists: {output.resolve()}")
            return
    if output.exists() and not (args.overwrite or args.resume):
        raise FileExistsError(f"Output already exists; pass --resume or --overwrite: {output}")
    interim.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(config["study"]["start"])
    end = pd.Timestamp(config["study"]["end"])
    levels = list(map(int, config["study"]["pressure_levels_hpa"]))
    hours = set(map(int, config["study"]["synoptic_hours_utc"]))
    min_levels = int(config["study"]["min_valid_levels_per_profile"])
    config_fingerprint = revision2_config_fingerprint(config)
    workers = int(args.workers or config["igra"].get("workers", 1))
    audit_rows: list[dict[str, Any]] = []
    pending = []
    for archive in archives:
        station_id = archive.name[:11]
        target, audit_path = _cache_paths(interim, station_id)
        cached = _cache_reusable(target, audit_path, archive, config_fingerprint) if args.resume else None
        if cached is not None:
            audit_rows.append(cached)
            continue
        if station_id not in station_meta.index:
            raise ValueError(f"Station metadata is missing for {station_id}")
        record = station_meta.loc[station_id].to_dict()
        record["station_id"] = station_id
        record["station_elevation_m"] = float(candidate_lookup.loc[station_id, "elevation_m"])
        pending.append((archive, target, audit_path, record))
    if workers <= 1:
        for archive, target, audit_path, record in tqdm(pending, desc="Parse IGRA stations"):
            audit_rows.append(_parse_archive_worker(
                str(archive), str(target), str(audit_path), record, str(start), str(end), sorted(hours), levels,
                min_levels, config_fingerprint,
            ))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _parse_archive_worker, str(archive), str(target), str(audit_path), record,
                    str(start), str(end), sorted(hours), levels, min_levels, config_fingerprint,
                )
                for archive, target, audit_path, record in pending
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Parse IGRA stations"):
                audit_rows.append(future.result())
    summary = pd.DataFrame(audit_rows).fillna({"retained_profiles": 0})
    summary["retained_profiles"] = summary["retained_profiles"].astype(int)
    temporal_coverage = {
        key: int(value) for key, value in config["study"]["station_temporal_coverage"].items()
    }
    sensitivity = threshold_sensitivity(
        summary,
        list(map(int, config["study"]["station_profile_threshold_candidates"])),
        int(config["study"]["min_total_stations"]),
        int(config["study"]["min_stations_per_macro_region"]),
    )
    if args.max_stations:
        summary["eligible"] = summary["retained_profiles"] > 0
        summary["eligibility_reason"] = np.where(summary["eligible"], "smoke_has_retained_profiles", "smoke_no_retained_profiles")
    else:
        pass_columns = []
        for role, threshold in temporal_coverage.items():
            count_column = f"profiles_{role}"
            pass_column = f"passes_{role}"
            if count_column not in summary:
                summary[count_column] = 0
            summary[pass_column] = summary[count_column].fillna(0).astype(int) >= int(threshold)
            pass_columns.append(pass_column)
        summary["eligible"] = summary[pass_columns].all(axis=1)
        failed = summary[pass_columns].apply(
            lambda row: ";".join(column.removeprefix("passes_") for column, passed in row.items() if not bool(passed)), axis=1
        )
        summary["eligibility_reason"] = np.where(summary["eligible"], "all_temporal_coverage_periods", "failed:" + failed)
        selected_now = summary.loc[summary["eligible"]]
        region_now = selected_now.groupby("macro_region")["station_id"].nunique().to_dict()
        required_regions = ["northern_extratropics", "tropics", "southern_extratropics"]
        if len(selected_now) < int(config["study"]["min_total_stations"]) or any(
            int(region_now.get(region, 0)) < int(config["study"]["min_stations_per_macro_region"])
            for region in required_regions
        ):
            summary.to_csv(paths["manifests"] / "igra_station_retention.csv", index=False)
            summary.to_csv(paths["manifests"] / "igra_eligibility_selection.csv", index=False)
            sensitivity.to_csv(paths["manifests"] / "igra_threshold_selection.csv", index=False)
            raise RuntimeError("The preregistered temporal-coverage rule does not meet global/regional station gates")
    eligible = set(summary.loc[summary["eligible"], "station_id"])
    if not eligible:
        raise RuntimeError("No eligible IGRA stations were retained")
    staging = output.with_name(output.name + ".building")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    batch = []
    for index, station_id in enumerate(tqdm(sorted(eligible), desc="Write partitioned IGRA dataset"), start=1):
        batch.append(pd.read_parquet(interim / f"{station_id}.parquet"))
        if len(batch) >= 25 or index == len(eligible):
            frame = pd.concat(batch, ignore_index=True)
            pq.write_to_dataset(
                pa.Table.from_pandas(frame, preserve_index=False), root_path=str(staging),
                partition_cols=["year", "month"], compression="zstd",
            )
            batch.clear()
    selected = summary.loc[summary["eligible"]].sort_values("station_id")
    region_counts = selected.groupby("macro_region")["station_id"].nunique().to_dict()
    selected_candidate_invalid_count = int((~_station_metadata_is_valid(
        candidates.rename(columns={"elevation_m": "elevation_m"}), config
    )).sum())
    data_fingerprint = stable_fingerprint(
        {
            "config_fingerprint": config_fingerprint,
            "mode": expected_mode,
            "selection_method": "all_temporal_coverage_periods" if not args.max_stations else "smoke_nonempty",
            "station_temporal_coverage": temporal_coverage,
            "station_metadata_sha256": download_manifest.get("station_metadata_sha256"),
            "stations": selected[["station_id", "archive_sha256", "retained_profiles"]].to_dict(orient="records"),
        }
    )
    write_json(staging / "_SUCCESS", {"data_fingerprint": data_fingerprint, "mode": expected_mode})
    atomic_promote_directory(staging, output)
    summary.to_csv(namespace["manifests"] / "igra_station_retention.csv", index=False)
    summary.to_csv(namespace["manifests"] / "igra_eligibility_selection.csv", index=False)
    sensitivity.to_csv(namespace["manifests"] / "igra_threshold_selection.csv", index=False)
    write_json(
        manifest_path,
        {
            "mode": expected_mode,
            "complete": True,
            "parser_version": 3,
            "dataset_version": config["study"].get("dataset_version"),
            "config_fingerprint": config_fingerprint,
            "data_fingerprint": data_fingerprint,
            "study_period": [config["study"]["start"], config["study"]["end"]],
            "synoptic_hours_utc": sorted(hours), "pressure_levels_hpa": levels,
            "minimum_valid_levels": min_levels,
            "selection_method": "all_temporal_coverage_periods" if not args.max_stations else "smoke_nonempty",
            "station_temporal_coverage": temporal_coverage,
            "legacy_total_thresholds_are_sensitivity_only": True,
            "candidate_stations": int(len(candidates)),
            "selected_candidate_invalid_count": selected_candidate_invalid_count,
            "selected_candidate_missing_elevation_count": int(candidates["elevation_m"].isna().sum()),
            "eligible_stations": len(eligible),
            "eligible_stations_by_macro_region": {key: int(value) for key, value in region_counts.items()},
            "station_metadata_sha256": download_manifest.get("station_metadata_sha256"),
            "download_manifest": str(download_manifest_path.resolve()),
            "output": str(output.resolve()),
            "legacy_smoke_migrated": migrated,
        },
    )
    print(f"Eligible stations: {len(eligible)}; selection: temporal coverage in all three periods")
    print(f"IGRA Parquet dataset: {output.resolve()}")


if __name__ == "__main__":
    main()
