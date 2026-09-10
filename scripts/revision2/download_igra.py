from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from _bootstrap import PROJECT_ROOT  # noqa: F401
from igra_forecast.revision2_data import (
    download_with_resume,
    data_namespace,
    ensure_data_directories,
    load_data_config,
    revision2_config_fingerprint,
    sha256_file,
    write_json,
)


def parse_station_list(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="ascii", errors="replace").splitlines():
        if len(line) < 11:
            continue
        station_id = line[:11].strip()
        if not re.fullmatch(r"[A-Z0-9]{11}", station_id):
            continue
        try:
            latitude = float(line[12:20])
            longitude = float(line[21:30])
            elevation_m = float(line[31:37])
            first_year = int(line[72:76])
            last_year = int(line[77:81])
            n_observations = int(line[82:88])
        except (ValueError, IndexError):
            parts = line.split()
            try:
                latitude, longitude, elevation_m = map(float, parts[1:4])
                first_year, last_year, n_observations = map(int, parts[-3:])
            except (ValueError, IndexError):
                continue
        rows.append(
            {
                "station_id": station_id,
                "latitude": latitude,
                "longitude": longitude,
                "elevation_m": elevation_m,
                "first_year": first_year,
                "last_year": last_year,
                "reported_observations": n_observations,
            }
        )
    frame = pd.DataFrame(rows).drop_duplicates("station_id")
    if frame.empty:
        raise ValueError(f"Could not parse station metadata: {path}")
    return frame


def classify_station_metadata(
    stations: pd.DataFrame, settings: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Exclude unusable coordinates while retaining stations with an auditable missing elevation."""
    frame = stations.copy()
    lat_min, lat_max = map(float, settings.get("valid_latitude_range", [-90.0, 90.0]))
    lon_min, lon_max = map(float, settings.get("valid_longitude_range", [-180.0, 180.0]))
    elev_min, elev_max = map(float, settings.get("valid_elevation_m_range", [-500.0, 9000.0]))
    exclusion_reasons: list[str] = []
    issue_reasons: list[str] = []
    for row in frame.itertuples(index=False):
        exclusions = []
        issues = []
        if not np.isfinite(row.latitude) or not lat_min <= float(row.latitude) <= lat_max:
            exclusions.append("invalid_latitude")
        if not np.isfinite(row.longitude) or not lon_min <= float(row.longitude) <= lon_max:
            exclusions.append("invalid_longitude")
        if not np.isfinite(row.elevation_m) or not elev_min <= float(row.elevation_m) <= elev_max:
            issues.append("missing_station_elevation")
        exclusion_reasons.append(";".join(exclusions))
        issue_reasons.append(";".join(issues))
    frame["exclusion_reason"] = exclusion_reasons
    frame["metadata_issue"] = issue_reasons
    valid = frame.loc[frame["exclusion_reason"].eq("")].drop(columns="exclusion_reason").copy()
    valid.loc[valid["metadata_issue"].eq("missing_station_elevation"), "elevation_m"] = np.nan
    excluded = frame.loc[frame["exclusion_reason"].ne("")].copy()
    issues = frame.loc[frame["metadata_issue"].ne("")].copy()
    return valid, excluded, issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official IGRA v2 station soundings with resume support.")
    parser.add_argument("--config", default="configs/revision2/data_pipeline.yaml")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-stations", type=int, help="Smoke-test limit; omit for the complete archive.")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    config = load_data_config(args.config)
    paths = ensure_data_directories(config)
    settings = config["igra"]
    smoke = args.max_stations is not None
    namespace = data_namespace(paths, smoke=smoke)
    raw = paths["raw"] / "igra"
    metadata_dir = raw / "metadata"
    sounding_dir = raw / "data-por"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    sounding_dir.mkdir(parents=True, exist_ok=True)
    base = settings["base_url"].rstrip("/")
    metadata = {}
    for key, relative in {
        "station_list": settings["station_list"],
        "country_list": settings["country_list"],
        "data_format": settings["data_format"],
    }.items():
        target = metadata_dir / Path(relative).name
        metadata[key] = download_with_resume(
            f"{base}/{relative}", target, timeout=settings["timeout_seconds"], retries=settings["retries"]
        )
    stations = parse_station_list(metadata_dir / Path(settings["station_list"]).name)
    active_raw = stations.loc[stations["last_year"] >= int(settings["active_end_year_min"])].copy()
    active, excluded, metadata_issues = classify_station_metadata(active_raw, settings)
    active = active.sort_values("station_id")
    if args.max_stations:
        active = active.head(args.max_stations)
    if args.metadata_only:
        candidate_path = paths["manifests"] / "igra_candidate_stations_all.csv"
        exclusion_path = paths["manifests"] / "igra_station_exclusions.csv"
        issue_path = paths["manifests"] / "igra_station_metadata_issues.csv"
        manifest_path = paths["manifests"] / "igra_metadata_manifest.json"
        mode = "metadata"
    else:
        candidate_path = namespace["manifests"] / "igra_candidate_stations.csv"
        exclusion_path = namespace["manifests"] / "igra_station_exclusions.csv"
        issue_path = namespace["manifests"] / "igra_station_metadata_issues.csv"
        manifest_path = namespace["manifests"] / "igra_download_manifest.json"
        mode = "smoke" if smoke else "formal"
    active.to_csv(candidate_path, index=False)
    excluded.to_csv(exclusion_path, index=False)
    metadata_issues.to_csv(issue_path, index=False)
    records: list[dict[str, Any]] = [dict(record, kind="metadata", status="complete") for record in metadata.values()]
    failures: list[dict[str, Any]] = []
    if not args.metadata_only:
        workers = args.workers or int(settings["workers"])

        def fetch(station_id: str) -> dict:
            filename = f"{station_id}-data.txt.zip"
            return download_with_resume(
                f"{base}/{settings['data_directory'].strip('/')}/{filename}",
                sounding_dir / filename,
                timeout=settings["timeout_seconds"],
                retries=settings["retries"],
            )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch, sid): sid for sid in active["station_id"]}
            for future in tqdm(as_completed(futures), total=len(futures), desc="IGRA stations"):
                station_id = futures[future]
                try:
                    records.append(dict(future.result(), station_id=station_id, kind="sounding", status="complete"))
                except Exception as exc:  # Keep successful stations and provide a retry audit.
                    failures.append(
                        {
                            "station_id": station_id,
                            "kind": "sounding",
                            "status": "failed",
                            "url": f"{base}/{settings['data_directory'].strip('/')}/{station_id}-data.txt.zip",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
    station_list_path = metadata_dir / Path(settings["station_list"]).name
    config_fingerprint = revision2_config_fingerprint(config)
    write_json(
        manifest_path,
        {
            "source": base,
            "mode": mode,
            "complete": args.metadata_only or not failures,
            "config_fingerprint": config_fingerprint,
            "station_metadata_path": str(station_list_path.resolve()),
            "station_metadata_sha256": sha256_file(station_list_path),
            "active_station_count_before_metadata_qc": int(len(active_raw)),
            "excluded_station_count": int(len(excluded)),
            "station_metadata_issue_count": int(len(metadata_issues)),
            "missing_station_elevation_count": int(active["elevation_m"].isna().sum()),
            "candidate_station_count": int(len(active)),
            "metadata_only": args.metadata_only,
            "candidate_path": str(candidate_path.resolve()),
            "exclusion_path": str(exclusion_path.resolve()),
            "metadata_issue_path": str(issue_path.resolve()),
            "successful_sounding_files": sum(record.get("kind") == "sounding" and record.get("status") == "complete" for record in records),
            "failed_sounding_files": len(failures),
            "files": records + failures,
        },
    )
    print(f"IGRA files: {raw.resolve()}")
    print(f"Candidates: {len(active)}; excluded invalid metadata: {len(excluded)}; failures: {len(failures)}")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
