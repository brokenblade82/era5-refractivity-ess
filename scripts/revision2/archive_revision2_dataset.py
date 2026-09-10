from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import PROJECT_ROOT  # noqa: F401
from igra_forecast.revision2_data import write_json


DEFAULT_ARCHIVE = "data/revision2/archive/sixlevel_min4_total730_20260828"
DATASET_DIRS = ("igra_profiles", "era5_igra_profiles")
UPSTREAM_MANIFESTS = {
    "era5_archive_manifest.json",
    "era5_archive_audit.csv",
    "igra_download_manifest.json",
    "igra_candidate_stations.csv",
    "igra_station_exclusions.csv",
    "igra_station_metadata_issues.csv",
}


def _move_once(source: Path, destination: Path) -> dict[str, object]:
    if destination.exists():
        if source.exists():
            raise FileExistsError(f"Both source and archive target exist: {source}, {destination}")
        return {"source": str(source.resolve()), "destination": str(destination.resolve()), "status": "already_archived"}
    if not source.exists():
        return {"source": str(source.resolve()), "destination": str(destination.resolve()), "status": "absent"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)
    return {"source": str(source.resolve()), "destination": str(destination.resolve()), "status": "moved"}


def archive_revision2_outputs(root: Path, archive: Path) -> dict[str, object]:
    root = root.resolve()
    archive = archive.resolve()
    if root not in archive.parents:
        raise ValueError(f"Archive must remain inside {root}: {archive}")
    archive.mkdir(parents=True, exist_ok=True)
    actions: list[dict[str, object]] = []
    processed = root / "processed"
    for name in DATASET_DIRS:
        actions.append(_move_once(processed / name, archive / "processed" / name))

    manifests = root / "manifests"
    archive_manifests = archive / "manifests"
    archive_manifests.mkdir(parents=True, exist_ok=True)
    if manifests.exists():
        for source in sorted(manifests.iterdir()):
            if not source.is_file():
                continue
            destination = archive_manifests / source.name
            if not destination.exists():
                shutil.copy2(source, destination)
            if source.name not in UPSTREAM_MANIFESTS:
                source.unlink()
            actions.append(
                {
                    "source": str(source.resolve()),
                    "destination": str(destination.resolve()),
                    "status": "copied_and_removed" if source.name not in UPSTREAM_MANIFESTS else "copied_upstream_retained",
                }
            )

    old_profile_manifest = archive_manifests / "igra_profile_manifest.json"
    old_collocation_manifest = archive_manifests / "era5_igra_profile_manifest.json"
    evidence = {}
    for label, path in [("igra", old_profile_manifest), ("collocation", old_collocation_manifest)]:
        if path.is_file():
            evidence[label] = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "archive_version": "sixlevel_min4_total730_20260828",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root),
        "archive_root": str(archive),
        "raw_igra_moved": False,
        "era5_moved": False,
        "actions": actions,
        "old_evidence": {
            "minimum_valid_levels": evidence.get("igra", {}).get("minimum_valid_levels"),
            "chosen_station_profile_threshold": evidence.get("igra", {}).get("chosen_station_profile_threshold"),
            "eligible_stations": evidence.get("igra", {}).get("eligible_stations"),
            "igra_data_fingerprint": evidence.get("igra", {}).get("data_fingerprint"),
            "collocation_config_fingerprint": evidence.get("collocation", {}).get("config_fingerprint"),
        },
    }
    write_json(archive / "archive_manifest.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-destructively archive the completed min4/total730 revision-2 product.")
    parser.add_argument("--root", default="data/revision2")
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    payload = archive_revision2_outputs(Path(args.root), Path(args.archive))
    print(f"Archived revision-2 outputs: {payload['archive_root']}")
    print("Raw IGRA ZIP files and the ERA5 archive were not moved.")


if __name__ == "__main__":
    main()
