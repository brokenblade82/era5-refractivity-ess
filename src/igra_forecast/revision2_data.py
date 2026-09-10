from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
import yaml


EPSILON = 0.622
G0 = 9.80665
MISSING_INTEGERS = {-9999, -8888, -99999, -88888}
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def stable_fingerprint(payload: Any) -> str:
    """Return a deterministic SHA-256 for JSON-compatible scientific metadata."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_data_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    required = {"study", "paths", "igra", "era5", "rapsodi", "cosmic2", "splits"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing data-pipeline config sections: {sorted(missing)}")
    return config


def resolve_era5_root(config: dict[str, Any], override: str | Path | None = None) -> Path:
    """Resolve the ERA5 archive root, with CLI override taking precedence."""
    raw = override if override is not None else config.get("era5", {}).get("root")
    if raw is None or not str(raw).strip():
        raise ValueError("ERA5 root is not configured; set era5.root or pass --era5-root")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def era5_archive_fingerprint(config: dict[str, Any], root: str | Path) -> str:
    """Fingerprint archive identity and the scientific fields required by this study."""
    payload = {
        "root": os.path.normcase(str(Path(root).resolve())),
        "study_start": config["study"]["start"],
        "study_end": config["study"]["end"],
        "pressure_levels_hpa": list(map(int, config["study"]["pressure_levels_hpa"])),
        "expected_hours_utc": list(map(int, config["era5"]["expected_hours_utc"])),
        "expected_grid_degrees": float(config["era5"]["expected_grid_degrees"]),
        "required_pressure_variables": config["era5"]["required_pressure_variables"],
        "required_single_variables": config["era5"]["required_single_variables"],
        "required_static_variables": config["era5"]["required_static_variables"],
    }
    return stable_fingerprint(payload)


def revision2_config_fingerprint(config: dict[str, Any]) -> str:
    """Fingerprint fields that define the revision-2 scientific data contract."""
    payload = {
        "version": config.get("version"),
        "study": config["study"],
        "igra": {
            key: config["igra"].get(key)
            for key in [
                "base_url", "station_list", "data_format", "data_directory", "active_end_year_min",
                "valid_latitude_range", "valid_longitude_range", "valid_elevation_m_range",
                "reject_variable_flags",
            ]
        },
        "era5": {
            "expected_grid_degrees": config["era5"]["expected_grid_degrees"],
            "expected_hours_utc": config["era5"]["expected_hours_utc"],
            "required_pressure_variables": config["era5"]["required_pressure_variables"],
            "required_single_variables": config["era5"]["required_single_variables"],
        },
        "splits": config["splits"],
    }
    return stable_fingerprint(payload)


def normalize_path(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def data_namespace(paths: dict[str, Path], *, smoke: bool) -> dict[str, Path]:
    """Return isolated paths for smoke or formal revision-2 products."""
    if not smoke:
        return paths
    root = paths["root"] / "smoke"
    namespace = {
        "root": root,
        "raw": paths["raw"],  # Raw downloads are immutable and shared.
        "interim": root / "interim",
        "processed": root / "processed",
        "manifests": root / "manifests",
    }
    for key in ["root", "interim", "processed", "manifests"]:
        namespace[key].mkdir(parents=True, exist_ok=True)
    return namespace


def atomic_promote_directory(staging: str | Path, target: str | Path) -> None:
    """Promote a completed staging directory without exposing partial output."""
    staging = Path(staging)
    target = Path(target)
    if not staging.is_dir() or not (staging / "_SUCCESS").is_file():
        raise ValueError(f"Staging directory is incomplete: {staging}")
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = target.with_name(target.name + ".previous")
    if backup.exists():
        shutil.rmtree(backup)
    if target.exists():
        os.replace(target, backup)
    try:
        os.replace(staging, target)
    except Exception:
        if backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _read_manifest(path: str | Path, label: str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} manifest is missing: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def require_igra_profile_manifest(
    config: dict[str, Any],
    manifest_path: str | Path,
    profile_root: str | Path,
    *,
    require_formal: bool,
) -> dict[str, Any]:
    """Validate that an IGRA profile product is complete and has the intended evidence level."""
    manifest = _read_manifest(manifest_path, "IGRA profile")
    expected_mode = "formal" if require_formal else "smoke"
    if manifest.get("mode") != expected_mode:
        raise ValueError(f"IGRA profile mode mismatch: expected {expected_mode}, found {manifest.get('mode')!r}")
    if not bool(manifest.get("complete")):
        raise ValueError("IGRA profile manifest is not complete")
    if normalize_path(manifest.get("output", "")) != normalize_path(profile_root):
        raise ValueError("IGRA profile output path does not match the requested dataset")
    if manifest.get("config_fingerprint") != revision2_config_fingerprint(config):
        raise ValueError("IGRA profile configuration fingerprint mismatch")
    if list(map(int, manifest.get("pressure_levels_hpa", []))) != list(map(int, config["study"]["pressure_levels_hpa"])):
        raise ValueError("IGRA profile pressure levels do not match the current configuration")
    if not (Path(profile_root) / "_SUCCESS").is_file():
        raise ValueError(f"IGRA profile dataset has no _SUCCESS marker: {profile_root}")
    if not manifest.get("data_fingerprint"):
        raise ValueError("IGRA profile manifest has no data fingerprint")
    if require_formal:
        expected_version = str(config["study"].get("dataset_version", ""))
        if manifest.get("dataset_version") != expected_version:
            raise ValueError(
                f"Formal IGRA dataset version mismatch: expected {expected_version!r}, "
                f"found {manifest.get('dataset_version')!r}"
            )
        expected_coverage = {
            key: int(value) for key, value in config["study"]["station_temporal_coverage"].items()
        }
        actual_coverage = {
            key: int(value) for key, value in manifest.get("station_temporal_coverage", {}).items()
        }
        if actual_coverage != expected_coverage:
            raise ValueError(
                f"Formal IGRA temporal-coverage rule mismatch: expected {expected_coverage}, "
                f"found {actual_coverage}"
            )
        if manifest.get("selection_method") != "all_temporal_coverage_periods":
            raise ValueError("Formal IGRA product was not selected by all temporal coverage periods")
        if int(manifest.get("minimum_valid_levels", -1)) != int(config["study"]["min_valid_levels_per_profile"]):
            raise ValueError("Formal IGRA minimum-valid-level rule mismatch")
        eligible = int(manifest.get("eligible_stations", 0))
        if eligible < int(config["study"]["min_total_stations"]):
            raise ValueError(f"Formal IGRA product has only {eligible} eligible stations")
        region_counts = manifest.get("eligible_stations_by_macro_region", {})
        required_regions = {"northern_extratropics", "tropics", "southern_extratropics"}
        minimum = int(config["study"]["min_stations_per_macro_region"])
        deficient = {region: int(region_counts.get(region, 0)) for region in required_regions if int(region_counts.get(region, 0)) < minimum}
        if deficient:
            raise ValueError(f"Formal IGRA regional station minimum is not met: {deficient}")
        if int(manifest.get("selected_candidate_invalid_count", -1)) != 0:
            raise ValueError("Formal IGRA product contains candidates with invalid station metadata")
    return manifest


def require_collocation_manifest(
    config: dict[str, Any],
    manifest_path: str | Path,
    output_root: str | Path,
    *,
    require_formal: bool,
) -> dict[str, Any]:
    """Validate a completed ERA5-IGRA collocation before downstream analysis."""
    manifest = _read_manifest(manifest_path, "ERA5-IGRA collocation")
    expected_mode = "formal" if require_formal else "smoke"
    if manifest.get("mode") != expected_mode:
        raise ValueError(f"Collocation mode mismatch: expected {expected_mode}, found {manifest.get('mode')!r}")
    if not bool(manifest.get("complete")):
        raise ValueError("ERA5-IGRA collocation is not complete")
    if normalize_path(manifest.get("output", "")) != normalize_path(output_root):
        raise ValueError("Collocation output path does not match the requested dataset")
    if manifest.get("config_fingerprint") != revision2_config_fingerprint(config):
        raise ValueError("Collocation configuration fingerprint mismatch")
    if manifest.get("dataset_version") != config["study"].get("dataset_version"):
        raise ValueError("Collocation dataset version mismatch")
    if not (Path(output_root) / "_SUCCESS").is_file():
        raise ValueError(f"Collocation dataset has no _SUCCESS marker: {output_root}")
    months = list(map(str, manifest.get("months", [])))
    expected_months = month_keys(config["study"]["start"], config["study"]["end"])
    if require_formal and months != expected_months:
        raise ValueError("Formal collocation does not contain the complete study period")
    if not manifest.get("era5_archive_fingerprint") or not manifest.get("igra_profile_fingerprint"):
        raise ValueError("Collocation manifest is missing an upstream data fingerprint")
    return manifest


def require_era5_audit(
    config: dict[str, Any],
    manifest_path: str | Path,
    root: str | Path,
    requested_months: Iterable[str],
    *,
    require_full_study: bool,
) -> dict[str, Any]:
    """Reject collocation when the audited archive does not match the requested run."""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"ERA5 audit manifest is missing: {manifest_path}. Run audit_era5_archive.py first."
        )
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    expected_root = os.path.normcase(str(Path(root).resolve()))
    actual_root = os.path.normcase(str(Path(manifest.get("era5_root", "")).resolve()))
    if actual_root != expected_root:
        raise ValueError(f"ERA5 audit root mismatch: audited={actual_root}, requested={expected_root}")
    expected_fingerprint = era5_archive_fingerprint(config, root)
    if manifest.get("archive_fingerprint") != expected_fingerprint:
        raise ValueError("ERA5 audit fingerprint does not match the current root/configuration")
    audited = set(map(str, manifest.get("audited_months", [])))
    requested = set(map(str, requested_months))
    invalid_months = set(map(str, manifest.get("invalid_months", [])))
    invalid_requested = sorted(requested & invalid_months)
    if invalid_requested:
        raise ValueError(f"Requested ERA5 months failed audit: {invalid_requested}")
    if require_full_study and not bool(manifest.get("complete")):
        raise ValueError(f"ERA5 audit is incomplete; invalid months: {sorted(invalid_months)}")
    absent = sorted(requested - audited)
    if absent:
        raise ValueError(f"Requested ERA5 months were not audited: {absent}")
    if require_full_study:
        study_months = set(month_keys(config["study"]["start"], config["study"]["end"]))
        absent_study = sorted(study_months - audited)
        if absent_study or not bool(manifest.get("full_study_complete")):
            raise ValueError(f"Full-study ERA5 audit is required; unaudited months: {absent_study}")
    return manifest


def ensure_data_directories(config: dict[str, Any]) -> dict[str, Path]:
    paths = {key: Path(value) for key, value in config["paths"].items()}
    for key in ["root", "raw", "interim", "processed", "manifests"]:
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(temporary, target)


def download_with_resume(
    url: str,
    target: str | Path,
    *,
    timeout: int = 120,
    retries: int = 5,
    chunk_size: int = 1024 * 1024,
    progress_desc: str | None = None,
    proxy: str | None = None,
) -> dict[str, Any]:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return {
            "url": url,
            "path": str(target),
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
            "reused_existing": True,
        }
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(retries):
        existing = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "IGRA-ERA5-revision2-data/1.0"}
        if existing:
            headers["Range"] = f"bytes={existing}-"
        try:
            proxies = {"http": proxy, "https": proxy} if proxy else None
            with requests.get(url, headers=headers, timeout=timeout, stream=True, proxies=proxies) as response:
                if response.status_code == 416 and partial.exists():
                    os.replace(partial, target)
                    break
                response.raise_for_status()
                append = existing > 0 and response.status_code == 206
                if existing > 0 and not append:
                    partial.unlink(missing_ok=True)
                mode = "ab" if append else "wb"
                total_header = int(response.headers.get("Content-Length", 0))
                total = total_header + (existing if append else 0)
                progress = None
                if progress_desc:
                    from tqdm import tqdm

                    progress = tqdm(total=total or None, initial=existing if append else 0, unit="B", unit_scale=True, desc=progress_desc)
                try:
                    with partial.open(mode) as stream:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                stream.write(chunk)
                                if progress is not None:
                                    progress.update(len(chunk))
                finally:
                    if progress is not None:
                        progress.close()
            os.replace(partial, target)
            break
        except (requests.RequestException, OSError):
            if attempt + 1 >= retries:
                raise
            time.sleep(min(15 * 2**attempt, 240))
    return {
        "url": url,
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        "reused_existing": False,
    }


def saturation_vapor_pressure_hpa(temperature_c: np.ndarray | float) -> np.ndarray:
    temperature_c = np.asarray(temperature_c, dtype=np.float64)
    return 6.112 * np.exp(17.67 * temperature_c / (temperature_c + 243.5))


def vapor_pressure_from_humidity(
    temperature_c: np.ndarray,
    relative_humidity_percent: np.ndarray,
    dewpoint_depression_c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    temperature_c = np.asarray(temperature_c, dtype=np.float64)
    relative_humidity_percent = np.asarray(relative_humidity_percent, dtype=np.float64)
    dewpoint_depression_c = np.asarray(dewpoint_depression_c, dtype=np.float64)
    e = np.full(temperature_c.shape, np.nan, dtype=np.float64)
    source = np.full(temperature_c.shape, "missing", dtype=object)
    dp_ok = np.isfinite(temperature_c) & np.isfinite(dewpoint_depression_c) & (dewpoint_depression_c >= 0)
    e[dp_ok] = saturation_vapor_pressure_hpa(temperature_c[dp_ok] - dewpoint_depression_c[dp_ok])
    source[dp_ok] = "dewpoint_depression"
    rh_ok = (
        np.isfinite(temperature_c)
        & np.isfinite(relative_humidity_percent)
        & (relative_humidity_percent >= 0)
        & (relative_humidity_percent <= 110)
        & ~dp_ok
    )
    e[rh_ok] = saturation_vapor_pressure_hpa(temperature_c[rh_ok]) * relative_humidity_percent[rh_ok] / 100.0
    source[rh_ok] = "relative_humidity"
    return e, source


def specific_humidity_from_vapor_pressure(vapor_pressure_hpa: np.ndarray, pressure_hpa: np.ndarray) -> np.ndarray:
    e = np.asarray(vapor_pressure_hpa, dtype=np.float64)
    p = np.asarray(pressure_hpa, dtype=np.float64)
    denominator = p - (1.0 - EPSILON) * e
    return np.where((denominator > 0) & (e >= 0), EPSILON * e / denominator, np.nan)


def refractivity_components(
    pressure_hpa: np.ndarray,
    temperature_k: np.ndarray,
    specific_humidity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(pressure_hpa, dtype=np.float64)
    t = np.asarray(temperature_k, dtype=np.float64)
    q = np.asarray(specific_humidity, dtype=np.float64)
    e = q * p / (EPSILON + (1.0 - EPSILON) * q)
    dry = 77.6 * p / t
    wet = 3.73e5 * e / np.square(t)
    return dry, wet, dry + wet, e


def month_keys(start: str, end: str) -> list[str]:
    start_month = np.datetime64(start[:7], "M")
    end_month = np.datetime64(end[:7], "M")
    values = np.arange(start_month, end_month + np.timedelta64(1, "M"), dtype="datetime64[M]")
    return [str(value).replace("-", "") for value in values]


def parse_int(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return None if parsed in MISSING_INTEGERS else parsed


def macro_region(latitude: float) -> str:
    if latitude >= 30.0:
        return "northern_extratropics"
    if latitude <= -30.0:
        return "southern_extratropics"
    return "tropics"


TEMPORAL_ROLE_BOUNDS = {
    "development_2024": (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")),
    "calibration_2025q1": (pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-04-01", tz="UTC")),
    "evaluation_2025q2_q4": (pd.Timestamp("2025-04-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")),
}


def temporal_role(value: Any) -> str:
    """Map a sounding time to the preregistered revision-2 temporal role."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    for name, (start, end) in TEMPORAL_ROLE_BOUNDS.items():
        if start <= timestamp < end:
            return name
    raise ValueError(f"Time is outside the revision-2 study protocol: {timestamp}")


def content_manifest(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(paths):
        rows.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows
