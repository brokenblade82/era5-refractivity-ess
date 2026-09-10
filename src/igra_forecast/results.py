from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def create_run_dir(results_dir: Path, model_name: str, experiment_name: str | None = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = _safe_name(model_name)
    suffix = f"_{_safe_name(experiment_name)}" if experiment_name else ""
    run_dir = results_dir / f"{safe_model}_{timestamp}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def save_environment(path: Path) -> None:
    distribution_names = {
        "torch": "torch",
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "sklearn": "scikit-learn",
        "statsmodels": "statsmodels",
        "seaborn": "seaborn",
    }
    packages = {}
    for import_name, distribution_name in distribution_names.items():
        try:
            packages[import_name] = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            packages[import_name] = None
    save_json(
        {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
        },
        path,
    )


def _safe_name(value: str | None) -> str:
    if not value:
        return ""
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
