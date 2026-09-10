from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict[str, Any]) -> None:
    required = ["project", "data", "task", "model"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Missing config sections: {missing}")
    if not cfg["data"].get("input_cols"):
        raise ValueError("data.input_cols must not be empty.")
    if not cfg["data"].get("target_cols"):
        raise ValueError("data.target_cols must not be empty.")
    if int(cfg["task"].get("input_length", 0)) <= 0:
        raise ValueError("task.input_length must be positive.")
    if not cfg["task"].get("horizons"):
        raise ValueError("task.horizons must not be empty.")
