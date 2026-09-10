from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def save_pickle(obj: Any, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "model.pkl").open("wb") as f:
        pickle.dump(obj, f)
