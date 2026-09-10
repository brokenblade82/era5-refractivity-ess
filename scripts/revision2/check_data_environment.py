from __future__ import annotations

import argparse
import importlib
import platform
import sys
from pathlib import Path

from _bootstrap import PROJECT_ROOT  # noqa: F401
from igra_forecast.revision2_data import ensure_data_directories, load_data_config


REQUIRED = ["numpy", "pandas", "pyarrow", "xarray", "netCDF4", "yaml", "requests", "tqdm"]
OPTIONAL = ["dask", "zarr", "fsspec", "ipfsspec", "h5netcdf"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the revision-2 data environment.")
    parser.add_argument("--config", default="configs/revision2/data_pipeline.yaml")
    args = parser.parse_args()
    config = load_data_config(args.config)
    paths = ensure_data_directories(config)
    missing = []
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    for name in REQUIRED + OPTIONAL:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "available")
            print(f"[OK] {name}: {version}")
        except ImportError:
            label = "required" if name in REQUIRED else "optional-external-data"
            print(f"[MISSING:{label}] {name}")
            if name in REQUIRED:
                missing.append(name)
    for key, path in paths.items():
        print(f"[PATH] {key}: {Path(path).resolve()}")
    if missing:
        raise SystemExit(f"Missing required packages: {', '.join(missing)}")


if __name__ == "__main__":
    main()
