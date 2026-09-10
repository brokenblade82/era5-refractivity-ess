from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
import shutil
from pathlib import Path

import fsspec
import xarray as xr
from dask.diagnostics import ProgressBar

from _bootstrap import PROJECT_ROOT  # noqa: F401
from igra_forecast.revision2_data import ensure_data_directories, load_data_config, write_json


EXTERNAL_REQUIREMENTS = Path("configs/revision2/requirements_external.txt")


def require_rapsodi_dependencies() -> dict[str, str]:
    required = ("zarr", "ipfsspec", "fsspec")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        install = f'python -m pip install -r "{EXTERNAL_REQUIREMENTS}"'
        raise RuntimeError(
            "RAPSODI IPFS access is missing required packages: "
            f"{', '.join(missing)}. In the activated torch environment run: {install}"
        )
    engines = xr.backends.list_engines()
    if "zarr" not in engines:
        raise RuntimeError(
            "The zarr package is installed but xarray did not register the zarr backend. "
            f'Reinstall the pinned external dependencies with: python -m pip install --force-reinstall -r "{EXTERNAL_REQUIREMENTS}"'
        )
    return {name: importlib.metadata.version(name) for name in required}


def schema_payload(dataset: xr.Dataset) -> dict[str, object]:
    return {
        "sizes": dict(dataset.sizes),
        "coordinates": {name: {"dims": list(value.dims), "dtype": str(value.dtype), "attrs": dict(value.attrs)} for name, value in dataset.coords.items()},
        "variables": {name: {"dims": list(value.dims), "dtype": str(value.dtype), "attrs": dict(value.attrs)} for name, value in dataset.data_vars.items()},
        "attributes": dict(dataset.attrs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or locally cache the official RAPSODI Level-2 Zarr dataset.")
    parser.add_argument("--config", default="configs/revision2/data_pipeline.yaml")
    parser.add_argument("--schema-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gateway")
    parser.add_argument(
        "--proxy",
        help="Optional HTTP proxy, e.g. http://127.0.0.1:7897. When set, read the CID through the gateway's HTTPS URL instead of direct ipfsspec access.",
    )
    args = parser.parse_args()
    config = load_data_config(args.config)
    dependencies = require_rapsodi_dependencies()
    paths = ensure_data_directories(config)
    settings = config["rapsodi"]
    gateway = args.gateway or settings["gateway_uri"]
    os.environ["IPFS_GATEWAY"] = gateway
    uri = f"ipfs://{settings['level2_cid']}"
    proxy = args.proxy or os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy:
        source_url = f"{gateway.rstrip('/')}/ipfs/{settings['level2_cid']}"
        store = fsspec.get_mapper(source_url, proxy=proxy)
        transport = {"kind": "https_gateway_via_proxy", "source_url": source_url, "proxy": proxy}
    else:
        store = uri
        transport = {"kind": "ipfsspec_direct", "source_url": uri, "proxy": None}
    try:
        dataset = xr.open_dataset(store, engine="zarr", chunks="auto", consolidated=True)
    except Exception as exc:
        raise RuntimeError(
            "All RAPSODI Python dependencies are present, but the remote IPFS Zarr store could not be opened. "
            "This indicates a gateway/network/proxy/remote-store problem rather than a missing package. "
            "If DNS resolves to 198.18.x.x or fdfe:..., pass the local proxy explicitly, for example "
            "--proxy http://127.0.0.1:7897. "
            f"gateway={gateway}, uri={uri}, transport={transport}, original_error={type(exc).__name__}: {exc}"
        ) from exc
    schema_path = paths["manifests"] / "rapsodi_level2_schema.json"
    write_json(schema_path, {
        "uri": uri, "gateway": gateway, "transport": transport,
        "dependencies": dependencies, **schema_payload(dataset),
    })
    if args.schema_only:
        print(f"Schema: {schema_path.resolve()}")
        dataset.close()
        return
    target = Path(settings["local_store"])
    if target.exists():
        if not args.overwrite:
            raise FileExistsError(f"Local RAPSODI store exists; pass --overwrite: {target}")
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Caching RAPSODI Level 2 to {target.resolve()}")
    with ProgressBar():
        dataset.to_zarr(target, mode="w", consolidated=True)
    dataset.close()
    write_json(paths["manifests"] / "rapsodi_download_manifest.json", {
        "uri": uri, "gateway": gateway, "transport": transport,
        "local_store": str(target.resolve()), "dependencies": dependencies,
    })


if __name__ == "__main__":
    main()
