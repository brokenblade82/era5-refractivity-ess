from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import yaml

from _bootstrap import PROJECT_ROOT  # noqa: F401
from igra_forecast.revision2_data import download_with_resume, ensure_data_directories, load_data_config, month_keys, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Download deterministic COSMIC-2 atmPrf archive days without changing the frozen evaluation design.")
    parser.add_argument("--config", default="configs/revision2/data_pipeline.yaml")
    parser.add_argument("--max-months", type=int, help="Use 1 for a download/schema smoke test.")
    parser.add_argument("--proxy", help="Optional HTTP/HTTPS proxy, for example http://127.0.0.1:7897.")
    parser.add_argument("--days-of-month", type=int, nargs="+", help="Fixed preregistered days, for example 5 15 25.")
    parser.add_argument("--manifest", help="Optional isolated download-manifest path.")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as stream:
        supplied = yaml.safe_load(stream) or {}
    data_config_path = supplied.get("data_config", args.config)
    config = load_data_config(data_config_path)
    paths = ensure_data_directories(config)
    settings = config["cosmic2"]
    days = list(map(int, args.days_of_month or [settings["day_of_month"]]))
    if len(set(days)) != len(days) or any(day < 1 or day > 28 for day in days):
        raise ValueError("--days-of-month must contain unique calendar days between 1 and 28")
    months = month_keys(config["study"]["start"], config["study"]["end"])
    if args.max_months:
        months = months[: args.max_months]
    raw = paths["raw"] / "cosmic2" / settings["stream"] / settings["product"]
    records = []
    for yyyymm in months:
        year, month = int(yyyymm[:4]), int(yyyymm[4:])
        for day_value in days:
            selected = date(year, month, day_value)
            doy = selected.timetuple().tm_yday
            filename = f"{settings['product']}_{settings['stream']}_{year}_{doy:03d}.tar.gz"
            url = f"{settings['base_url'].rstrip('/')}/{settings['stream']}/level2/{year}/{doy:03d}/{filename}"
            target = raw / f"{year}" / filename
            print(f"COSMIC-2 selected date: {selected.isoformat()} (DOY {doy:03d})")
            record = download_with_resume(
                url, target, timeout=int(settings["timeout_seconds"]), retries=int(settings["retries"]),
                progress_desc=filename, proxy=args.proxy,
            )
            record.update({"selected_date": selected.isoformat(), "year_month": yyyymm, "day_of_month": day_value})
            records.append(record)
    manifest_path = Path(args.manifest or paths["manifests"] / "cosmic2_download_manifest.json")
    write_json(
        manifest_path,
        {"selection_rule": f"fixed days {days} of every month", "days_of_month": days, "stream": settings["stream"],
         "product": settings["product"], "months": months, "files": records,
         "expected_file_count": len(months) * len(days),
         "proxy_used": bool(args.proxy),
         "interpretation": "cross-platform evaluation; not assimilation-independent truth"},
    )
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
