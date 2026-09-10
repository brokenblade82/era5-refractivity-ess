from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


LAYERS = [925, 850, 700]
BASE_FEATURES = ["Temperature_C", "Specific_Humidity_gkg", "U_Wind", "V_Wind", "Refractivity_N"]
MASK_FEATURES = ["U_Wind_missing", "V_Wind_missing"]


@dataclass
class MultiSiteSplit:
    x: np.ndarray
    y: np.ndarray
    station_ids: np.ndarray
    issue_times: np.ndarray
    target_times: np.ndarray
    coords: np.ndarray
    calendar: np.ndarray


@dataclass
class MultiSiteData:
    train: MultiSiteSplit
    val: MultiSiteSplit
    test: MultiSiteSplit
    input_length: int
    horizons: list[int]
    layers: list[int]
    feature_names: list[str]
    target_names: list[str]
    n_feature_scaled: int
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    station_split: dict[str, list[str]]
    station_audit: pd.DataFrame
    summary: dict[str, Any]

    @property
    def splits(self) -> dict[str, MultiSiteSplit]:
        return {"train": self.train, "val": self.val, "test": self.test}


def prepare_multisite_data(cfg: dict[str, Any]) -> MultiSiteData:
    data_cfg = cfg["data"]
    task_cfg = cfg["task"]
    root = Path(data_cfg["path"])
    if not root.exists():
        raise FileNotFoundError(f"Multisite data directory not found: {root}")

    cache_cfg = data_cfg.get("cache", {}) or {}
    cache_enabled = bool(cache_cfg.get("enabled", True))
    cache_path = build_prepared_cache_path(cfg) if cache_enabled else None
    if cache_path is not None and cache_path.exists() and not bool(cache_cfg.get("refresh", False)):
        tqdm.write(f"[IGRA] 直接读取多站预处理缓存：{cache_path}")
        return joblib.load(cache_path)

    layers = [int(v) for v in data_cfg.get("layers", LAYERS)]
    input_length = int(task_cfg.get("input_length", 60))
    horizons = [int(h) for h in task_cfg.get("horizons", [1, 2, 6])]
    include_wind = bool(data_cfg.get("include_wind", True))
    feature_names = ["Temperature_C", "Specific_Humidity_gkg", "Refractivity_N"]
    if include_wind:
        feature_names = ["Temperature_C", "Specific_Humidity_gkg", "U_Wind", "V_Wind", "Refractivity_N"]
    feature_names = feature_names + (MASK_FEATURES if include_wind else [])

    series, station_audit = load_station_series_cached(root, layers, data_cfg)
    eligible = station_audit[station_audit["complete_times"] >= int(data_cfg.get("min_complete_times", 1000))]
    if len(eligible) < 10:
        raise ValueError(f"Too few eligible stations: {len(eligible)}")
    eligible_ids = set(eligible["station_id"])
    series = {sid: value for sid, value in series.items() if sid in eligible_ids}

    station_split = split_stations(sorted(series), data_cfg.get("split", {}), int(cfg["project"].get("seed", 42)))
    raw_splits = {
        name: build_windows_for_stations(
            {sid: series[sid] for sid in ids},
            input_length,
            horizons,
            layers,
            include_wind,
            max_gap_hours=float(data_cfg.get("max_gap_hours", 12.0)),
        )
        for name, ids in station_split.items()
    }
    raw_splits = apply_debug_limits(raw_splits, cfg.get("debug", {}))
    scaled_splits, feature_scaler, target_scaler, n_feature_scaled = scale_splits(raw_splits, include_wind)

    summary = build_summary(root, station_audit, eligible, raw_splits, scaled_splits, station_split, layers, feature_names)
    prepared = MultiSiteData(
        train=scaled_splits["train"],
        val=scaled_splits["val"],
        test=scaled_splits["test"],
        input_length=input_length,
        horizons=horizons,
        layers=layers,
        feature_names=feature_names,
        target_names=[f"N_{layer}" for layer in layers],
        n_feature_scaled=n_feature_scaled,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        station_split=station_split,
        station_audit=eligible.reset_index(drop=True),
        summary=summary,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tqdm.write(f"[IGRA] 保存多站预处理缓存：{cache_path}")
        joblib.dump(prepared, cache_path, compress=3)
    return prepared


def load_station_series_cached(
    root: Path,
    layers: list[int],
    data_cfg: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    cache_cfg = data_cfg.get("cache", {}) or {}
    if not bool(cache_cfg.get("enabled", True)):
        return load_station_series(root, layers, data_cfg)
    cache_dir = Path(cache_cfg.get("dir", "processed/multisite_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = station_series_cache_key(root, layers, data_cfg)
    cache_path = cache_dir / f"station_series_{key}.joblib"
    if cache_path.exists() and not bool(cache_cfg.get("refresh", False)):
        tqdm.write(f"[IGRA] 读取站点标准层缓存：{cache_path}")
        return joblib.load(cache_path)
    out = load_station_series(root, layers, data_cfg)
    tqdm.write(f"[IGRA] 保存站点标准层缓存：{cache_path}")
    joblib.dump(out, cache_path, compress=3)
    return out


def load_station_series(root: Path, layers: list[int], data_cfg: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    files = sorted(root.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {root}")
    allowed_hours = set(int(h) for h in data_cfg.get("allowed_hours", [0, 12]))
    station_series: dict[str, dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []

    for path in tqdm(files, desc="读取多站 IGRA 文件", unit="station", dynamic_ncols=True, smoothing=0.05):
        station_id = path.stem
        df = pd.read_csv(path)
        required = {"Lon", "Lat", "Time", "Pressure_hPa", "Temperature_C", "Specific_Humidity_gkg", "U_Wind", "V_Wind", "Layer_Type", "Refractivity_N"}
        missing = required.difference(df.columns)
        if missing:
            audit_rows.append({"station_id": station_id, "file": path.name, "error": f"missing columns: {sorted(missing)}"})
            continue
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        df = df[df["Time"].notna()]
        df = df[df["Time"].dt.hour.isin(allowed_hours)]
        df = df[df["Layer_Type"].eq("Standard")].copy()
        df["pressure_round"] = pd.to_numeric(df["Pressure_hPa"], errors="coerce").round().astype("Int64")
        df = df[df["pressure_round"].isin(layers)]
        if df.empty:
            audit_rows.append({"station_id": station_id, "file": path.name, "rows": 0, "complete_times": 0})
            continue
        df = df.sort_values(["Time", "pressure_round"]).drop_duplicates(["Time", "pressure_round"], keep="first")

        target_cols = ["Temperature_C", "Specific_Humidity_gkg", "Refractivity_N"]
        complete_flags = (
            df.assign(required_ok=df[target_cols].apply(pd.to_numeric, errors="coerce").notna().all(axis=1))
            .pivot_table(index="Time", columns="pressure_round", values="required_ok", aggfunc="first")
            .reindex(columns=layers)
            .eq(True)
        )
        complete_times = complete_flags.all(axis=1)
        df = df[df["Time"].isin(complete_times[complete_times].index)]

        wide_values: list[np.ndarray] = []
        times: list[pd.Timestamp] = []
        for time, group in df.groupby("Time", sort=True):
            rows = []
            ok = True
            for layer in layers:
                row = group[group["pressure_round"].eq(layer)].head(1)
                if row.empty:
                    ok = False
                    break
                rows.append(row.iloc[0])
            if not ok:
                continue
            frame = pd.DataFrame(rows)
            values = frame[BASE_FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
            values[:, 2:4] = values[:, 2:4]  # wind may contain nan and is handled later
            wide_values.append(values)
            times.append(pd.Timestamp(time))

        if not wide_values:
            complete_count = 0
            values_array = np.empty((0, len(layers), len(BASE_FEATURES)), dtype=np.float32)
        else:
            complete_count = len(wide_values)
            values_array = np.stack(wide_values).astype(np.float32)

        lon = float(pd.to_numeric(df["Lon"], errors="coerce").dropna().iloc[0]) if df["Lon"].notna().any() else np.nan
        lat = float(pd.to_numeric(df["Lat"], errors="coerce").dropna().iloc[0]) if df["Lat"].notna().any() else np.nan
        station_series[station_id] = {"times": np.array(times, dtype="datetime64[ns]"), "values": values_array, "coord": np.array([lat, lon], dtype=np.float32)}
        audit_rows.append(
            {
                "station_id": station_id,
                "file": path.name,
                "rows": int(len(df)),
                "complete_times": int(complete_count),
                "start": str(pd.Timestamp(times[0])) if times else "",
                "end": str(pd.Timestamp(times[-1])) if times else "",
                "lat": lat,
                "lon": lon,
                "wind_missing_frac": float(np.isnan(values_array[:, :, 2:4]).mean()) if values_array.size else np.nan,
            }
        )

    return station_series, pd.DataFrame(audit_rows)


def build_prepared_cache_path(cfg: dict[str, Any]) -> Path:
    data_cfg = cfg["data"]
    cache_cfg = data_cfg.get("cache", {}) or {}
    cache_dir = Path(cache_cfg.get("dir", "processed/multisite_cache"))
    key = prepared_cache_key(cfg)
    return cache_dir / f"prepared_multisite_{key}.joblib"


def prepared_cache_key(cfg: dict[str, Any]) -> str:
    data_cfg = cfg["data"]
    task_cfg = cfg["task"]
    root = Path(data_cfg["path"])
    payload = {
        "data_signature": data_directory_signature(root),
        "layers": [int(v) for v in data_cfg.get("layers", LAYERS)],
        "allowed_hours": [int(v) for v in data_cfg.get("allowed_hours", [0, 12])],
        "min_complete_times": int(data_cfg.get("min_complete_times", 1000)),
        "include_wind": bool(data_cfg.get("include_wind", True)),
        "max_gap_hours": float(data_cfg.get("max_gap_hours", 12.0)),
        "split": data_cfg.get("split", {}),
        "seed": int(cfg["project"].get("seed", 42)),
        "input_length": int(task_cfg.get("input_length", 60)),
        "horizons": [int(h) for h in task_cfg.get("horizons", [1, 2, 6])],
        "debug": cfg.get("debug", {}) or {},
        "version": 2,
    }
    return short_hash(payload)


def station_series_cache_key(root: Path, layers: list[int], data_cfg: dict[str, Any]) -> str:
    payload = {
        "data_signature": data_directory_signature(root),
        "layers": [int(v) for v in layers],
        "allowed_hours": [int(v) for v in data_cfg.get("allowed_hours", [0, 12])],
        "version": 2,
    }
    return short_hash(payload)


def data_directory_signature(root: Path) -> list[dict[str, int | str]]:
    files = sorted(root.glob("*.csv"))
    return [
        {
            "name": path.name,
            "size": int(path.stat().st_size),
            "mtime_ns": int(path.stat().st_mtime_ns),
        }
        for path in files
    ]


def short_hash(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def split_stations(station_ids: list[str], split_cfg: dict[str, float], seed: int) -> dict[str, list[str]]:
    train_frac = float(split_cfg.get("train", 0.7))
    val_frac = float(split_cfg.get("val", 0.1))
    rng = np.random.default_rng(seed)
    ids = np.array(station_ids)
    rng.shuffle(ids)
    n = len(ids)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    if train_end <= 0 or val_end <= train_end or val_end >= n:
        raise ValueError("Invalid station split.")
    return {
        "train": sorted(ids[:train_end].tolist()),
        "val": sorted(ids[train_end:val_end].tolist()),
        "test": sorted(ids[val_end:].tolist()),
    }


def build_windows_for_stations(
    station_series: dict[str, dict[str, Any]],
    input_length: int,
    horizons: list[int],
    layers: list[int],
    include_wind: bool,
    max_gap_hours: float,
) -> MultiSiteSplit:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    station_ids: list[str] = []
    issue_times: list[np.datetime64] = []
    target_times: list[list[np.datetime64]] = []
    coords: list[np.ndarray] = []
    calendars: list[np.ndarray] = []
    max_horizon = max(horizons)

    for station_id, item in station_series.items():
        times = item["times"]
        values = item["values"].copy()
        if len(times) < input_length + max_horizon + 1:
            continue
        segments = contiguous_segments(times, max_gap_hours)
        for start_seg, end_seg in segments:
            seg_times = times[start_seg:end_seg]
            seg_values = values[start_seg:end_seg]
            for start in range(0, len(seg_times) - input_length - max_horizon):
                input_end = start + input_length
                issue_idx = input_end - 1
                target_idx = [issue_idx + h for h in horizons]
                x_raw = seg_values[start:input_end].copy()
                y = seg_values[target_idx, :, BASE_FEATURES.index("Refractivity_N")]
                if np.isnan(y).any() or np.isnan(x_raw[:, :, [0, 1, 4]]).any():
                    continue
                if include_wind:
                    u_missing = np.isnan(x_raw[:, :, BASE_FEATURES.index("U_Wind")]).astype(np.float32)
                    v_missing = np.isnan(x_raw[:, :, BASE_FEATURES.index("V_Wind")]).astype(np.float32)
                    x = np.concatenate([x_raw, u_missing[..., None], v_missing[..., None]], axis=-1)
                else:
                    keep = [BASE_FEATURES.index("Temperature_C"), BASE_FEATURES.index("Specific_Humidity_gkg"), BASE_FEATURES.index("Refractivity_N")]
                    x = x_raw[:, :, keep]
                xs.append(x.astype(np.float32))
                ys.append(y.astype(np.float32))
                station_ids.append(station_id)
                issue_times.append(seg_times[issue_idx])
                target_times.append([seg_times[idx] for idx in target_idx])
                coords.append(item["coord"])
                calendars.append(calendar_features(pd.Timestamp(seg_times[issue_idx])))

    if not xs:
        n_features = len(BASE_FEATURES) + len(MASK_FEATURES) if include_wind else 3
        empty_x = np.empty((0, input_length, len(layers), n_features), dtype=np.float32)
        empty_y = np.empty((0, len(horizons), len(layers)), dtype=np.float32)
        return MultiSiteSplit(empty_x, empty_y, np.array([]), np.array([]), np.empty((0, len(horizons))), np.empty((0, 2)), np.empty((0, 4)))

    return MultiSiteSplit(
        x=np.stack(xs).astype(np.float32),
        y=np.stack(ys).astype(np.float32),
        station_ids=np.array(station_ids),
        issue_times=np.array(issue_times),
        target_times=np.array(target_times),
        coords=np.stack(coords).astype(np.float32),
        calendar=np.stack(calendars).astype(np.float32),
    )


def contiguous_segments(times: np.ndarray, max_gap_hours: float) -> list[tuple[int, int]]:
    if len(times) == 0:
        return []
    diffs = np.diff(times).astype("timedelta64[m]").astype(float) / 60.0
    breaks = np.where(diffs > max_gap_hours + 1e-6)[0] + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [len(times)]])
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e > s]


def calendar_features(time: pd.Timestamp) -> np.ndarray:
    doy = float(time.dayofyear)
    hour = float(time.hour)
    return np.array(
        [
            np.sin(2 * np.pi * doy / 366.0),
            np.cos(2 * np.pi * doy / 366.0),
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
        ],
        dtype=np.float32,
    )


def scale_splits(
    splits: dict[str, MultiSiteSplit],
    include_wind: bool,
) -> tuple[dict[str, MultiSiteSplit], StandardScaler, StandardScaler, int]:
    train = splits["train"]
    n_feature_scaled = len(BASE_FEATURES) if include_wind else 3
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    feature_train = train.x[:, :, :, :n_feature_scaled].reshape(-1, n_feature_scaled)
    medians = np.nanmedian(feature_train, axis=0)
    feature_train = np.where(np.isnan(feature_train), medians, feature_train)
    feature_scaler.fit(feature_train)
    target_scaler.fit(train.y.reshape(-1, 1))

    out: dict[str, MultiSiteSplit] = {}
    for name, split in splits.items():
        x = split.x.copy()
        feature = x[:, :, :, :n_feature_scaled].reshape(-1, n_feature_scaled)
        feature = np.where(np.isnan(feature), medians, feature)
        feature = feature_scaler.transform(feature).reshape(x[:, :, :, :n_feature_scaled].shape)
        x[:, :, :, :n_feature_scaled] = feature.astype(np.float32)
        y = target_scaler.transform(split.y.reshape(-1, 1)).reshape(split.y.shape).astype(np.float32)
        out[name] = MultiSiteSplit(x, y, split.station_ids, split.issue_times, split.target_times, split.coords, split.calendar)
    return out, feature_scaler, target_scaler, n_feature_scaled


def inverse_multisite_targets(data: MultiSiteData, values: np.ndarray) -> np.ndarray:
    return data.target_scaler.inverse_transform(values.reshape(-1, 1)).reshape(values.shape)


def multisite_predictions_to_frame(
    data: MultiSiteData,
    split_name: str,
    y_pred_scaled: np.ndarray,
) -> pd.DataFrame:
    split = data.splits[split_name]
    y_true = inverse_multisite_targets(data, split.y)
    y_pred = inverse_multisite_targets(data, y_pred_scaled)
    rows: list[dict[str, Any]] = []
    for i, station_id in enumerate(split.station_ids):
        for h_idx, horizon in enumerate(data.horizons):
            for l_idx, layer in enumerate(data.layers):
                truth = float(y_true[i, h_idx, l_idx])
                pred = float(y_pred[i, h_idx, l_idx])
                rows.append(
                    {
                        "station_id": station_id,
                        "issue_time": pd.Timestamp(split.issue_times[i]),
                        "target_time": pd.Timestamp(split.target_times[i, h_idx]),
                        "horizon_step": int(horizon),
                        "horizon_hours": int(horizon * 12),
                        "layer_hpa": int(layer),
                        "target": f"N_{layer}",
                        "y_true": truth,
                        "y_pred": pred,
                        "error": pred - truth,
                    }
                )
    return pd.DataFrame(rows)


def apply_debug_limits(splits: dict[str, MultiSiteSplit], debug_cfg: dict[str, Any]) -> dict[str, MultiSiteSplit]:
    mapping = {"train": "max_train_samples", "val": "max_val_samples", "test": "max_test_samples"}
    out = {}
    for name, split in splits.items():
        limit = debug_cfg.get(mapping[name])
        if limit is None:
            out[name] = split
            continue
        keep = min(int(limit), len(split.x))
        out[name] = MultiSiteSplit(
            split.x[:keep],
            split.y[:keep],
            split.station_ids[:keep],
            split.issue_times[:keep],
            split.target_times[:keep],
            split.coords[:keep],
            split.calendar[:keep],
        )
    return out


def build_summary(
    root: Path,
    audit: pd.DataFrame,
    eligible: pd.DataFrame,
    raw_splits: dict[str, MultiSiteSplit],
    scaled_splits: dict[str, MultiSiteSplit],
    station_split: dict[str, list[str]],
    layers: list[int],
    feature_names: list[str],
) -> dict[str, Any]:
    return {
        "data_dir": str(root),
        "all_station_files": int(len(audit)),
        "eligible_stations": int(len(eligible)),
        "layers": layers,
        "features": feature_names,
        "station_split_counts": {k: len(v) for k, v in station_split.items()},
        "window_counts": {k: int(len(v.x)) for k, v in scaled_splits.items()},
        "raw_window_counts": {k: int(len(v.x)) for k, v in raw_splits.items()},
        "complete_times_summary": eligible["complete_times"].describe().to_dict() if not eligible.empty else {},
        "wind_missing_summary": eligible["wind_missing_frac"].describe().to_dict() if "wind_missing_frac" in eligible else {},
    }


def save_station_artifacts(data: MultiSiteData, audit_path: Path, split_path: Path) -> None:
    split_rows = []
    for split, stations in data.station_split.items():
        for station in stations:
            split_rows.append({"split": split, "station_id": station})
    pd.DataFrame(split_rows).to_csv(split_path, index=False, encoding="utf-8-sig")
