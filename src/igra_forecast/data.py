from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class SplitArrays:
    frame: pd.DataFrame
    x: np.ndarray
    y: np.ndarray
    issue_times: pd.DatetimeIndex
    target_times: np.ndarray
    calendar_features: np.ndarray
    climatology_prior: np.ndarray
    persistence_prior: np.ndarray


@dataclass
class ExperimentData:
    raw: pd.DataFrame
    scaled: pd.DataFrame
    train: SplitArrays
    val: SplitArrays
    test: SplitArrays
    input_cols: list[str]
    target_cols: list[str]
    horizons: list[int]
    input_length: int
    target_scaler: StandardScaler | MinMaxScaler | None
    feature_scaler: StandardScaler | MinMaxScaler | None
    summary: dict[str, Any]

    @property
    def splits(self) -> dict[str, SplitArrays]:
        return {"train": self.train, "val": self.val, "test": self.test}


def prepare_experiment_data(cfg: dict[str, Any]) -> ExperimentData:
    data_cfg = cfg["data"]
    task_cfg = cfg["task"]
    path = Path(data_cfg["path"])
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    datetime_col = data_cfg["datetime_col"]
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col).drop_duplicates(datetime_col).reset_index(drop=True)
    df = _apply_scenario(df, datetime_col, data_cfg)

    input_cols = list(data_cfg["input_cols"])
    target_cols = list(data_cfg["target_cols"])
    _validate_columns(df, [datetime_col, *input_cols, *target_cols])

    split_frames = _chronological_split(df, data_cfg["split"])
    scaled_frames, feature_scaler, target_scaler = _scale_splits(split_frames, input_cols, target_cols, data_cfg)

    input_length = int(task_cfg["input_length"])
    horizons = [int(h) for h in task_cfg["horizons"]]
    climatology, climatology_global = _build_climatology_prior(scaled_frames["train"], target_cols, datetime_col)
    splits = {
        name: _make_windows(
            frame,
            input_cols,
            target_cols,
            input_length,
            horizons,
            datetime_col,
            climatology,
            climatology_global,
        )
        for name, frame in scaled_frames.items()
    }
    _apply_debug_sample_limits(splits, cfg.get("debug", {}))

    summary = _build_summary(df, split_frames, input_cols, target_cols, input_length, horizons, data_cfg)
    return ExperimentData(
        raw=df,
        scaled=pd.concat(scaled_frames.values(), axis=0).sort_values(datetime_col).reset_index(drop=True),
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
        input_cols=input_cols,
        target_cols=target_cols,
        horizons=horizons,
        input_length=input_length,
        target_scaler=target_scaler,
        feature_scaler=feature_scaler,
        summary=summary,
    )


def _apply_debug_sample_limits(splits: dict[str, SplitArrays], debug_cfg: dict[str, Any]) -> None:
    mapping = {
        "train": "max_train_samples",
        "val": "max_val_samples",
        "test": "max_test_samples",
    }
    for split_name, cfg_name in mapping.items():
        limit = debug_cfg.get(cfg_name)
        if limit is None:
            continue
        limit = int(limit)
        if limit <= 0:
            raise ValueError(f"debug.{cfg_name} must be positive when set.")
        split = splits[split_name]
        keep = min(limit, len(split.x))
        split.x = split.x[:keep]
        split.y = split.y[:keep]
        split.issue_times = split.issue_times[:keep]
        split.target_times = split.target_times[:keep]
        split.calendar_features = split.calendar_features[:keep]
        split.climatology_prior = split.climatology_prior[:keep]
        split.persistence_prior = split.persistence_prior[:keep]


def inverse_targets(data: ExperimentData, values: np.ndarray) -> np.ndarray:
    if data.target_scaler is None:
        return values
    original_shape = values.shape
    flat = values.reshape(-1, len(data.target_cols))
    restored = data.target_scaler.inverse_transform(flat)
    return restored.reshape(original_shape)


def predictions_to_frame(
    data: ExperimentData,
    split: str,
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
) -> pd.DataFrame:
    split_data = data.splits[split]
    y_true = inverse_targets(data, y_true_scaled)
    y_pred = inverse_targets(data, y_pred_scaled)
    records: list[dict[str, Any]] = []
    for sample_idx, issue_time in enumerate(split_data.issue_times):
        for horizon_idx, horizon in enumerate(data.horizons):
            target_time = pd.Timestamp(split_data.target_times[sample_idx, horizon_idx])
            for target_idx, target in enumerate(data.target_cols):
                truth = float(y_true[sample_idx, horizon_idx, target_idx])
                pred = float(y_pred[sample_idx, horizon_idx, target_idx])
                records.append(
                    {
                        "issue_time": issue_time,
                        "target_time": target_time,
                        "split": split,
                        "horizon": int(horizon),
                        "target": target,
                        "y_true": truth,
                        "y_pred": pred,
                        "error": pred - truth,
                    }
                )
    return pd.DataFrame.from_records(records)


def _apply_scenario(df: pd.DataFrame, datetime_col: str, data_cfg: dict[str, Any]) -> pd.DataFrame:
    scenario = data_cfg.get("scenario", "full")
    if scenario == "full":
        return df
    start = pd.Timestamp(data_cfg["scenario_date_range"]["start"])
    end = pd.Timestamp(data_cfg["scenario_date_range"]["end"])
    in_range = df[datetime_col].between(start, end)
    if scenario == "drop_2004_2007":
        return df.loc[~in_range].reset_index(drop=True)
    if scenario == "mask_2004_2007_windows":
        out = df.copy()
        out["_blocked_for_windows"] = in_range
        return out
    raise ValueError(f"Unknown data scenario: {scenario}")


def _validate_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data file: {missing}")


def _chronological_split(df: pd.DataFrame, split_cfg: dict[str, float]) -> dict[str, pd.DataFrame]:
    n = len(df)
    train_end = int(n * float(split_cfg["train"]))
    val_end = train_end + int(n * float(split_cfg["val"]))
    if train_end <= 0 or val_end <= train_end or val_end >= n:
        raise ValueError("Invalid chronological split configuration.")
    return {
        "train": df.iloc[:train_end].copy().reset_index(drop=True),
        "val": df.iloc[train_end:val_end].copy().reset_index(drop=True),
        "test": df.iloc[val_end:].copy().reset_index(drop=True),
    }


def _make_scaler(method: str) -> StandardScaler | MinMaxScaler:
    if method == "standard":
        return StandardScaler()
    if method == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unsupported scaler method: {method}")


def _scale_splits(
    split_frames: dict[str, pd.DataFrame],
    input_cols: list[str],
    target_cols: list[str],
    data_cfg: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], StandardScaler | MinMaxScaler | None, StandardScaler | MinMaxScaler | None]:
    if not data_cfg.get("scaling", {}).get("enabled", True):
        return split_frames, None, None

    method = data_cfg["scaling"].get("method", "standard")
    feature_scaler = _make_scaler(method)
    target_scaler = _make_scaler(method)
    feature_scaler.fit(split_frames["train"][input_cols])
    target_scaler.fit(split_frames["train"][target_cols])

    scaled: dict[str, pd.DataFrame] = {}
    for name, frame in split_frames.items():
        out = frame.copy()
        out[input_cols] = feature_scaler.transform(frame[input_cols])
        out[target_cols] = target_scaler.transform(frame[target_cols])
        scaled[name] = out
    return scaled, feature_scaler, target_scaler


def _make_windows(
    frame: pd.DataFrame,
    input_cols: list[str],
    target_cols: list[str],
    input_length: int,
    horizons: list[int],
    datetime_col: str,
    climatology: pd.DataFrame,
    climatology_global: np.ndarray,
) -> SplitArrays:
    max_horizon = max(horizons)
    x_values = frame[input_cols].to_numpy(dtype=np.float32)
    y_values = frame[target_cols].to_numpy(dtype=np.float32)
    times = pd.DatetimeIndex(frame[datetime_col])
    blocked = frame["_blocked_for_windows"].to_numpy(dtype=bool) if "_blocked_for_windows" in frame else None

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    issue_times: list[pd.Timestamp] = []
    target_times: list[list[pd.Timestamp]] = []
    calendar_features: list[np.ndarray] = []
    climatology_priors: list[np.ndarray] = []
    persistence_priors: list[np.ndarray] = []
    target_input_indices = [input_cols.index(col) for col in target_cols]
    for start in range(0, len(frame) - input_length - max_horizon + 1):
        input_end = start + input_length
        issue_idx = input_end - 1
        target_indices = [issue_idx + h for h in horizons]
        all_indices = list(range(start, input_end)) + target_indices
        if blocked is not None and blocked[all_indices].any():
            continue
        xs.append(x_values[start:input_end])
        ys.append(y_values[target_indices])
        issue_times.append(times[issue_idx])
        sample_target_times = [times[idx] for idx in target_indices]
        target_times.append(sample_target_times)
        calendar_features.append(_calendar_features(sample_target_times))
        climatology_priors.append(_lookup_climatology(sample_target_times, climatology, climatology_global))
        last_target = x_values[issue_idx, target_input_indices]
        persistence_priors.append(np.repeat(last_target[None, :], len(horizons), axis=0))

    if not xs:
        raise ValueError("No valid sliding-window samples were generated.")
    return SplitArrays(
        frame=frame,
        x=np.stack(xs),
        y=np.stack(ys),
        issue_times=pd.DatetimeIndex(issue_times),
        target_times=np.array(target_times, dtype="datetime64[ns]"),
        calendar_features=np.stack(calendar_features).astype(np.float32),
        climatology_prior=np.stack(climatology_priors).astype(np.float32),
        persistence_prior=np.stack(persistence_priors).astype(np.float32),
    )


def _build_climatology_prior(
    train_frame: pd.DataFrame,
    target_cols: list[str],
    datetime_col: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    train = train_frame.copy()
    train["_dayofyear"] = pd.to_datetime(train[datetime_col]).dt.dayofyear
    climatology = train.groupby("_dayofyear")[target_cols].mean()
    global_mean = train[target_cols].mean().to_numpy(dtype=np.float32)
    return climatology, global_mean


def _lookup_climatology(
    times: list[pd.Timestamp],
    climatology: pd.DataFrame,
    global_mean: np.ndarray,
) -> np.ndarray:
    values = []
    for ts in times:
        key = pd.Timestamp(ts).dayofyear
        if key in climatology.index:
            values.append(climatology.loc[key].to_numpy(dtype=np.float32))
        else:
            values.append(global_mean)
    return np.stack(values)


def _calendar_features(times: list[pd.Timestamp]) -> np.ndarray:
    features = []
    for ts in times:
        day = pd.Timestamp(ts).dayofyear
        angle = 2.0 * np.pi * (day - 1) / 366.0
        features.append([np.sin(angle), np.cos(angle)])
    return np.asarray(features, dtype=np.float32)


def _build_summary(
    df: pd.DataFrame,
    split_frames: dict[str, pd.DataFrame],
    input_cols: list[str],
    target_cols: list[str],
    input_length: int,
    horizons: list[int],
    data_cfg: dict[str, Any],
) -> dict[str, Any]:
    datetime_col = data_cfg["datetime_col"]
    return {
        "source_path": data_cfg["path"],
        "scenario": data_cfg.get("scenario", "full"),
        "rows": int(len(df)),
        "start": str(df[datetime_col].min()),
        "end": str(df[datetime_col].max()),
        "missing_values": int(df[[*input_cols, *target_cols]].isna().sum().sum()),
        "input_cols": input_cols,
        "target_cols": target_cols,
        "input_length": input_length,
        "horizons": horizons,
        "splits": {
            name: {
                "rows": int(len(frame)),
                "start": str(frame[datetime_col].min()),
                "end": str(frame[datetime_col].max()),
            }
            for name, frame in split_frames.items()
        },
    }
