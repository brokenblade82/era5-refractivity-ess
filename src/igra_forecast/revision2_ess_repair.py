"""Isolated numerical repairs for ESS. Never changes legacy model semantics."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, kurtosis
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .revision2_baselines import row_training_mask
from .revision2_data import refractivity_components, sha256_file

FEATURE_NAMES = (
    "temperature_k", "specific_humidity", "geopotential_height_div10000",
    "n_dry", "n_wet", "n_total", "pressure_div1000", "surface_pressure_hpa_div1000",
    "sin_lat", "cos_lat", "sin_lon", "cos_lon", "sin_doy", "cos_doy",
    "sin_utc_hour", "cos_utc_hour", "below_ground", "level_mask",
)
CONFIDENCES = (.50, .60, .70, .80, .90, .95)
MODELS = ("seasonal_mean", "ridge_original", "ridge_repaired", "hgb")


def features64(frame: pd.DataFrame) -> np.ndarray:
    """Same definitions/order as legacy make_row_features; no float32 rounding."""
    time = pd.DatetimeIndex(pd.to_datetime(frame.time, utc=True))
    day, hour = time.dayofyear.to_numpy(float), time.hour.to_numpy(float)
    lat, lon = np.deg2rad(frame.latitude.to_numpy(float)), np.deg2rad(frame.longitude.to_numpy(float))
    result = np.column_stack([
        frame.era5_temperature_k, frame.era5_specific_humidity, frame.era5_height_m / 10000,
        frame.era5_n_dry, frame.era5_n_wet, frame.era5_n, frame.pressure_hpa / 1000,
        frame.era5_surface_pressure_pa / 100000,
        np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon),
        np.sin(2*np.pi*day/365.25), np.cos(2*np.pi*day/365.25),
        np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
        frame.below_ground_level.astype(float), frame.level_mask.astype(float),
    ]).astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite Ridge input features; no evaluation-dependent imputation")
    return result


class RepairedRidge:
    """Training-only near-constant filter followed by fixed standardized Ridge."""
    def fit(self, frame: pd.DataFrame, fingerprint: str) -> "RepairedRidge":
        train = frame.loc[row_training_mask(frame)]
        if len(train) < 2:
            raise ValueError("Ridge requires at least two licensed valid training rows")
        x = features64(train)
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.threshold_ = 1e-12 * np.maximum(1., np.abs(self.mean_))
        self.keep_ = self.std_ > self.threshold_
        if not self.keep_.any():
            raise ValueError("No identifiable Ridge predictors")
        self.minimum_, self.maximum_ = x.min(axis=0), x.max(axis=0)
        self.scaler_ = StandardScaler().fit(x[:, self.keep_])
        xs = self.scaler_.transform(x[:, self.keep_])
        self.models_ = {}
        for component in tqdm(("dry", "wet"), desc="Fit repaired Ridge (fixed alpha=1)"):
            self.models_[component] = Ridge(alpha=1., solver="lsqr").fit(xs, train[f"residual_{component}"].to_numpy(float))
        self.training_fingerprint_ = fingerprint
        self.training_rows_ = len(train)
        return self

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = self.scaler_.transform(features64(frame)[:, self.keep_])
        values = tuple(self.models_[c].predict(x) for c in ("dry", "wet"))
        if not all(np.isfinite(v).all() for v in values):
            raise ValueError("Repaired Ridge produced non-finite values; refusing to clip/drop rows")
        return values

    def audit(self) -> pd.DataFrame:
        result = pd.DataFrame({"feature": FEATURE_NAMES, "mean": self.mean_, "std": self.std_,
                               "threshold": self.threshold_, "retained": self.keep_,
                               "minimum": self.minimum_, "maximum": self.maximum_})
        for component, model in self.models_.items():
            coef = np.zeros(len(FEATURE_NAMES))
            coef[self.keep_] = model.coef_
            result[f"standardized_coefficient_{component}"] = coef
        return result

    def range_audit(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = features64(frame)
        return pd.DataFrame({"feature": FEATURE_NAMES, "retained": self.keep_,
                             "n": len(x), "nonfinite": (~np.isfinite(x)).sum(axis=0),
                             "below_train_min": (x < self.minimum_).sum(axis=0),
                             "above_train_max": (x > self.maximum_).sum(axis=0),
                             "target_min": x.min(axis=0), "target_max": x.max(axis=0)})


def geometric_height(h: np.ndarray, radius: float = 6371000.) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    if np.any(np.isfinite(h) & (h >= radius)):
        raise ValueError("Geopotential height outside spherical conversion domain")
    return radius*h/(radius-h)


def batch_weights(x: np.ndarray, y: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Independent profile interpolation, sorted with masked levels, no extrapolation."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    valid = np.asarray(valid, bool) & np.isfinite(x)
    order = np.argsort(np.where(valid, x, np.inf), axis=1, kind="stable")
    ordered = np.take_along_axis(np.where(valid, x, np.inf), order, axis=1)
    n = valid.sum(axis=1)
    rows = np.arange(len(x))
    if np.any((ordered[:, 1:] == ordered[:, :-1]) & np.isfinite(ordered[:, 1:])):
        raise ValueError("Duplicate finite source heights")
    last = ordered[rows, np.maximum(n-1, 0)]
    support = (n >= 2) & np.isfinite(y) & (y >= ordered[:, 0]) & (y <= last)
    right = np.sum(ordered < y[:, None], axis=1).clip(0, x.shape[1]-1)
    left = np.maximum(right-1, 0)
    a, b = ordered[rows, left], ordered[rows, right]
    fraction = np.zeros(len(x))
    use = support & (b > a)
    numerator, denominator = np.zeros(len(x)), np.ones(len(x))
    np.subtract(y, a, out=numerator, where=use)
    np.subtract(b, a, out=denominator, where=use)
    np.divide(numerator, denominator, out=fraction, where=use)
    fraction[~support] = 0
    weights = np.zeros_like(x)
    active = rows[support]
    weights[active, order[active, left[support]]] += 1-fraction[support]
    weights[active, order[active, right[support]]] += fraction[support]
    return weights, support


def weighted_values(weights: np.ndarray, values: np.ndarray, support: np.ndarray, log: bool = False) -> np.ndarray:
    values = np.asarray(values, float)
    active = weights > 0
    invalid = ~np.isfinite(values) | ((values <= 0) if log else False)
    good = support & ~np.any(active & invalid, axis=1)
    safe = np.where(invalid, 1. if log else 0., values)
    transformed = np.log(safe) if log else safe
    result = np.sum(weights*transformed, axis=1)
    if log:
        result = np.exp(result)
    return np.where(good, result, np.nan)


def propagated_variance(weights: np.ndarray, covariance: np.ndarray, support: np.ndarray) -> np.ndarray:
    used = (weights[:, :, None] * weights[:, None, :]) > 0
    bad = np.any(used & ~np.isfinite(covariance), axis=(1, 2))
    values = np.einsum("bi,bij,bj->b", weights, np.nan_to_num(covariance), weights)
    if np.any(support & ~bad & (values < -1e-10)):
        raise ValueError("Negative propagated variance")
    return np.where(support & ~bad, np.maximum(values, 0), np.nan)


def thermo_attribution(p, to, qo, te, qe) -> dict[str, np.ndarray]:
    def f(t, q):
        return refractivity_components(p, t, q)[2]
    oo, ee, oe, eo = f(to, qo), f(te, qe), f(to, qe), f(te, qo)
    t = .5*((oe-ee)+(oo-eo))
    q = .5*((eo-ee)+(oo-oe))
    closure = t+q-(oo-ee)
    if np.any(np.abs(closure[np.isfinite(closure)]) >= 1e-10):
        raise ValueError("Temperature/humidity attribution failed to close")
    return {"phi_temperature": t, "phi_humidity": q, "thermo_residual": oo-ee, "closure": closure}


def error_budget(d: np.ndarray, w: np.ndarray) -> dict[str, float]:
    d, w = np.asarray(d, float), np.asarray(w, float)
    total = d+w
    covariance = float(np.mean((d-d.mean())*(w-w.mean())))
    cross = float(2*np.mean(d*w))
    result = {"dry_mse": float(np.mean(d*d)), "wet_mse": float(np.mean(w*w)),
              "cross_term": cross, "total_mse": float(np.mean(total*total)),
              "twice_covariance": 2*covariance, "twice_bias_product": float(2*d.mean()*w.mean()),
              "bias_squared": float(total.mean()**2), "centered_variance": float(total.var())}
    result["closure_error"] = result["total_mse"]-result["dry_mse"]-result["wet_mse"]-cross
    return result


def probability_shape(error: np.ndarray, sd: np.ndarray) -> dict[str, Any]:
    error, sd = np.asarray(error, float), np.asarray(sd, float)
    valid = np.isfinite(error) & np.isfinite(sd) & (sd > 0)
    if not valid.all():
        raise ValueError("Invalid probability rows; do not silently change common samples")
    z = error/sd
    mu, spread = float(z.mean()), float(z.std())
    result = {"n": len(z), "z_mean": mu, "z_std": spread, "sd_mean": float(sd.mean()),
              "coverage_90": float(np.mean(np.abs(z) <= norm.ppf(.95))),
              "diagnostic_scale_90": float(np.quantile(np.abs(z), .9)/norm.ppf(.95)),
              "tail_z_gt3": float(np.mean(np.abs(z) > 3)),
              "mean_interval_width_90": float(2*norm.ppf(.95)*sd.mean()),
              "pit_mean": float(norm.cdf(z).mean()), "pit_variance": float(norm.cdf(z).var()),
              "sd_absolute_error_correlation": float(np.corrcoef(sd, np.abs(error))[0, 1]) if sd.std() > 0 and np.abs(error).std() > 0 else np.nan,
              "status": "ok" if spread > 0 else "constant_standardized_error"}
    if spread > 0:
        u = (z-mu)/spread
        result.update({"gaussian_same_mean_sd_tail_gt3": float(norm.cdf(-3, mu, spread)+norm.sf(3, mu, spread)),
                       "u_excess_kurtosis": float(kurtosis(u, fisher=True, bias=True)),
                       "u_tail_gt1p96": float(np.mean(np.abs(u) > 1.96)),
                       "u_tail_gt3": float(np.mean(np.abs(u) > 3))})
        probabilities = (.005, .025, .05, .5, .95, .975, .995)
        for q, value in zip(probabilities, np.quantile(u, probabilities)):
            result[f"u_q{q:g}"] = float(value)
    for confidence in CONFIDENCES:
        result[f"coverage_{int(confidence*100)}"] = float(np.mean(np.abs(z) <= norm.ppf((1+confidence)/2)))
    return result


def utc_support(time: pd.Series) -> np.ndarray:
    utc = pd.DatetimeIndex(pd.to_datetime(time, utc=True))
    hours = utc.hour + utc.minute/60 + utc.second/3600 + utc.microsecond/3.6e9
    remainder = np.mod(hours, 12)
    return np.minimum(remainder, 12-remainder) <= 1


def distance_km(lat, lon, center_lat: float, center_lon: float) -> np.ndarray:
    lat, lon = np.deg2rad(np.asarray(lat, float)), np.deg2rad(np.asarray(lon, float))
    clat, clon = np.deg2rad(center_lat), np.deg2rad(center_lon)
    a = np.sin((lat-clat)/2)**2 + np.cos(lat)*np.cos(clat)*np.sin((lon-clon)/2)**2
    return 6371*2*np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def point_metrics(residual, correction) -> dict[str, float]:
    r, c = np.asarray(residual, float), np.asarray(correction, float)
    if not len(r):
        return {"n": 0, "status": "no_samples"}
    if not (np.isfinite(r).all() and np.isfinite(c).all()):
        raise ValueError("Non-finite prediction on common evaluation sample")
    e = c-r
    return {"n": len(r), "rmse": float(np.sqrt(np.mean(e*e))), "mae": float(np.mean(np.abs(e))),
            "bias": float(e.mean()), "era5_rmse": float(np.sqrt(np.mean(r*r))),
            "rmse_delta": float(np.sqrt(np.mean(e*e))-np.sqrt(np.mean(r*r))),
            "mean_correction": float(c.mean()), "correction_penalty": float(np.mean(c*c)),
            "alignment_term": float(-2*np.mean(r*c)),
            "delta_mse": float(np.mean(e*e)-np.mean(r*r)), "status": "ok"}


def gaussian_metrics(residual, correction, sd) -> dict[str, float]:
    r, c, s = map(lambda x: np.asarray(x, float), (residual, correction, sd))
    if not (np.isfinite(s).all() and (s > 0).all()):
        raise ValueError("Invalid SD on common sample")
    z = (r-c)/s
    a = norm.ppf(.95)
    absolute = np.abs(r-c)
    result = {"nll": float(np.mean(.5*np.log(2*np.pi)+np.log(s)+.5*z*z)),
              "crps": float(np.mean(s*(z*(2*norm.cdf(z)-1)+2*norm.pdf(z)-1/np.sqrt(np.pi)))),
              "interval_score_90": float(np.mean(2*a*s+20*np.maximum(absolute-a*s, 0))),
              "width_90": float(2*a*s.mean()), "predictive_sd": float(s.mean())}
    for confidence in CONFIDENCES:
        result[f"coverage_{int(confidence*100)}"] = float(np.mean(np.abs(z) <= norm.ppf((1+confidence)/2)))
    return result


def paired_bootstrap(frame: pd.DataFrame, cluster: str, corrections: list[str], replicates: int = 10000,
                     seed: int = 42, station_mean: bool = False, minimum_clusters: int = 10) -> pd.DataFrame:
    """All models reuse identical cluster draws. Pooled or station-mean estimand explicit."""
    work = pd.DataFrame({"cluster": frame[cluster].astype(str), "n": 1., "base_sq": np.square(frame.residual_total),
                         "base_abs": np.abs(frame.residual_total), "base_error": -frame.residual_total})
    for i, column in enumerate(corrections):
        error = frame[column].to_numpy(float)-frame.residual_total.to_numpy(float)
        if not np.isfinite(error).all():
            raise ValueError("Non-finite bootstrap model predictions")
        work[f"sq{i}"], work[f"abs{i}"], work[f"error{i}"] = error**2, np.abs(error), error
    grouped = work.groupby("cluster", sort=True).sum()
    count = len(grouped)
    if not count:
        return pd.DataFrame([{"n": 0, "clusters": 0, "status": "no_samples"}])
    n = grouped.n.to_numpy()
    base = np.column_stack([grouped.base_sq, grouped.base_abs, grouped.base_error])
    outputs = []
    rng = np.random.default_rng(seed)
    # Cluster indices, not row samples: small enough even for all 630 stations.
    draws = rng.integers(count, size=(replicates, count)) if count >= minimum_clusters else None
    for i, column in enumerate(tqdm(corrections, desc="Paired cluster bootstrap", leave=False)):
        model = grouped[[f"sq{i}", f"abs{i}", f"error{i}"]].to_numpy()
        def transform(values):
            out = np.array(values, copy=True)
            out[..., 0] = np.sqrt(np.maximum(out[..., 0], 0))
            return out
        if station_mean:
            differences = transform(model/n[:, None])-transform(base/n[:, None])
            estimate = differences.mean(axis=0)
            samples = differences[draws].mean(axis=1) if draws is not None else None
        else:
            estimate = transform(model.sum(axis=0)/n.sum())-transform(base.sum(axis=0)/n.sum())
            samples = None
            if draws is not None:
                samples = np.empty((replicates, 3))
                for start in range(0, replicates, 250):
                    ids = draws[start:start+250]
                    denominator = n[ids].sum(axis=1)[:, None]
                    samples[start:start+len(ids)] = transform(model[ids].sum(axis=1)/denominator)-transform(base[ids].sum(axis=1)/denominator)
        for j, metric in enumerate(("rmse", "mae", "bias")):
            lo, hi = np.quantile(samples[:, j], [.025, .975]) if samples is not None else (np.nan, np.nan)
            outputs.append({"model": column, "metric": metric, "estimate": float(estimate[j]),
                            "ci_lower": lo, "ci_upper": hi, "clusters": count, "n": int(n.sum()),
                            "bootstrap_unit": cluster, "estimand": "station_mean_difference" if station_mean else "pooled_row_difference",
                            "replicates": replicates, "seed": seed, "status": "ok" if draws is not None else "insufficient_clusters"})
    return pd.DataFrame(outputs)


def hierarchical_bootstrap(frame: pd.DataFrame, corrections: list[str], replicates=10000, seed=42, minimum_clusters=10) -> pd.DataFrame:
    work = frame.copy()
    work["month"] = pd.to_datetime(work.time, utc=True).dt.strftime("%Y-%m")
    work["base_sq"] = np.square(work.residual_total)
    columns = []
    for i, c in enumerate(corrections):
        columns.append(f"sq{i}")
        work[columns[-1]] = np.square(work[c]-work.residual_total)
    monthly = work.groupby(["station_id", "month"], sort=True)[["base_sq", *columns]].mean()
    differences = np.sqrt(monthly[columns].to_numpy())-np.sqrt(monthly.base_sq.to_numpy())[:, None]
    monthly = pd.DataFrame(differences, index=monthly.index, columns=corrections)
    by_station = [g.to_numpy() for _, g in monthly.groupby(level=0, sort=True)]
    estimate = np.mean([g.mean(axis=0) for g in by_station], axis=0)
    n_stations = len(by_station)
    samples = np.empty((replicates, len(corrections)))
    rng = np.random.default_rng(seed)
    if n_stations >= minimum_clusters:
        for i in tqdm(range(replicates), desc="Station-then-month bootstrap"):
            ids = rng.integers(n_stations, size=n_stations)
            samples[i] = np.mean([by_station[j][rng.integers(len(by_station[j]), size=len(by_station[j]))].mean(axis=0) for j in ids], axis=0)
    rows = []
    for j, column in enumerate(corrections):
        ci = np.quantile(samples[:, j], [.025, .975]) if n_stations >= minimum_clusters else (np.nan, np.nan)
        rows.append({"model": column, "metric": "rmse", "estimate": estimate[j], "ci_lower": ci[0], "ci_upper": ci[1],
                     "clusters": n_stations, "replicates": replicates, "seed": seed,
                     "estimand": "mean_across_stations_of_mean_monthly_rmse_difference",
                     "bootstrap_unit": "station_then_month", "status": "ok" if n_stations >= minimum_clusters else "insufficient_clusters"})
    return pd.DataFrame(rows)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    os.replace(temporary, path)


def _json_default(value):
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    raise TypeError(type(value).__name__)


def atomic_parquet(frame: pd.DataFrame, path: Path, fingerprint: str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)
    meta = {"complete": True, "fingerprint": fingerprint, "rows": len(frame), "sha256": sha256_file(path)}
    atomic_json(path.with_suffix(".json"), meta)
    return meta


def completed_partition(path: Path, fingerprint: str) -> bool:
    meta_path = path.with_suffix(".json")
    if not path.is_file() or not meta_path.is_file():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("fingerprint") != fingerprint:
        raise ValueError(f"Partition fingerprint mismatch: {path}; choose a new version, never mix runs")
    if not meta.get("complete") or meta.get("sha256") != sha256_file(path):
        raise ValueError(f"Corrupt completed partition: {path}")
    return True
