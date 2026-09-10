"""Read-only, post-hoc checks of existing predictions; never fit or infer a model.

Formal bootstrap is deliberately user-run. All outputs are isolated from source
results and smoke outputs cannot be promoted to formal evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results_mapping/revision2_confirmatory"
OUTPUT = ROOT / "paper_outputs/ess_ars_review/statistics"
LEVELS = [1000, 925, 850, 700, 500, 300]
RUNS = [11, 23, 42, 71, 101]
KEYS = ["station_id", "time", "pressure_hpa"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def write_csv(path, frame):
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def valid_predictions(path, max_profiles=None):
    f = pd.read_parquet(path)
    f["time"] = pd.to_datetime(f["time"])
    if f.duplicated(KEYS).any():
        raise ValueError(f"Duplicate station/time/pressure: {path}")
    if not set(f.pressure_hpa.unique()).issubset(LEVELS):
        raise ValueError("Unexpected pressure level")
    f = f.loc[f.evaluation_mask.eq(True)].copy()
    if not np.isfinite(f[["residual_total", "prediction_total"]].to_numpy()).all():
        raise ValueError("Nonfinite error on evaluation rows")
    if f.empty:
        raise ValueError("No evaluable rows")
    if max_profiles:
        ids = f[["station_id", "time"]].drop_duplicates().sort_values(["station_id", "time"]).head(max_profiles)
        f = f.merge(ids, on=["station_id", "time"], validate="many_to_one")
    return f.sort_values(KEYS).reset_index(drop=True)


def cluster_mean_ci(values, replicates=10000, seed=42):
    """Rows are stations; columns are paired contrasts using identical draws."""
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Empty or nonfinite station values")
    point = x.mean(axis=0)
    if len(x) < 10:
        return point, np.full(x.shape[1], np.nan), np.full(x.shape[1], np.nan)
    rng = np.random.default_rng(seed)
    draws = []
    for start in tqdm(range(0, replicates, 250), desc="Station bootstrap", unit="batch"):
        idx = rng.integers(len(x), size=(min(250, replicates - start), len(x)))
        draws.append(x[idx].mean(axis=1))
    bounds = np.quantile(np.concatenate(draws), [.025, .975], axis=0)
    return point, bounds[0], bounds[1]


def matched_cells(first, second):
    """Equal station-month-level weighting, then equal stations (not pooled RMSE)."""
    tables = []
    for year, frame in [(2024, first), (2025, second)]:
        frame = frame.loc[(frame.time.dt.year == year) & frame.time.dt.month.between(4, 12)].copy()
        frame["month"] = frame.time.dt.month
        frame["era5_squared"] = frame.residual_total ** 2
        frame["hgb_squared"] = (frame.prediction_total - frame.residual_total) ** 2
        grouped = frame.groupby(["station_id", "month", "pressure_hpa"], sort=True)
        t = grouped.agg(n=("time", "size"), era5_mse=("era5_squared", "mean"), hgb_mse=("hgb_squared", "mean"))
        t["delta_rmse"] = np.sqrt(t.hgb_mse) - np.sqrt(t.era5_mse)
        tables.append(t.add_suffix(f"_{year}"))
    cells = tables[0].join(tables[1], how="outer")
    cells["matched"] = cells.n_2024.notna() & cells.n_2025.notna()
    common = cells.loc[cells.matched].copy()
    if common.empty:
        raise ValueError("No common station-month-level cells")
    common["change_2025_minus_2024"] = common.delta_rmse_2025 - common.delta_rmse_2024
    station = common.groupby(level="station_id")[["delta_rmse_2024", "delta_rmse_2025", "change_2025_minus_2024"]].mean()
    station["common_cells"] = common.groupby(level="station_id").size()
    return cells.reset_index(), station.reset_index()


def matched_stage(max_profiles, reps, out):
    paths = [SOURCE / p / "hgb_run42/predictions.parquet" for p in ["spatial_station_disjoint", "space_time_holdout"]]
    frames = []
    for path in tqdm(paths, desc="Read paired periods"):
        f = valid_predictions(path)
        f = f.loc[f.time.dt.month.between(4, 12)]
        if max_profiles:
            ids = f[["station_id", "time"]].drop_duplicates().sort_values(["station_id", "time"]).head(max_profiles)
            f = f.merge(ids, on=["station_id", "time"], validate="many_to_one")
        frames.append(f)
    cells, stations = matched_cells(*frames)
    columns = ["delta_rmse_2024", "delta_rmse_2025", "change_2025_minus_2024"]
    point, lo, hi = cluster_mean_ci(stations[columns], reps)
    summary = pd.DataFrame({"contrast": columns, "estimate": point, "ci_lower": lo, "ci_upper": hi,
                            "stations": len(stations), "unit": "station cluster", "estimand": "equal station mean of equal matched month-level RMSE contrasts",
                            "ci_status": "available" if len(stations) >= 10 else "insufficient_clusters"})
    common_index = pd.MultiIndex.from_frame(cells.loc[cells.matched, ["station_id", "month", "pressure_hpa"]])
    pooled = []
    for year, f in zip([2024, 2025], frames):
        f = f.assign(month=f.time.dt.month)
        keep = pd.MultiIndex.from_frame(f[["station_id", "month", "pressure_hpa"]]).isin(common_index)
        g = f.loc[keep]
        pooled.append({"year": year, "eligible_rows_apr_dec": len(f), "matched_rows": len(g), "excluded_rows": len(f)-len(g),
                       "era5_pooled_rmse": np.sqrt(np.mean(g.residual_total**2)),
                       "hgb_pooled_rmse": np.sqrt(np.mean((g.prediction_total-g.residual_total)**2))})
    return {"matched_cells.csv": cells, "matched_station_metrics.csv": stations,
            "matched_season_summary.csv": summary, "matched_pooled_audit.csv": pd.DataFrame(pooled)}


def masked_nll(errors, covariance, mask):
    """Analytical log score; no jitter or silent covariance repairs."""
    e, c, mask = np.asarray(errors), np.asarray(covariance), np.asarray(mask, dtype=bool)
    scores = np.full(len(e), np.nan)
    if not mask.any(axis=1).all():
        raise ValueError("Profile without valid levels")
    for pattern in np.unique(mask, axis=0):
        ids = np.flatnonzero(np.all(mask == pattern, axis=1))
        levels = np.flatnonzero(pattern)
        for start in range(0, len(ids), 2048):
            idx = ids[start:start+2048]
            sub = c[idx][:, levels][:, :, levels]
            err = e[idx][:, levels]
            if not np.isfinite(sub).all() or not np.isfinite(err).all():
                raise ValueError("Nonfinite valid covariance/error")
            if not np.allclose(sub, sub.transpose(0, 2, 1), rtol=0, atol=1e-12):
                raise ValueError("Asymmetric covariance")
            factor = np.linalg.cholesky(sub)
            z = np.linalg.solve(factor, err[..., None])[..., 0]
            scores[idx] = .5 * (len(levels)*np.log(2*np.pi) + 2*np.log(np.diagonal(factor, axis1=1, axis2=2)).sum(1) + (z*z).sum(1))
    return scores


def profile_scores(frame, cov):
    if cov.duplicated(["station_id", "time"]).any():
        raise ValueError("Duplicate covariance profile")
    values = frame.assign(error=frame.prediction_total-frame.residual_total).pivot(index=["station_id", "time"], columns="pressure_hpa", values="error").reindex(columns=LEVELS)
    indexed = cov.set_index(["station_id", "time"])
    if not values.index.isin(indexed.index).all():
        raise ValueError("Missing covariance profiles")
    indexed = indexed.reindex(values.index)
    c = np.empty((len(values), 6, 6))
    for i, left in enumerate(LEVELS):
        for j in range(i, 6):
            c[:, i, j] = c[:, j, i] = indexed[f"cov_{left}_{LEVELS[j]}"].to_numpy()
    mask = np.isfinite(values.to_numpy())
    # Stored mask bits are checked against row-level evaluation masks.
    bits = (mask * (1 << np.arange(6))).sum(1)
    if not np.array_equal(bits, indexed.mask_bits.to_numpy()):
        raise ValueError("Covariance mask differs from evaluation rows")
    result = values.index.to_frame(index=False)
    result["nll"] = masked_nll(values.to_numpy(), c, mask)
    result["valid_levels"] = mask.sum(1)
    return result


def paired_probability_stage(max_profiles, reps, out):
    station_tables, pooled, draws = [], [], []
    frozen = valid_predictions(SOURCE / "space_time_holdout/hgb_run42/predictions.parquet", max_profiles)
    for seed in tqdm(RUNS, desc="Paired probability runs", unit="run"):
        pair = []
        for variant in ["diag", "structured"]:
            folder = SOURCE / "space_time_holdout" / f"hgb_prob_hetero_{variant}_run{seed}"
            f = valid_predictions(folder / "predictions.parquet", max_profiles)
            if not f[KEYS].equals(frozen[KEYS]):
                raise ValueError("Unpaired evaluation samples")
            if not np.allclose(f.prediction_total, frozen.prediction_total, rtol=0, atol=1e-10) or not np.allclose(f.residual_total, frozen.residual_total, rtol=0, atol=1e-10):
                raise ValueError("Frozen means or references differ")
            cov = pd.read_parquet(folder / "profile_covariance.parquet")
            cov["time"] = pd.to_datetime(cov.time)
            score = profile_scores(f, cov)
            old = pd.read_csv(folder / "joint_probability_metrics.csv").iloc[0]
            if not max_profiles and not np.isclose(score.nll.mean(), old.multivariate_gaussian_nll, atol=1e-8, rtol=0):
                raise ValueError("Analytic NLL does not reproduce frozen pooled score")
            pooled.append({"run": seed, "variant": variant, "profiles": len(score), "pooled_nll": score.nll.mean(),
                           "frozen_pooled_nll": old.multivariate_gaussian_nll, "frozen_energy_score": old.profile_energy_score})
            pair.append(score.rename(columns={"nll": variant}))
        joined = pair[0].merge(pair[1], on=["station_id", "time", "valid_levels"], validate="one_to_one")
        joined["delta_nll"] = joined.structured - joined.diag
        station = joined.groupby("station_id", sort=True).agg(delta_nll=("delta_nll", "mean"), profiles=("time", "size")).reset_index()
        station["run"] = seed
        station_tables.append(station)
    stations = pd.concat(station_tables, ignore_index=True)
    wide = stations.pivot(index="station_id", columns="run", values="delta_nll").reindex(columns=RUNS)
    if wide.isna().any().any():
        raise ValueError("Station sets differ between runs")
    wide["run_average"] = wide.mean(axis=1)
    point, lo, hi = cluster_mean_ci(wide, reps)
    summary = pd.DataFrame({"run": list(wide.columns), "estimate": point, "ci_lower": lo, "ci_upper": hi,
                            "stations": len(wide), "unit": "station cluster", "estimand": "equal station mean profile NLL difference: structured minus diagonal",
                            "ci_status": "available" if len(wide) >= 10 else "insufficient_clusters"})
    return {"probability_station_metrics.csv": stations, "probability_pooled_reproduction.csv": pd.DataFrame(pooled), "paired_nll_summary.csv": summary}


def stage_inputs(stage):
    if stage == "matched-season":
        return [SOURCE / p / "hgb_run42/predictions.parquet" for p in ["spatial_station_disjoint", "space_time_holdout"]]
    paths = [SOURCE / "space_time_holdout/hgb_run42/predictions.parquet"]
    for seed in RUNS:
        for variant in ["diag", "structured"]:
            folder = SOURCE / "space_time_holdout" / f"hgb_prob_hetero_{variant}_run{seed}"
            paths.extend(folder / n for n in ["predictions.parquet", "profile_covariance.parquet", "joint_probability_metrics.csv", "protocol_manifest.json"])
    return paths


def fingerprint(paths):
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def verify_receipt(receipt):
    for section in ["inputs", "outputs"]:
        for name, digest in receipt[section].items():
            if sha(ROOT / name) != digest:
                raise ValueError(f"Changed {section}: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["matched-season", "paired-probability", "summarize"], required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-profiles", type=int)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_profiles is not None and (not args.smoke or args.max_profiles < 1):
        parser.error("--max-profiles requires --smoke and must be positive")
    if args.bootstrap_replicates < 1:
        parser.error("Positive bootstrap count required")
    if not args.smoke and args.bootstrap_replicates != 10000:
        parser.error("Formal analysis is fixed to 10000 replicates")
    reps = min(args.bootstrap_replicates, 200) if args.smoke else 10000
    max_profiles = (args.max_profiles or 10) if args.smoke else None
    out = OUTPUT / ("smoke" if args.smoke else "formal")
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "summarize":
        receipts = []
        for stage in ["matched-season", "paired-probability"]:
            receipt = json.loads((out / f"{stage}_manifest.json").read_text(encoding="utf-8"))
            verify_receipt(receipt)
            if receipt["smoke"] != args.smoke or receipt["code_sha256"] != sha(Path(__file__)):
                raise ValueError("Stale implementation or mixed smoke/formal receipts")
            receipts.append(receipt)
        atomic_json(out / "evidence_manifest.json", {"status": "smoke_only" if args.smoke else "posthoc_evidence_complete_not_submission_ready", "stages": receipts,
            "no_training": True, "no_new_inference": True, "interpretation": "post-hoc sensitivity; not model selection; no automatic manuscript numeric updates"})
        print(f"Evidence verified: {out}")
        return
    signature = {"stage": args.stage, "smoke": args.smoke, "max_profiles": max_profiles, "bootstrap_replicates": reps, "random_state": 42, "code_sha256": sha(Path(__file__))}
    receipt_path = out / f"{args.stage}_manifest.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not args.resume:
            raise ValueError("Outputs exist; use --resume. Never silently overwrite a completed analysis.")
        if any(receipt.get(k) != v for k, v in signature.items()):
            raise ValueError("Resume configuration/code mismatch")
        verify_receipt(receipt)
        print(f"Reused verified stage: {args.stage}")
        return
    paths = stage_inputs(args.stage)
    inputs = fingerprint(paths)
    result = (matched_stage if args.stage == "matched-season" else paired_probability_stage)(max_profiles, reps, out)
    if inputs != fingerprint(paths):
        raise ValueError("Source inputs changed during analysis")
    for name, frame in result.items():
        write_csv(out / name, frame)
    atomic_json(receipt_path, {**signature, "inputs": inputs, "outputs": fingerprint([out/n for n in result]),
                              "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__})
    print(f"Completed {args.stage}: {out}")


if __name__ == "__main__":
    main()
