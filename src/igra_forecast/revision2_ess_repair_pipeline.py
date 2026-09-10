"""File orchestration for the isolated ESS repair evidence package.

Legacy scripts are imported only for read-only sampling/prediction helpers.
No legacy main(), writer, training function, or global plotting entry is called.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xarray as xr
from scipy.stats import norm
from tqdm import tqdm

from .revision2_baselines import LEVELS, make_row_features, rows_to_profiles
from .revision2_data import load_data_config, require_era5_audit, resolve_era5_root, sha256_file, stable_fingerprint
from .revision2_ess import (load_yaml, build_seasonal_lookup, validate_seasonal_reconstruction,
    training_wet_edges, add_common_groups, predict_row_model, predict_structured_hgb)
from .revision2_publication import validate_geometric_height_schema
from .revision2_ess_repair import (MODELS, RepairedRidge, geometric_height, batch_weights, weighted_values,
    propagated_variance, thermo_attribution, error_budget, probability_shape, utc_support, distance_km,
    point_metrics, gaussian_metrics, paired_bootstrap, hierarchical_bootstrap,
    atomic_json, atomic_parquet, completed_partition)

PROTOCOLS = ("spatial_station_disjoint", "temporal_holdout", "space_time_holdout")
STAGES = ("freeze", "fit-ridge", "cache-cosmic", "evaluate", "bootstrap", "diagnose", "summarize", "validate")
ROW_KEYS = ["station_id", "time", "pressure_hpa"]
RO_KEYS = ["profile_id", "time", "height_m", "source_archive"]


def take_profiles(frame, maximum, key="station_id"):
    if maximum is None:
        return frame.reset_index(drop=True)
    keys = frame[[key, "time"]].drop_duplicates().sort_values([key, "time"]).head(maximum)
    return frame.merge(keys, on=[key, "time"], how="inner", validate="many_to_one").reset_index(drop=True)


def safe_output(path: Path, allowed: Path) -> Path:
    path, allowed = path.resolve(), allowed.resolve()
    if path != allowed and allowed not in path.parents:
        raise ValueError(f"Output must stay inside isolated root: {path}")
    return path


class RepairPipeline:
    def __init__(self, root, args):
        self.project = Path(root).resolve()
        self.args = args
        self.cfg_path = (self.project / args.config).resolve()
        self.cfg = load_yaml(self.cfg_path)
        self.ess = load_yaml(self.project / self.cfg["ess_config"])
        self.confirm = load_yaml(self.project / self.cfg["confirmatory_config"])
        if any(self.cfg["boundaries"][k] for k in ("train_hgb", "train_neural_models", "external_recalibration", "download_observations", "generate_global_products")):
            raise ValueError("Frozen repair boundaries violated")
        if self.cfg["ridge"] != {"alpha": 1., "solver": "lsqr", "near_constant_relative_threshold": 1e-12, "feature_hour_definition": "integer_hour_unchanged"}:
            raise ValueError("Ridge repair specification is fixed; no parameter search")
        if self.cfg["height"]["variants"] != ["A", "B", "C"] or self.cfg["height"]["primary_variant"] != "B":
            raise ValueError("Height variants and primary B are fixed")
        if self.cfg["height"]["radius_m"] != 6371000. or self.cfg["height"]["g0"] != 9.80665:
            raise ValueError("Height constants are fixed")
        if self.cfg["matching"] != {"radius_km": 1000., "synoptic_tolerance_hours": 1.}:
            raise ValueError("Matching rules are fixed, not selected from evaluation results")
        if self.cfg["statistics"] != {"replicates": 10000, "seed": 42, "minimum_clusters": 10,
                "closure_tolerance": 1e-10, "legacy_reproduction_tolerance": 1e-6}:
            raise ValueError("Statistical specification is fixed")
        self.smoke = bool(args.smoke)
        if args.max_profiles is not None and not self.smoke:
            raise ValueError("--max-profiles requires --smoke; partial data cannot enter formal outputs")
        self.maximum = args.max_profiles or 10 if self.smoke else None
        if self.smoke and not 1 <= self.maximum <= 10:
            raise ValueError("Smoke is limited to 1--10 profiles per source/protocol")
        self.replicates = args.bootstrap_replicates or (200 if self.smoke else self.cfg["statistics"]["replicates"])
        if self.replicates < 1 or (self.smoke and self.replicates > 200) or (not self.smoke and self.replicates != 10000):
            raise ValueError("Formal bootstrap requires 10000 replicates; smoke allows 1--200")
        code = sorted((self.project / "src/igra_forecast").glob("revision2*.py"))
        code += sorted((self.project / "scripts/revision2").glob("*.py"))
        self.code_hashes = {str(p.relative_to(self.project)): sha256_file(p) for p in code}
        self.fp = stable_fingerprint({"config": self.cfg, "ess": self.ess, "code": self.code_hashes,
                                      "smoke": self.smoke, "maximum": self.maximum, "replicates": self.replicates})
        for name in ("results", "paper", "docs"):
            allowed = self.project / {"results": "results_mapping", "paper": "paper_outputs", "docs": "docs"}[name] / "revision2_ess_repair"
            path = safe_output(self.project / self.cfg["outputs"][name], allowed)
            if self.smoke:
                path = path / "smoke" / self.fp[:12]
            setattr(self, name, path)
        self.stats = self.paper / "statistics"
        self.outputs = []
        self.input_records = {}
        self._models = None
        self._ridge = None
        self.scripts = self.project / "scripts/revision2"
        if str(self.scripts) not in sys.path:
            sys.path.insert(0, str(self.scripts))

    def path(self, p):
        return (self.project / p).resolve()

    def read(self, path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def record(self, path):
        path = Path(path)
        self.outputs.append({"path": str(path.relative_to(self.project)), "sha256": sha256_file(path)})

    def json(self, path, value):
        atomic_json(path, value)
        self.record(path)

    def csv(self, name, frame):
        path = self.stats / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not len(frame.columns):
            frame = pd.DataFrame(columns=["status", "reason"])
        temporary = path.with_suffix(".csv.tmp")
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
        self.record(path)

    def parquet(self, path, frame):
        atomic_parquet(frame, path, self.fp)
        self.record(path)
        self.record(path.with_suffix(".json"))

    def reuse(self, path):
        if self.args.resume and completed_partition(path, self.fp):
            self.record(path)
            self.record(path.with_suffix(".json"))
            return True
        return False

    def stage_file(self, stage):
        return self.results / "stages" / f"{stage}.json"

    def require(self, stage):
        p = self.stage_file(stage)
        if not p.is_file():
            raise ValueError(f"Run --stage {stage} first (same smoke/config settings)")
        item = self.read(p)
        if not item.get("complete") or item["fingerprint"] != self.fp:
            raise ValueError(f"Stage/config/code fingerprint mismatch: {p}; do not combine versions")
        for record in item["artifacts"]:
            if sha256_file(self.path(record["path"])) != record["sha256"]:
                raise ValueError(f"Changed stage artifact: {record['path']}")
        return item

    def frozen_models(self):
        if self._models is None:
            self._models = {k: joblib.load(self.path(self.ess["frozen_runs"][k]["path"]) / "model.joblib") for k in ("ridge", "hgb", "hgb_probability")}
            context = joblib.load(self.results / "ridge" / "training_context.joblib")
            self._models["seasonal_mean"] = context["seasonal_mean"]
            self.wet_edges = np.asarray(context["wet_edges"])
        return self._models

    def ridge(self):
        if self._ridge is None:
            self._ridge = joblib.load(self.results / "ridge" / "model.joblib")
        return self._ridge

    def verify_inputs(self, full=False):
        frozen = self.read(self.results / "freeze_manifest.json")
        if frozen["fingerprint"] != self.fp:
            raise ValueError("Repair freeze fingerprint changed; use a new version, not mixed outputs")
        self.input_records = frozen["inputs"]
        for rel, item in tqdm(self.input_records.items(), desc="Verify frozen inputs", leave=False):
            p = self.path(rel)
            stat = p.stat()
            if full or stat.st_size != item["size"] or stat.st_mtime_ns != item["mtime_ns"] or stat.st_size < 1_000_000:
                if sha256_file(p) != item["sha256"]:
                    raise ValueError(f"Frozen input content changed: {rel}")
        return frozen

    def run(self, stage):
        self.outputs = []
        started = time.perf_counter()
        print(f"[ESS repair] {stage}; mode={'SMOKE' if self.smoke else 'FORMAL'}; output={self.results}", flush=True)
        for directory in (self.results, self.stats, self.docs):
            directory.mkdir(parents=True, exist_ok=True)
        if stage != "freeze":
            self.require("freeze")
            self.verify_inputs(full=stage == "validate")
        if self.stage_file(stage).is_file() and stage not in ("validate",):
            self.require(stage)
            if self.args.resume:
                print(f"[ESS repair] Complete matching stage reused: {stage}", flush=True)
                return
        getattr(self, "stage_"+stage.replace("-", "_"))()
        self.json(self.stage_file(stage), {"complete": True, "stage": stage, "fingerprint": self.fp,
            "smoke": self.smoke, "elapsed_seconds": time.perf_counter()-started, "artifacts": list(self.outputs)})
        print(f"[ESS repair] Completed {stage}: {time.perf_counter()-started:.1f}s", flush=True)

    def stage_freeze(self):
        if (self.results / "freeze_manifest.json").exists():
            old = self.read(self.results / "freeze_manifest.json")
            if old["fingerprint"] != self.fp:
                raise ValueError("Existing formal freeze uses another code/config version; do not overwrite it")
        files = {}
        def add(p, expected=None):
            p = self.path(p)
            if not p.is_file():
                raise FileNotFoundError(p)
            actual = sha256_file(p)
            if expected is not None and actual != expected:
                raise ValueError(f"Upstream SHA256 mismatch: {p}")
            stat = p.stat()
            files[str(p.relative_to(self.project))] = {"sha256": actual, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        for p in (self.cfg_path, self.path(self.cfg["ess_config"]), self.path(self.cfg["confirmatory_config"]), self.path(self.cfg["data_config"])):
            add(p)
        for label, item in self.ess["frozen_inputs"].items():
            add(item["path"], item["sha256"])
            for key in ("collocation_manifest", "preparation_manifest"):
                if key in item:
                    add(item[key], item[key+"_sha256"])
            if "platform_audit" in item:
                add(item["platform_audit"])
        for name, item in self.ess["frozen_runs"].items():
            run = self.path(item["path"])
            meta = self.read(run / "protocol_manifest.json")
            if meta.get("smoke") or not meta.get("complete") or meta.get("protocol") != "space_time_holdout":
                raise ValueError("Unlicensed frozen model")
            add(run / "protocol_manifest.json")
            if "model_sha256" in item:
                add(run / "model.joblib", item["model_sha256"])
            if "frozen_prediction_sha256" in item:
                add(run / "predictions.parquet", item["frozen_prediction_sha256"])
        audit = self.read(self.path("data/revision2/manifests/revision2_data_audit.json"))
        if not audit.get("passed") or audit["dataset_version"] != "sixlevel_min3_temporalcoverage_v1" or audit["stations"] != 630:
            raise ValueError("Formal six-level data audit missing/incompatible")
        for p in ("data/revision2/manifests/revision2_data_audit.json", "data/revision2/manifests/era5_archive_manifest.json"):
            add(p)
        licenses = pd.read_csv(self.path(self.ess["protocol_license_matrix"]))
        train = licenses.loc[licenses.protocol.isin(PROTOCOLS) & licenses.phase.eq("train")]
        if len(train) != 3 or len(train[["station_license", "time_start_utc", "time_end_utc_exclusive"]].drop_duplicates()) != 1:
            raise ValueError("Cannot reuse one Ridge: training licenses differ")
        if train.iloc[0].station_license != "train" or train.iloc[0].time_start_utc != "2024-01-01" or train.iloc[0].time_end_utc_exclusive != "2025-01-01":
            raise ValueError("Unexpected training license")
        roots = [self.path(self.ess["input"]), self.path("results_mapping/revision2_confirmatory"),
                 self.path("results_mapping/revision2_ess"), self.path("paper_outputs/revision2_ess"),
                 self.path("paper_outputs/revision2_publication_final/statistics")]
        candidates = sorted({p for root in roots for p in root.rglob("*") if p.is_file() and p.suffix in (".parquet", ".csv", ".json", ".joblib")})
        for p in tqdm(candidates, desc="Freeze evidence SHA256"):
            add(p)
        # Snapshot manuscript files to ensure implementation never edits them.
        for root in ("论文/MDPI_THAP_ACS", "论文/JSTARS_HGB_CRN_Submission", "论文/ESS_ERA5_Refractivity_Submission"):
            for p in self.path(root).glob("*"):
                if p.is_file() and p.suffix in (".tex", ".bib", ".pdf"):
                    add(p)
        self.json(self.results / "freeze_manifest.json", {"fingerprint": self.fp, "smoke": self.smoke,
            "inputs": files, "code": self.code_hashes, "claims": "repair verification and exploratory diagnosis; not retrospective preregistration",
            "height_conversion": "ECMWF spherical approximation; horizontal gravity variations neglected",
            "environments": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "xarray", "pyarrow", "torch")}})
        definitions = [
            ("D01", "N_d=77.6*p/T; N_w=3.73e5*e/T**2", "revision2_data.refractivity_components", "Correct manuscript p-e formula; do not change data"),
            ("D02", "E(e_d*e_w)=Cov(e_d,e_w)+mean(e_d)*mean(e_w)", "error_budget", "Cross term is not covariance alone"),
            ("D03", "HGB auto early stopping uses internal train subset; neural early-val is separate", "checkpoint get_params", "Report by model"),
            ("D04", "Scale regression target log(max(abs(station-OOF error),0.05)); bounds [-5,4]", "revision2_confirmatory._raw_scale_by_variant", "Not direct variance maximum likelihood"),
            ("D05", "2025Q1-calibrated distribution applied retrospectively to 2024 RAPSODI", "protocol_license_matrix", "Not a prospective 2024 forecast"),
            ("D06", "0/12 UTC only makes sin(hour) numerically near-constant", "RepairedRidge", "Remove using train-only threshold; keep off-hour applicability caveat"),
            ("D07", "ERA5 geopotential/g0 is geopotential height, not geometric altitude", "geometric_height", "Keep model features; repair comparison coordinates"),
            ("D08", "Station-then-month point estimate equals average station monthly RMSE differences", "hierarchical_bootstrap", "Never pair with pooled station RMSE estimate"),
            ("D09", "Large raw |z| is not proof of heavy tails after matching mean/variance", "probability_shape", "Use shape diagnostics, no external recalibration"),
        ]
        self.csv("definition_audit.csv", pd.DataFrame(definitions, columns=["issue_id", "actual_definition", "code_source", "manuscript_action"]))
        self.csv("protocol_license_audit.csv", licenses)
        model = joblib.load(self.path(self.ess["frozen_runs"]["hgb"]["path"]) / "model.joblib")
        prob = joblib.load(self.path(self.ess["frozen_runs"]["hgb_probability"]["path"]) / "model.joblib")
        self.json(self.results / "model_definition_audit.json", {"hgb": {k: {"params": v.get_params(), "n_iter": v.n_iter_, "do_early_stopping": v.do_early_stopping_} for k, v in model.items()},
            "probability_keys": list(prob),
            "scale_models": {k: {"kind": v["kind"], "log_scale_bounds": v.get("log_scale_bounds"),
                "params": v["model"].get_params() if "model" in v else None} for k, v in prob["scale_models"].items()},
            "calibration_coverage": prob.get("calibration_coverage"), "calibration_scales": prob.get("calibration_scales"),
            "correlation_total": prob.get("correlation_total"), "correlation_pair_counts": prob.get("correlation_pair_counts"),
            "license_audit": prob.get("license_audit")})

    def read_licensed(self, protocol, phase="evaluation", thermo=False):
        license_table = pd.read_csv(self.path(self.ess["protocol_license_matrix"]))
        license_row = license_table.loc[license_table.protocol.eq(protocol) & license_table.phase.eq(phase)].iloc[0]
        split = pd.read_csv(self.path(self.ess["station_split"]), dtype={"station_id": str})
        allowed = set(split.loc[split.split.eq(license_row.station_license), "station_id"])
        start, end = pd.Timestamp(license_row.time_start_utc, tz="UTC"), pd.Timestamp(license_row.time_end_utc_exclusive, tz="UTC")
        parts = []
        for period in tqdm(pd.period_range(start.tz_localize(None), (end-pd.Timedelta(seconds=1)).tz_localize(None), freq="M"), desc=f"Load licensed {protocol}/{phase}"):
            p = self.path(self.ess["input"]) / f"year={period.year}" / f"month={period.month}" / f"part-{period.strftime('%Y%m')}.parquet"
            f = pd.read_parquet(p)
            f.time = pd.to_datetime(f.time, utc=True)
            f = f.loc[f.station_id.astype(str).isin(allowed) & f.time.ge(start) & f.time.lt(end)]
            if self.smoke:
                if phase == "train":
                    # Smoke needs every pressure level for seasonal fallback. This is
                    # a smoke-only subset, never a change to formal station eligibility.
                    valid = f.level_mask.astype(bool) & ~f.below_ground_level.astype(bool)
                    counts = f.assign(_valid=valid).groupby(["station_id", "time"])._valid.sum()
                    keys = counts.loc[counts.eq(6)].reset_index()[["station_id", "time"]]
                    # Spread the tiny training smoke across stations/times instead
                    # of fitting a single station's first ten soundings. Selection
                    # uses only training identifiers, never evaluation outcomes.
                    indices = np.linspace(0, len(keys)-1, min(len(keys), self.maximum), dtype=int)
                    keys = keys.iloc[indices]
                    f = f.merge(keys, on=["station_id", "time"], validate="many_to_one")
                f = take_profiles(f, self.maximum)
            parts.append(f)
            if self.smoke and not f.empty:
                break
        result = pd.concat(parts, ignore_index=True)
        if result.empty or result.duplicated(ROW_KEYS).any():
            raise ValueError("Empty or duplicated licensed profile rows")
        return result

    def stage_fit_ridge(self):
        train = self.read_licensed("space_time_holdout", "train")
        ridge = RepairedRidge().fit(train, self.fp)
        directory = self.results / "ridge"
        directory.mkdir(parents=True, exist_ok=True)
        for name, value in (("model", ridge), ("training_context", {"seasonal_mean": build_seasonal_lookup(train), "wet_edges": training_wet_edges(train)})):
            path = directory / f"{name}.joblib"
            temporary = path.with_suffix(".joblib.tmp")
            joblib.dump(value, temporary)
            os.replace(temporary, path)
            self.record(path)
        if not self.smoke:
            check = validate_seasonal_reconstruction(build_seasonal_lookup(train), self.path(self.ess["frozen_runs"]["seasonal_mean"]["path"]) / "predictions.parquet")
            self.json(directory / "seasonal_reproduction.json", check)
        self.csv("ridge_training_feature_audit.csv", ridge.audit())
        self.json(directory / "manifest.json", {"training_fingerprint": self.fp, "smoke": self.smoke,
            "training_rows": ridge.training_rows_, "training_profiles": len(train)//6, "alpha": 1., "solver": "lsqr",
            "train_role": "train", "time_start": "2024-01-01", "time_end_exclusive": "2025-01-01",
            "external_or_calibration_rows_used": False, "train_station_count": train.station_id.nunique()})

    def build_cache_chunk(self, target, arrays, static_z, static_lsm, levels):
        import evaluate_cosmic2_height as cosmic
        import evaluate_ess_external_suite as old
        target = target.copy().reset_index(drop=True)
        target["surface_height_m"] = cosmic.static_points(static_z, target)/9.80665
        target["land_sea"] = np.where(cosmic.static_points(static_lsm, target) >= .5, "land", "sea")
        collocated = cosmic.collocate_pressure_profiles(target, arrays, levels)
        six = old._six_level_frame(target, collocated, levels)
        models = self.frozen_models()
        probability = predict_structured_hgb(models["hgb_probability"], six)
        cached = target.copy()
        for key, values in collocated.items():
            name = "era5_geopotential_height_m" if key == "height" else key
            cached[name] = list(values) if values.ndim == 2 else values
        cached["era5_geopotential_m2_s2"] = list(collocated["height"]*9.80665)
        cached["era5_geometric_height_m"] = list(geometric_height(collocated["height"]))
        cached["surface_geopotential_m2_s2"] = target.surface_height_m*9.80665
        cached["surface_geometric_height_m"] = geometric_height(target.surface_height_m)
        cached["input_fingerprint"] = self.fp
        for name in MODELS:
            if name == "ridge_repaired":
                dry, wet = self.ridge().predict(six)
            else:
                lookup = "ridge" if name == "ridge_original" else name
                dry, wet = predict_row_model(lookup, six, models[lookup])
            dry, wet = dry.reshape(-1, 6), wet.reshape(-1, 6)
            cached[f"{name}_dry6"], cached[f"{name}_wet6"] = list(dry), list(wet)
            if name == "hgb":
                if not np.allclose(dry, probability.mean_dry, rtol=0, atol=1e-10) or not np.allclose(wet, probability.mean_wet, rtol=0, atol=1e-10):
                    raise ValueError("Frozen deterministic/probabilistic HGB means differ")
        cached["hgb_covariance6"] = list(probability.covariance_total.reshape(-1, 36))
        if np.linalg.eigvalsh(probability.covariance_total).min() <= 0:
            raise ValueError("Frozen six-level covariance is not positive definite")
        audit = self.ridge().range_audit(six)
        audit["source"] = "cosmic2_sixlevel_cache"
        return cached, audit

    def stage_cache_cosmic(self):
        self.require("fit-ridge")
        import evaluate_cosmic2_height as cosmic
        frozen = self.ess["frozen_inputs"]["cosmic2_height"]
        frame = pd.read_parquet(self.path(frozen["path"]))
        validate_geometric_height_schema(frame)
        frame.time = pd.to_datetime(frame.time, utc=True)
        if frame.duplicated(RO_KEYS).any():
            raise ValueError("Duplicate COSMIC profile-time-height-archive keys")
        meta = self.read(self.path(frozen["preparation_manifest"]))
        if meta.get("dry_pressure_used") is not False or meta.get("output_sha256") != frozen["sha256"]:
            raise ValueError("COSMIC geometric-height manifest mismatch")
        original_rows = len(frame)
        if self.smoke:
            frame = take_profiles(frame, self.maximum, "profile_id")
        frame["period"] = frame.time.dt.strftime("%Y%m")
        dc = load_data_config(self.path(self.cfg["data_config"]))
        root = resolve_era5_root(dc, None)
        archive = self.path("data/revision2/manifests/era5_archive_manifest.json")
        periods = sorted(frame.period.unique())
        require_era5_audit(dc, archive, root, periods, require_full_study=not self.smoke)
        levels = np.asarray(self.ess["cosmic2_height"]["era5_pressure_levels_hpa"])
        all_files, range_tables, preloads = [], [], []
        with xr.open_dataset(self.read(archive)["static"]["path"]) as ds:
            static_z, static_lsm = ds.z.squeeze(drop=True).load(), ds.lsm.squeeze(drop=True).load()
        for period in tqdm(periods, desc="Cache COSMIC months"):
            target = frame.loc[frame.period.eq(period)].drop(columns="period")
            keys = target[["profile_id", "time"]].drop_duplicates().reset_index(drop=True)
            starts = list(range(0, len(keys), int(self.cfg["height"]["chunk_profiles"])))
            pending = []
            for index, start in enumerate(starts):
                p = self.results / "cosmic_cache" / period / f"block_{index:05d}.parquet"
                all_files.append(p)
                if self.reuse(p):
                    audit_path = p.with_suffix(".features.csv")
                    if not audit_path.exists():
                        raise ValueError(f"Completed cache missing feature audit: {audit_path}")
                    range_tables.append(pd.read_csv(audit_path))
                    self.record(audit_path)
                else:
                    pending.append((p, start))
            if not pending:
                continue
            opened = {}
            try:
                for logical in ("temperature", "specific_humidity", "geopotential", "surface_pressure"):
                    opened[logical] = cosmic.open_era5_variable(root, pd.Period(period, freq="M"), dc, logical)
                arrays, preload = cosmic.preload_month_arrays(opened, target, levels)
                preloads.append({"period": period, "audit": preload})
                for p, start in tqdm(pending, desc=f"{period} cache blocks", leave=False):
                    part = target.merge(keys.iloc[start:start+int(self.cfg["height"]["chunk_profiles"])], on=["profile_id", "time"], how="inner", validate="many_to_one")
                    cached, feature_audit = self.build_cache_chunk(part, arrays, static_z, static_lsm, levels)
                    feature_audit["block"] = str(p.relative_to(self.results))
                    audit_path = p.with_suffix(".features.csv")
                    audit_path.parent.mkdir(parents=True, exist_ok=True)
                    feature_audit.to_csv(audit_path, index=False)
                    self.record(audit_path)
                    self.parquet(p, cached)
                    range_tables.append(feature_audit)
                    if len(all_files) == 1 or (self.smoke and len(pending) == 1):
                        bytes_per_row = p.stat().st_size/max(1, len(cached))
                        estimate = int(bytes_per_row*original_rows)
                        free = shutil.disk_usage(self.results).free
                        self.json(self.results / "disk_estimate.json", {"cache_bytes_per_row": bytes_per_row,
                            "estimated_full_cache_bytes": estimate, "available_bytes": free,
                            "note": "Cache only; predictions and evidence add disk usage. Estimate measured from first block."})
                        print(f"[ESS repair] Estimated full cache: {estimate/1024**3:.2f} GiB; available {free/1024**3:.1f} GiB", flush=True)
                        if not self.smoke and free < 2*estimate:
                            raise OSError("Insufficient disk reserve: require twice estimated cache size; existing blocks retained")
            finally:
                for array in opened.values():
                    array.close()
        self.csv("ridge_cosmic_feature_ranges.csv", pd.concat(range_tables, ignore_index=True))
        self.json(self.results / "cache_manifest.json", {"fingerprint": self.fp, "smoke": self.smoke,
            "input_rows": len(frame), "input_profiles": frame[["profile_id", "time"]].drop_duplicates().shape[0],
            "source_archives": frame.source_archive.nunique(), "pressure_levels": levels,
            "files": [str(p.relative_to(self.project)) for p in all_files], "preload_audits": preloads,
            "coordinates": {"features": "geopotential height", "comparison_B_C": "approximate MSL geometric height", "comparison_A": "legacy geopotential height"}})

    def evaluate_cached(self, cached, variant):
        models = self.frozen_models()
        levels = np.asarray(self.ess["cosmic2_height"]["era5_pressure_levels_hpa"])
        idx = np.asarray([int(np.flatnonzero(levels == level)[0]) for level in LEVELS])
        stack = lambda key: np.stack(cached[key].to_numpy()).astype(float)
        coord = stack("era5_geopotential_height_m" if variant == "A" else "era5_geometric_height_m")
        surface = cached["surface_height_m" if variant == "A" else "surface_geometric_height_m"].to_numpy(float)
        target = cached.height_m.to_numpy(float)
        pressure = cached.surface_pressure.to_numpy(float)
        total, wet = stack("n_total"), stack("n_wet")
        valid = (levels[None, :]*100.0 <= pressure[:, None]) & np.isfinite(total) & (total > 0)
        weights, support = batch_weights(coord, target, valid)
        baseline = weighted_values(weights, total, support, log=variant != "C")
        fraction = weighted_values(weights, wet/total, support)
        fraction = np.clip(fraction, 0, 1)
        # LEVELS is int16 in the legacy interface: multiplying by integer 100
        # overflows at 1000 hPa. Use floating conversion before physical QC.
        cweights, csupport = batch_weights(coord[:, idx], target, LEVELS[None, :]*100.0 <= pressure[:, None])
        covariance = stack("hgb_covariance6").reshape(-1, 6, 6)
        variance = propagated_variance(cweights, covariance, csupport)
        keep = [c for c in cached if c not in ("period",) and not c.endswith("6") and c not in (
            "era5_geopotential_height_m", "era5_geometric_height_m", "era5_geopotential_m2_s2",
            "temperature", "humidity", "n_dry", "n_wet", "n_total", "input_fingerprint")]
        result = cached[keep].copy()
        result["variant"] = variant
        result["era5_n_height"] = baseline
        result["era5_n_wet_height"] = baseline*fraction
        result["era5_n_dry_height"] = baseline-result.era5_n_wet_height
        result["hgb_prediction_std"] = np.sqrt(variance)
        result["residual_total"] = result.observed_n-baseline
        result["below_surface"] = target <= surface
        result["within_era5_height_support"] = np.isfinite(baseline)
        result["within_correction_height_support"] = csupport
        result["nonfinite_probability"] = ~np.isfinite(variance) | (variance <= 0)
        for name in MODELS:
            dry = weighted_values(cweights, stack(f"{name}_dry6"), csupport)
            wet_c = weighted_values(cweights, stack(f"{name}_wet6"), csupport)
            result[f"{name}_correction_dry"] = dry
            result[f"{name}_correction_wet"] = wet_c
            result[f"{name}_correction_total"] = dry+wet_c
            result[f"{name}_prediction_n"] = baseline+dry+wet_c
        result["evaluation_mask"] = (cached.height_mask.astype(bool).to_numpy() & np.isfinite(cached.observed_n)
            & ~result.below_surface & support & csupport & np.isfinite(baseline)
            & np.isfinite(result.hgb_correction_total) & ~result.nonfinite_probability)
        for name in MODELS:
            selected = result.loc[result.evaluation_mask, f"{name}_correction_total"]
            if not np.isfinite(selected).all():
                raise ValueError(f"{name} cannot be evaluated on the common sample; do not drop it")
        return self.groups(result, "cosmic2")

    def groups(self, frame, source):
        self.frozen_models()
        frame = add_common_groups(frame, self.wet_edges)
        frame["utc_support"] = np.where(utc_support(frame.time), "within_1h_of_00_12", "other_utc")
        frame["height_band"] = "not_applicable"
        if source == "cosmic2":
            frame["height_band"] = pd.cut(frame.height_m, [500, 2000, 5000, 9000], labels=["0.5-2", "2-5", "5-9"], include_lowest=True).astype(str)
        return frame

    def evaluate_internal(self, protocol):
        p = self.results / "evaluation" / protocol / "predictions.parquet"
        if self.reuse(p):
            return
        frame = self.read_licensed(protocol)
        dry, wet = self.ridge().predict(frame)
        frame["ridge_repaired_correction_dry"], frame["ridge_repaired_correction_wet"] = dry, wet
        frame["ridge_repaired_correction_total"] = dry+wet
        frame["observed_n"] = frame.era5_n+frame.residual_total
        runs = self.path("results_mapping/revision2_confirmatory") / protocol
        for name in ("seasonal_mean", "ridge", "hgb", "hgb_prob_hetero_structured"):
            original = pd.read_parquet(runs / f"{name}_run42" / "predictions.parquet")
            original.time = pd.to_datetime(original.time, utc=True)
            cols = [*ROW_KEYS, "evaluation_mask", "prediction_dry", "prediction_wet", "prediction_total"]
            if name == "hgb_prob_hetero_structured":
                cols = [*ROW_KEYS, "std_total"]
            original = original[cols]
            if original.duplicated(ROW_KEYS).any():
                raise ValueError("Duplicate frozen internal predictions")
            lookup = "ridge_original" if name == "ridge" else name
            rename = {"prediction_dry": f"{lookup}_correction_dry", "prediction_wet": f"{lookup}_correction_wet", "prediction_total": f"{lookup}_correction_total"}
            if name == "hgb_prob_hetero_structured":
                rename = {"std_total": "hgb_prediction_std"}
            else:
                rename["evaluation_mask"] = f"{lookup}_mask"
            frame = frame.merge(original.rename(columns=rename), on=ROW_KEYS, how="left", validate="one_to_one", indicator=True)
            if not frame._merge.eq("both").all():
                raise ValueError("Frozen/internal evaluation keys disagree")
            frame = frame.drop(columns="_merge")
        frame["evaluation_mask"] = frame.hgb_mask.astype(bool)
        if not frame.seasonal_mean_mask.eq(frame.hgb_mask).all() or not frame.ridge_original_mask.eq(frame.hgb_mask).all():
            raise ValueError("Internal baseline evaluation masks differ")
        frame["variant"] = "pressure"
        frame["common_mask"] = frame.evaluation_mask
        self.parquet(p, self.groups(frame, "igra"))
        audit = self.ridge().range_audit(frame.loc[frame.evaluation_mask])
        audit["source"] = protocol
        self.csv(f"ridge_feature_ranges_{protocol}.csv", audit)

    def evaluate_rapsodi(self):
        p = self.results / "evaluation" / "rapsodi" / "predictions.parquet"
        if self.reuse(p):
            return
        from evaluate_external_profiles import prepare_external_rows
        raw, audit = prepare_external_rows(pd.read_parquet(self.path(self.ess["frozen_inputs"]["rapsodi"]["path"])), "rapsodi", self.path(self.ess["frozen_inputs"]["rapsodi"]["platform_audit"]))
        raw = take_profiles(raw, self.maximum)
        rows = rows_to_profiles(raw).row_frame
        frozen = pd.read_parquet(self.path("results_mapping/revision2_ess/external/rapsodi_model_suite/predictions.parquet"))
        frozen.time = pd.to_datetime(frozen.time, utc=True)
        columns = [*ROW_KEYS, "evaluation_mask", "hgb_prediction_std", "land_sea"]
        for name in ("seasonal_mean", "ridge", "hgb"):
            columns.extend([f"{name}_correction_{c}" for c in ("dry", "wet", "total")])
        merged = rows.merge(frozen[columns], on=ROW_KEYS, how="left", validate="one_to_one", indicator=True)
        if not merged._merge.eq("both").all():
            raise ValueError("RAPSODI original and frozen prediction keys disagree")
        merged = merged.drop(columns="_merge").rename(columns={f"ridge_correction_{c}": f"ridge_original_correction_{c}" for c in ("dry", "wet", "total")})
        dry, wet = self.ridge().predict(merged)
        merged["ridge_repaired_correction_dry"], merged["ridge_repaired_correction_wet"] = dry, wet
        merged["ridge_repaired_correction_total"] = dry+wet
        merged["variant"], merged["common_mask"] = "pressure", merged.evaluation_mask
        self.parquet(p, self.groups(merged, "rapsodi"))
        self.json(self.results / "evaluation/rapsodi/platform_audit.json", audit)

    def legacy_comparison(self, new, old):
        columns = [*RO_KEYS, "evaluation_mask", "era5_n_height", "hgb_correction_total", "hgb_prediction_std"]
        old = old[columns].copy()
        old.time = pd.to_datetime(old.time, utc=True)
        join = new.merge(old, on=RO_KEYS, how="left", suffixes=("", "_old"), validate="one_to_one", indicator=True)
        if not join._merge.eq("both").all():
            raise ValueError("Legacy COSMIC prediction keys missing")
        same = join.evaluation_mask.astype(bool) & join.evaluation_mask_old.astype(bool)
        row = {"n_common": int(same.sum()), "mask_disagreements": int((join.evaluation_mask != join.evaluation_mask_old).sum())}
        for column in ("era5_n_height", "hgb_correction_total", "hgb_prediction_std"):
            row[f"max_abs_{column}"] = float(np.max(np.abs(join.loc[same, column]-join.loc[same, column+"_old"]))) if same.any() else np.nan
            if same.any() and row[f"max_abs_{column}"] > 1e-6:
                raise ValueError(f"Legacy A failed reproduction: {column} {row}")
        if row["mask_disagreements"]:
            raise ValueError(f"Legacy A masks changed: {row}")
        return row

    def stage_evaluate(self):
        self.require("fit-ridge")
        self.require("cache-cosmic")
        for protocol in tqdm(PROTOCOLS, desc="Evaluate fixed internal protocols"):
            self.evaluate_internal(protocol)
        self.evaluate_rapsodi()
        cache = self.read(self.results / "cache_manifest.json")
        reproduction = []
        old_month, old = None, None
        for rel in tqdm(cache["files"], desc="Evaluate cached A/B/C blocks"):
            p = self.path(rel)
            period = p.parent.name
            output = self.results / "evaluation/cosmic2" / period / p.name
            if self.reuse(output):
                reproduction.append(self.read(output.with_suffix(".reproduction.json")))
                self.record(output.with_suffix(".reproduction.json"))
                continue
            frame = pd.read_parquet(p)
            variants = [self.evaluate_cached(frame, v) for v in ("A", "B", "C")]
            common = np.logical_and.reduce([f.evaluation_mask.to_numpy(bool) for f in variants])
            for f in variants:
                f["common_mask"] = common
            if old_month != period:
                old = pd.read_parquet(self.path("results_mapping/revision2_ess/external/cosmic2_height_model_suite/predictions") / f"predictions_{period}.parquet")
                old_month = period
            audit = self.legacy_comparison(variants[0], old)
            audit.update({"period": period, "block": p.name})
            self.json(output.with_suffix(".reproduction.json"), audit)
            reproduction.append(audit)
            self.parquet(output, pd.concat(variants, ignore_index=True))
        self.csv("legacy_A_reproduction.csv", pd.DataFrame(reproduction))
        self.build_metrics()

    def evaluation_files(self, source):
        directory = self.results / "evaluation" / source
        return sorted(directory.rglob("*.parquet"))

    def load_evaluation(self, source, variant=None, columns=None):
        parts = []
        for p in tqdm(self.evaluation_files(source), desc=f"Read {source} evidence", leave=False):
            if not completed_partition(p, self.fp):
                raise ValueError(f"Incomplete evaluation file: {p}")
            available = pq.read_schema(p).names
            use = [c for c in columns if c in available] if columns else available
            f = pd.read_parquet(p, columns=use, filters=[("variant", "==", variant)] if variant else None)
            parts.append(f)
        if not parts:
            raise ValueError(f"No complete evaluation partitions for {source}")
        return pd.concat(parts, ignore_index=True)

    def group_specs(self, source, full=False):
        groups = [None, "height_m" if source == "cosmic2" else "pressure_hpa", "macro_region"]
        if source == "cosmic2":
            groups += ["height_band", "land_sea"]
        if full:
            groups += ["season", "wet_regime", "utc_support", "month"]
        return groups

    @staticmethod
    def grouped(frame, grouping):
        if grouping is None:
            return [("overall", frame)]
        return frame.groupby(grouping, observed=True, dropna=False, sort=True)

    def build_metrics(self):
        # Accumulate numeric sufficient statistics per block, never join all A/B/C rows.
        accum = {}
        mask_rows = []
        for source in (*PROTOCOLS, "rapsodi", "cosmic2"):
            for p in tqdm(self.evaluation_files(source), desc=f"Metrics {source}"):
                f = pd.read_parquet(p)
                legacy_mask = f.loc[f.variant.eq("A")].set_index(RO_KEYS).evaluation_mask if source == "cosmic2" else None
                for variant, vf in f.groupby("variant", sort=True):
                    reasons = {"n_input": len(vf), "n_valid": int(vf.evaluation_mask.sum()), "n_common": int(vf.common_mask.sum())}
                    if source == "cosmic2":
                        previous = legacy_mask.reindex(pd.MultiIndex.from_frame(vf[RO_KEYS])).to_numpy(bool)
                        current = vf.evaluation_mask.to_numpy(bool)
                        removed = previous & ~current
                        reasons.update({"added_vs_A": int((~previous & current).sum()), "removed_vs_A": int(removed.sum()),
                            "removed_below_surface": int((removed & vf.below_surface).sum()),
                            "removed_outside_era5_support": int((removed & ~vf.within_era5_height_support).sum()),
                            "removed_outside_correction_support": int((removed & ~vf.within_correction_height_support).sum()),
                            "removed_nonfinite_probability": int((removed & vf.nonfinite_probability).sum())})
                        reasons.update({"below_surface": int(vf.below_surface.sum()),
                            "outside_era5_support": int((~vf.within_era5_height_support).sum()),
                            "outside_correction_support": int((~vf.within_correction_height_support).sum()),
                            "nonfinite_probability": int(vf.nonfinite_probability.sum()),
                            "input_masked": int((~vf.height_mask).sum())})
                    mask_rows.append({"source": source, "variant": variant, "block": str(p.relative_to(self.results)), **reasons})
                    for cohort in (["own", "common"] if source == "cosmic2" else ["own"]):
                        selected = vf.loc[vf.evaluation_mask if cohort == "own" else vf.common_mask]
                        for grouping in self.group_specs(source, full=True):
                            for group, part in self.grouped(selected, grouping):
                                if part.empty:
                                    continue
                                r = part.residual_total.to_numpy(float)
                                for model in ("era5", *MODELS):
                                    c = np.zeros(len(r)) if model == "era5" else part[f"{model}_correction_total"].to_numpy(float)
                                    m = point_metrics(r, c)
                                    if model == "hgb":
                                        m.update(gaussian_metrics(r, c, part.hgb_prediction_std))
                                    key = (source, variant, cohort, grouping or "overall", str(group), model)
                                    row = accum.setdefault(key, {"n": 0})
                                    n = len(r)
                                    row["n"] += n
                                    for name, value in m.items():
                                        if name in ("n", "status", "rmse_delta"):
                                            continue
                                        metric = name+"_squared" if name in ("rmse", "era5_rmse") else name
                                        row[metric] = row.get(metric, 0.) + n*(value**2 if name in ("rmse", "era5_rmse") else value)
        records = []
        for key, sums in accum.items():
            n = sums["n"]
            row = dict(zip(("source", "variant", "cohort", "grouping", "group", "model"), key))
            row["n"] = n
            for metric, value in sums.items():
                if metric == "n":
                    continue
                name = metric.removesuffix("_squared") if metric in ("rmse_squared", "era5_rmse_squared") else metric
                row[name] = np.sqrt(value/n) if name in ("rmse", "era5_rmse") else value/n
            row["rmse_delta"] = row["rmse"]-row["era5_rmse"]
            row["status"] = "legacy_numerical_failure_not_comparable" if row["model"] == "ridge_original" and row["source"] in ("cosmic2", "rapsodi") else "ok"
            row["mse_closure"] = row["delta_mse"]-row["correction_penalty"]-row["alignment_term"]
            if row["status"] == "ok" and abs(row["mse_closure"]) >= 1e-10:
                raise ValueError(f"Correction MSE identity failed: {row}")
            records.append(row)
        self.csv("point_probability_metrics.csv", pd.DataFrame(records))
        masks = pd.DataFrame(mask_rows)
        self.csv("mask_audit_by_block.csv", masks)
        self.csv("mask_audit_summary.csv", masks.groupby(["source", "variant"]).sum(numeric_only=True).reset_index())

    def stage_bootstrap(self):
        self.require("evaluate")
        tables, hierarchical = [], []
        columns = ["station_id", "profile_id", "time", "source_archive", "date", "pressure_hpa", "height_m", "macro_region", "height_band", "land_sea", "evaluation_mask", "common_mask", "residual_total", "variant"]
        corrections = [f"{m}_correction_total" for m in ("seasonal_mean", "ridge_repaired", "hgb")]
        columns += corrections
        for source in (*PROTOCOLS, "rapsodi", "cosmic2"):
            variants = ("A", "B", "C") if source == "cosmic2" else ("pressure",)
            for variant in variants:
                frame = self.load_evaluation(source, variant, columns)
                cluster = "source_archive" if source == "cosmic2" else ("date" if source == "rapsodi" else "station_id")
                for cohort in (("own", "common") if source == "cosmic2" else ("own",)):
                    valid = frame.loc[frame.evaluation_mask if cohort == "own" else frame.common_mask]
                    for grouping in self.group_specs(source):
                        for group, part in tqdm(list(self.grouped(valid, grouping)), desc=f"Bootstrap {source}/{variant}/{cohort}/{grouping}", leave=False):
                            if part.empty:
                                continue
                            table = paired_bootstrap(part, cluster, corrections, self.replicates, 42,
                                                     station_mean=source in PROTOCOLS, minimum_clusters=10)
                            for key, value in {"source": source, "variant": variant, "cohort": cohort,
                                               "grouping": grouping or "overall", "group": str(group)}.items():
                                table[key] = value
                            tables.append(table)
                if source in PROTOCOLS:
                    values = hierarchical_bootstrap(frame.loc[frame.evaluation_mask], corrections, self.replicates, 42)
                    values["source"] = source
                    hierarchical.append(values)
        self.csv("paired_cluster_bootstrap.csv", pd.concat(tables, ignore_index=True))
        self.csv("hierarchical_bootstrap_repaired.csv", pd.concat(hierarchical, ignore_index=True))

    def attribution_diagnostics(self):
        summaries, budgets, bootstrap_rows = [], [], []
        for source in ("spatial_station_disjoint", "space_time_holdout", "rapsodi"):
            frame = self.load_evaluation(source)
            frame = frame.loc[frame.evaluation_mask].copy()
            to = frame.temperature_k if source == "rapsodi" else frame.igra_temperature_k
            qo = frame.specific_humidity if source == "rapsodi" else frame.igra_specific_humidity
            attr = thermo_attribution(frame.pressure_hpa.to_numpy(float), to.to_numpy(float), qo.to_numpy(float),
                                     frame.era5_temperature_k.to_numpy(float), frame.era5_specific_humidity.to_numpy(float))
            for key, value in attr.items():
                frame[key] = value
            if not np.isfinite(frame[list(attr)].to_numpy()).all():
                raise ValueError("Valid thermodynamic observations contain non-finite values")
            if np.max(np.abs(frame.thermo_residual-frame.residual_total)) >= 1e-10:
                raise ValueError("Thermodynamic source does not reproduce frozen residual")
            detail = frame[[*ROW_KEYS, *attr]]
            self.parquet(self.results / "diagnostics" / f"thermo_{source}.parquet", detail)
            cluster = "date" if source == "rapsodi" else "station_id"
            for grouping in (None, "pressure_hpa", "season", "macro_region", "wet_regime"):
                for group, part in self.grouped(frame, grouping):
                    prefix = {"source": source, "grouping": grouping or "overall", "group": str(group),
                              "n": len(part), "clusters": part[cluster].nunique(), "bootstrap_unit": cluster}
                    b = error_budget(part.phi_temperature, part.phi_humidity)
                    summaries.append({**prefix, **{k.replace("dry", "temperature").replace("wet", "humidity"): v for k, v in b.items()},
                                      "max_attribution_closure": float(np.abs(part.closure).max())})
                    for model in ("era5", "seasonal_mean", "ridge_repaired", "hgb"):
                        d = -part.residual_dry.to_numpy(float)
                        w = -part.residual_wet.to_numpy(float)
                        if model != "era5":
                            d += part[f"{model}_correction_dry"].to_numpy(float)
                            w += part[f"{model}_correction_wet"].to_numpy(float)
                        budget = error_budget(d, w)
                        if abs(budget["closure_error"]) >= 1e-10 or abs(budget["cross_term"]-budget["twice_covariance"]-budget["twice_bias_product"]) >= 1e-10:
                            raise ValueError("Physical MSE/covariance/bias identity failed")
                        budgets.append({**prefix, "model": model, **budget})
                    # Thermodynamic attribution CIs: pooled component MSE, common draws.
                    values = pd.DataFrame({"cluster": part[cluster], "n": 1., "temperature_mse": part.phi_temperature**2,
                        "humidity_mse": part.phi_humidity**2, "cross_term": 2*part.phi_temperature*part.phi_humidity,
                        "total_mse": part.thermo_residual**2}).groupby("cluster", sort=True).sum()
                    numeric = values.drop(columns="n").to_numpy()
                    counts = values.n.to_numpy()
                    ci = None
                    if len(values) >= 10:
                        rng = np.random.default_rng(42)
                        samples = np.empty((self.replicates, 4))
                        for start in tqdm(range(0, self.replicates, 200), desc=f"Attribution bootstrap {source}/{group}", leave=False):
                            ids = rng.integers(len(values), size=(min(200, self.replicates-start), len(values)))
                            samples[start:start+len(ids)] = numeric[ids].sum(axis=1)/counts[ids].sum(axis=1)[:, None]
                        ci = np.quantile(samples, [.025, .975], axis=0)
                    for j, metric in enumerate(values.drop(columns="n").columns):
                        bootstrap_rows.append({**prefix, "metric": metric, "estimate": numeric[:, j].sum()/counts.sum(),
                            "ci_lower": ci[0,j] if ci is not None else np.nan, "ci_upper": ci[1,j] if ci is not None else np.nan,
                            "replicates": self.replicates, "estimand": "pooled_row_component_mse", "status": "ok" if ci is not None else "insufficient_clusters"})
        self.csv("temperature_humidity_attribution.csv", pd.DataFrame(summaries))
        self.csv("temperature_humidity_bootstrap.csv", pd.DataFrame(bootstrap_rows))
        self.csv("physical_budget_models.csv", pd.DataFrame(budgets))

    def shape_diagnostics(self):
        summaries, influences, pits, quantiles = [], [], [], []
        for source in ("spatial_station_disjoint", "space_time_holdout", "rapsodi", "cosmic2"):
            columns = ["time", "station_id", "source_archive", "date", "pressure_hpa", "height_m", "macro_region",
                       "residual_total", "hgb_correction_total", "hgb_prediction_std", "evaluation_mask", "variant"]
            frame = self.load_evaluation(source, "B" if source == "cosmic2" else "pressure", columns)
            frame = frame.loc[frame.evaluation_mask].copy()
            cluster = "source_archive" if source == "cosmic2" else ("date" if source == "rapsodi" else "station_id")
            for grouping in (None, "height_m" if source == "cosmic2" else "pressure_hpa", "macro_region"):
                for group, part in tqdm(list(self.grouped(frame, grouping)), desc=f"Probability shape {source}/{grouping}", leave=False):
                    error = (part.residual_total-part.hgb_correction_total).to_numpy(float)
                    sd = part.hgb_prediction_std.to_numpy(float)
                    values = probability_shape(error, sd)
                    prefix = {"source": source, "grouping": grouping or "overall", "group": str(group), "clusters": part[cluster].nunique(), "cluster_unit": cluster}
                    summaries.append({**prefix, **values})
                    z = error/sd
                    hist, edges = np.histogram(norm.cdf(z), bins=np.linspace(0, 1, 21))
                    for i, count in enumerate(hist):
                        pits.append({**prefix, "left": edges[i], "right": edges[i+1], "count": count, "fraction": count/len(z)})
                    if z.std() > 0:
                        u = (z-z.mean())/z.std()
                        probabilities = np.linspace(.005, .995, 199)
                        # A single multi-quantile partition avoids scanning the full
                        # two-million-row sample separately for all 199 quantiles.
                        for q, gq, zq, uq in zip(probabilities, norm.ppf(probabilities), np.quantile(z, probabilities), np.quantile(u, probabilities)):
                            quantiles.append({**prefix, "probability": q, "gaussian_quantile": gq, "z_quantile": zq, "u_quantile": uq})
                    for key in tqdm(sorted(part[cluster].astype(str).unique()), desc="Leave-one-cluster shape influence", leave=False):
                        keep = part[cluster].astype(str).to_numpy() != key
                        if not keep.any():
                            continue
                        drop = probability_shape(error[keep], sd[keep])
                        influences.append({**prefix, "left_out_cluster": key,
                            **{k: drop.get(k, np.nan) for k in ("z_mean", "z_std", "coverage_90", "diagnostic_scale_90", "u_excess_kurtosis", "u_tail_gt3")}})
        self.csv("probability_shape_diagnostics.csv", pd.DataFrame(summaries))
        self.csv("probability_leave_one_cluster.csv", pd.DataFrame(influences))
        self.csv("pit_histograms.csv", pd.DataFrame(pits))
        self.csv("probability_qq_data.csv", pd.DataFrame(quantiles))

    def matched_diagnostics(self):
        # Selection is based only on frozen INMG observation coordinates/times.
        from evaluate_external_profiles import prepare_external_rows
        r, _ = prepare_external_rows(pd.read_parquet(self.path(self.ess["frozen_inputs"]["rapsodi"]["path"])), "rapsodi", self.path(self.ess["frozen_inputs"]["rapsodi"]["platform_audit"]))
        locations = r[["station_id", "time", "latitude", "longitude"]].drop_duplicates()
        start, end = locations.time.min(), locations.time.max()
        clat, clon = float(locations.latitude.median()), float(locations.longitude.median())
        criteria = {"start_utc": start, "end_utc_inclusive": end, "latitude": clat, "longitude": clon,
                    "radius_km": 1000, "synoptic_tolerance_hours": 1, "wet_edges": self.wet_edges,
                    "coordinate_matching": False, "interpretation": "exploratory conditioning; no causal observing-platform effect"}
        self.json(self.results / "diagnostics/matching_definition.json", criteria)
        records = []
        for source in ("spatial_station_disjoint", "space_time_holdout", "rapsodi", "cosmic2"):
            cols = ["station_id", "profile_id", "time", "date", "source_archive", "latitude", "longitude", "pressure_hpa", "height_m",
                    "wet_regime", "utc_support", "evaluation_mask", "residual_total", *[f"{m}_correction_total" for m in ("seasonal_mean", "ridge_repaired", "hgb")]]
            f = self.load_evaluation(source, "B" if source == "cosmic2" else "pressure", cols)
            f = f.loc[f.evaluation_mask].copy()
            cluster = "source_archive" if source == "cosmic2" else ("date" if source == "rapsodi" else "station_id")
            temporal = pd.to_datetime(f.time, utc=True).between(start, end)
            spatial = distance_km(f.latitude, f.longitude, clat, clon) <= 1000
            for subset, mask in (("all", np.ones(len(f), bool)), ("same_period", temporal), ("within_1000km", spatial), ("same_period_and_1000km", temporal & spatial)):
                part = f.loc[mask]
                for grouping in (None, "wet_regime", "utc_support"):
                    groups = self.grouped(part, grouping) if not part.empty else [("overall", part)]
                    for group, g in groups:
                        ncluster = g[cluster].nunique()
                        for model in ("seasonal_mean", "ridge_repaired", "hgb"):
                            metrics = point_metrics(g.residual_total, g[f"{model}_correction_total"])
                            records.append({"source": source, "subset": subset, "grouping": grouping or "overall", "group": str(group),
                                "model": model, "clusters": ncluster, "cluster_unit": cluster, "days": g.date.nunique(),
                                "profiles": g[["profile_id" if source == "cosmic2" else "station_id", "time"]].drop_duplicates().shape[0],
                                "vertical_coordinate": "MSL_geometric_height" if source == "cosmic2" else "pressure_hpa", **metrics,
                                "status": "no_samples" if g.empty else ("insufficient_clusters_descriptive_only" if ncluster < 10 else "exploratory_descriptive_only")})
        self.csv("conditioned_cross_platform_comparison.csv", pd.DataFrame(records))

    def stage_diagnose(self):
        self.require("evaluate")
        self.frozen_models()
        self.attribution_diagnostics()
        self.shape_diagnostics()
        self.matched_diagnostics()

    def diagnostic_figures(self, metrics, shape):
        # Adapted from plot-from-data line_confidence_band template: real data only.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIX Two Text", "DejaVu Serif"],
                             "text.usetex": False, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
        root = self.paper / "figures"
        root.mkdir(parents=True, exist_ok=True)
        selected = metrics.loc[metrics.source.eq("cosmic2") & metrics.model.eq("hgb") & metrics.cohort.eq("common") & metrics.grouping.eq("height_m")].copy()
        selected["height_km"] = selected.group.astype(float)/1000
        self.csv("figure_coordinate_sensitivity_data.csv", selected)
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), layout="constrained")
        colors = {"A": "#999999", "B": "#3B6BB5", "C": "#3A8B3A"}
        intervals = pd.read_csv(self.stats / "paired_cluster_bootstrap.csv")
        intervals = intervals.loc[intervals.source.eq("cosmic2") & intervals.model.eq("hgb_correction_total")
            & intervals.cohort.eq("common") & intervals.grouping.eq("height_m") & intervals.metric.eq("rmse")]
        for v, part in selected.groupby("variant"):
            part = part.sort_values("height_km")
            axes[0].plot(part.height_km, part.rmse_delta, label=v, color=colors[v], lw=1.8)
            axes[1].plot(part.height_km, part.coverage_90, label=v, color=colors[v], lw=1.8)
            band = intervals.loc[intervals.variant.eq(v)].copy()
            band["height_km"] = band.group.astype(float)/1000
            band = band.sort_values("height_km")
            if band.ci_lower.notna().any():
                axes[0].fill_between(band.height_km, band.ci_lower, band.ci_upper, color=colors[v], alpha=.15)
        axes[0].axhline(0, ls="--", color="#999999", lw=1)
        axes[1].axhline(.9, ls="--", color="#999999", lw=1)
        axes[0].set(ylabel="HGB minus ERA5 RMSE (N-units)", xlabel="MSL target height (km)", title="(a) Common-sample correction response")
        axes[1].set(ylabel="90% marginal coverage", xlabel="MSL target height (km)", title="(b) Frozen probability transfer", ylim=(0, 1))
        axes[0].legend(title="Coordinate variant", frameon=False)
        for suffix in ("png", "pdf"):
            p = root / f"coordinate_sensitivity.{suffix}"
            fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=.04)
            self.record(p)
        plt.close(fig)
        qq = pd.read_csv(self.stats / "probability_qq_data.csv")
        qq = qq.loc[qq.grouping.eq("overall")]
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), layout="constrained")
        for ax, source, title in zip(axes, ("space_time_holdout", "rapsodi", "cosmic2"), ("IGRA joint holdout", "RAPSODI", "COSMIC-2")):
            part = qq.loc[qq.source.eq(source)]
            ax.plot(part.gaussian_quantile, part.u_quantile, color="#3B6BB5", lw=1.8)
            ax.plot([-3, 3], [-3, 3], color="#999999", ls="--", lw=1)
            ax.set(title=title, xlabel="Gaussian quantile")
        axes[0].set_ylabel("Centered, scaled error quantile")
        for suffix in ("png", "pdf"):
            p = root / f"probability_shape_qq.{suffix}"
            fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=.04)
            self.record(p)
        plt.close(fig)

    def stage_summarize(self):
        self.require("bootstrap")
        self.require("diagnose")
        metrics = pd.read_csv(self.stats / "point_probability_metrics.csv")
        shape = pd.read_csv(self.stats / "probability_shape_diagnostics.csv")
        overall = metrics.loc[metrics.grouping.eq("overall") & metrics.cohort.eq("own")]
        cosmic = overall.loc[overall.source.eq("cosmic2") & overall.model.eq("hgb")].set_index("variant")
        sign_flip = bool(np.sign(cosmic.loc["A", "rmse_delta"]) != np.sign(cosmic.loc["B", "rmse_delta"]))
        bands = metrics.loc[metrics.source.eq("cosmic2") & metrics.model.eq("hgb") & metrics.cohort.eq("common") & metrics.grouping.eq("height_band")]
        band_pivot = bands.pivot(index="group", columns="variant", values="rmse_delta")
        changed_bands = band_pivot.index[np.sign(band_pivot.A) != np.sign(band_pivot.B)].tolist()
        valid_external = overall.loc[overall.source.isin(["rapsodi", "cosmic2"]) & overall.model.isin(["seasonal_mean", "ridge_repaired", "hgb"])
            & (overall.variant.eq("pressure") | overall.variant.eq("B"))]
        matrix = valid_external.pivot(index="source", columns="model", values="rmse_delta")
        disagreement = any(len(set(np.sign(row))) > 1 for _, row in matrix.iterrows())
        actions = [
            ("formula", "confirmed", "Replace p-e by actual total-pressure form; clarify component convention", "definition_audit.csv"),
            ("ridge", "repaired_implementation", "Report training-only near-constant removal; original Ridge numerical failure is not physical evidence", "ridge_training_feature_audit.csv;point_probability_metrics.csv"),
            ("height", "manual_review" if sign_flip or changed_bands else "direction_stable", "Use B; disclose spherical approximation and A/C sensitivity", "legacy_A_reproduction.csv;mask_audit_summary.csv;point_probability_metrics.csv"),
            ("bootstrap", "repaired_estimand", "Separate pooled, station-mean, and station-month estimands", "hierarchical_bootstrap_repaired.csv;paired_cluster_bootstrap.csv"),
            ("physical", "exploratory_algebraic_attribution", "Temperature/humidity replacements are not causal identification", "temperature_humidity_attribution.csv;physical_budget_models.csv"),
            ("probability", "diagnosis_only", "Compare centered/scaled tails before retaining heavy-tail wording; never externally recalibrate", "probability_shape_diagnostics.csv"),
            ("platforms", "restrict_to_model_cases" if disagreement else "direction_consistent_for_tested_forms", "Do not generalize beyond tested models; matched native coordinates are not a controlled platform experiment", "conditioned_cross_platform_comparison.csv;point_probability_metrics.csv"),
        ]
        self.csv("repair_action_evidence_matrix.csv", pd.DataFrame(actions, columns=["issue", "status", "next_manuscript_action", "evidence"]))
        self.json(self.paper / "repair_review_gate.json", {"smoke": self.smoke, "scientific_review_required": True,
            "cosmic_overall_sign_changed": sign_flip, "cosmic_common_sample_height_bands_sign_changed": changed_bands,
            "model_direction_disagreement": disagreement, "primary_coordinate_variant": "B",
            "automatic_model_selection": False, "automatic_new_data_download": False,
            "submission_ready": False, "next_step": "Return evidence for scientific review before manuscript changes"})
        self.diagnostic_figures(metrics, shape)
        lines = ["# 下一轮论文修改清单（本轮未修改论文）", "",
                 "本文件仅记录待修订内容；新结果不能因不利而删除。", "",
                 "| 问题 | 待修改内容 | 证据 |", "|---|---|---|"]
        lines += [f"| {issue} | {action} | {evidence} |" for issue, _, action, evidence in actions]
        lines += ["", "额外核查：补全三类协议的校准许可；HGB 内部早停与神经 early-val 分开；",
                  "2025Q1 校准用于 2024 RAPSODI 属回顾性诊断；补充真实模型配置、文献元数据、AI 使用说明与开放研究链接。",
                  "本轮修复后评价及分组诊断不追溯称为预注册。匹配不足不自动追加资料。"]
        p = self.docs / "下一轮论文修订清单.md"
        p.write_text("\n".join(lines), encoding="utf-8")
        self.record(p)
        self.json(self.paper / "repair_evidence_manifest.json", {"fingerprint": self.fp, "smoke": self.smoke,
            "stages": {stage: str(self.stage_file(stage).relative_to(self.project)) for stage in STAGES[:6]},
            "frozen_input_manifest": str((self.results / "freeze_manifest.json").relative_to(self.project)),
            "bootstrap_replicates": self.replicates, "primary_variant": "B", "interpretation": "repair re-evaluation and exploratory diagnostics",
            "files": list(self.outputs)})

    def stage_validate(self):
        for stage in STAGES[:-1]:
            self.require(stage)
        metrics = pd.read_csv(self.stats / "point_probability_metrics.csv")
        physical = pd.read_csv(self.stats / "physical_budget_models.csv")
        thermo = pd.read_csv(self.stats / "temperature_humidity_attribution.csv")
        reproduction = pd.read_csv(self.stats / "legacy_A_reproduction.csv")
        masks = pd.read_csv(self.stats / "mask_audit_summary.csv")
        features = pd.read_csv(self.stats / "ridge_training_feature_audit.csv")
        checks = {"legacy_reproduction": bool((reproduction.filter(like="max_abs_").max() <= 1e-6).all()),
                  "legacy_masks": bool(reproduction.mask_disagreements.eq(0).all()),
                  "correction_identity": bool(metrics.loc[metrics.status.eq("ok"), "mse_closure"].abs().max() < 1e-10),
                  "physical_identity": bool(physical.closure_error.abs().max() < 1e-10),
                  "thermodynamic_identity": bool(thermo.max_attribution_closure.max() < 1e-10),
                  "near_constant_sin_removed": bool(not features.loc[features.feature.eq("sin_utc_hour"), "retained"].iloc[0]),
                  "common_mask_counts_equal": bool(masks.loc[masks.source.eq("cosmic2"), "n_common"].nunique() == 1),
                  "frozen_inputs_unchanged": True, "main_manuscripts_unchanged": True}
        cosmic = metrics.loc[metrics.source.eq("cosmic2") & metrics.model.eq("hgb") & metrics.cohort.eq("own") & metrics.grouping.eq("height_m")]
        checks["height_grid_complete"] = self.smoke or all(set(g.group.astype(float)) == set(range(500, 9001, 500)) for _, g in cosmic.groupby("variant"))
        cache = self.read(self.results / "cache_manifest.json")
        checks["archive_count"] = self.smoke or cache["source_archives"] == 24
        checks["prepared_profile_count"] = self.smoke or cache["input_profiles"] == 120908
        legacy_valid = int(masks.loc[masks.source.eq("cosmic2") & masks.variant.eq("A"), "n_valid"].iloc[0])
        checks["legacy_valid_height_count"] = self.smoke or legacy_valid == 2044878
        for name in ("coordinate_sensitivity", "probability_shape_qq"):
            for ext in ("png", "pdf"):
                p = self.paper / "figures" / f"{name}.{ext}"
                checks[f"figure_{name}_{ext}"] = p.is_file() and p.stat().st_size > 0
            from PIL import Image
            with Image.open(self.paper / "figures" / f"{name}.png") as im:
                checks[f"figure_{name}_300dpi"] = all(abs(v-300) < .1 for v in im.info.get("dpi", (0, 0)))
        tables = pd.read_csv(self.stats / "paired_cluster_bootstrap.csv")
        low = tables.clusters < 10
        checks["small_clusters_not_overclaimed"] = bool(tables.loc[low, ["ci_lower", "ci_upper"]].isna().all().all())
        checks["finite_valid_bootstrap"] = bool(np.isfinite(tables.loc[~low, ["estimate", "ci_lower", "ci_upper"]]).all().all())
        passed = all(checks.values())
        self.json(self.paper / "validation_report.json", {"status": "smoke_passed" if passed and self.smoke else ("repair_evidence_complete" if passed else "failed"),
            "checks": checks, "smoke": self.smoke, "submission_ready": False,
            "main_manuscript_changed": False, "interpretation": "Integrity checks do not certify scientific publication readiness"})
        if not passed:
            raise ValueError(f"Repair validation failed: {[k for k, v in checks.items() if not v]}")
