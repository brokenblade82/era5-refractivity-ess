"""Review of RAPSODI segment identity and level-resolved collocation.

No fitting functions are used. Old evidence is read-only. Stable profile keys
remain independent of measurement time; only marginal probabilities are used.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from scipy.stats import norm

from .revision2_data import sha256_file, stable_fingerprint, refractivity_components, load_data_config, resolve_era5_root, require_era5_audit
from .revision2_ess import load_yaml, predict_row_model
from .revision2_confirmatory import predict_raw_scale, apply_level_scales
from .revision2_ess_repair import atomic_json, atomic_parquet, completed_partition, point_metrics, gaussian_metrics, paired_bootstrap, probability_shape, thermo_attribution

KEYS = ['profile_id', 'pressure_hpa']
MODELS = ['seasonal_mean', 'ridge_repaired', 'hgb']
STAGES = ['audit', 'collocate', 'evaluate', 'summarize', 'validate']


def paired_validity(origin, trajectory):
    """Keep absent trajectory rows visible; never match on measurement time."""
    if origin.duplicated(KEYS).any() or trajectory.duplicated(KEYS).any():
        raise ValueError('Duplicate profile-pressure keys in paired comparison')
    common = origin[KEYS + ['evaluation_mask']].merge(trajectory[KEYS + ['evaluation_mask']], on=KEYS, how='outer', suffixes=('_o', '_t'), validate='one_to_one')
    common['common_mask'] = common.evaluation_mask_o.eq(True) & common.evaluation_mask_t.eq(True)
    common['change'] = np.select([common.common_mask, common.evaluation_mask_o.eq(True), common.evaluation_mask_t.eq(True)], ['common_valid', 'origin_only', 'trajectory_only'], default='neither_valid')
    return common


def pressure_interp(p, value, targets, circular=False):
    p, value = np.asarray(p, float), np.asarray(value, float)
    good = np.isfinite(p) & (p > 0) & np.isfinite(value)
    if good.sum() < 2:
        return np.full(len(targets), np.nan)
    order = np.argsort(p[good])
    x, first = np.unique(np.log(p[good][order]), return_index=True)
    y = value[good][order][first]
    if circular:
        y = np.unwrap(np.deg2rad(y))
    out = np.interp(np.log(targets), x, y, left=np.nan, right=np.nan)
    return (np.rad2deg(out) + 180) % 360 - 180 if circular else out


def relative_seconds(array, start):
    """Decoded absolute datetime takes precedence over misleading prose attrs."""
    values = np.asarray(array.values)
    if np.issubdtype(values.dtype, np.datetime64):
        out = (values - np.datetime64(pd.Timestamp(start).tz_localize(None))) / np.timedelta64(1, 's')
        mode = 'decoded_absolute_datetime'
    elif np.issubdtype(values.dtype, np.timedelta64):
        out = values / np.timedelta64(1, 's')
        mode = 'decoded_relative_timedelta'
    else:
        units = str(array.attrs.get('units', array.encoding.get('units', ''))).lower()
        if units not in {'s', 'seconds', 'seconds since launch', 'seconds since launch_time'}:
            raise ValueError(f'Ambiguous numeric measurement time units: {units!r}')
        out = values.astype(float)
        mode = 'explicit_relative_seconds'
    return np.asarray(out, float), mode


def marginal_predictions(frame, hgb, probability, ridge, seasonal):
    """Row-level features use measurement time, not profile-start grouping time."""
    frame = frame.reset_index(drop=True)
    out = frame.copy()
    for name, model in [('seasonal_mean', seasonal), ('hgb', hgb), ('ridge_repaired', ridge)]:
        dry, wet = model.predict(frame) if name == 'ridge_repaired' else predict_row_model(name, frame, model)
        out[f'{name}_correction_dry'] = dry
        out[f'{name}_correction_wet'] = wet
        out[f'{name}_correction_total'] = dry + wet
    out['hgb_prediction_std'] = apply_level_scales(frame, predict_raw_scale(probability['scale_models']['total'], frame), probability['calibration_scales']['total'])
    # Fixed HGB and the probability model must share the same means.
    pdry, pwet = predict_row_model('hgb', frame, probability['mean_models'])
    if not np.allclose(pdry + pwet, out.hgb_correction_total, atol=1e-10, rtol=0):
        raise ValueError('Probability and point HGB means differ')
    if not np.isfinite(out.hgb_prediction_std).all() or not (out.hgb_prediction_std > 0).all():
        raise ValueError('Invalid marginal predictive SD')
    return out


def probability_bootstrap(frame, replicates, minimum_clusters=10):
    """Row-weighted marginal scores, resampling days (not individual rows)."""
    r = frame.residual_total.to_numpy(float)
    c = frame.hgb_correction_total.to_numpy(float)
    s = frame.hgb_prediction_std.to_numpy(float)
    z = (r-c)/s
    a = norm.ppf(.95)
    data = pd.DataFrame({'cluster': frame.cluster_day.to_numpy(), 'n': 1.,
        'nll': .5*np.log(2*np.pi)+np.log(s)+.5*z*z,
        'crps': s*(z*(2*norm.cdf(z)-1)+2*norm.pdf(z)-1/np.sqrt(np.pi)),
        'interval_score_90': 2*a*s+20*np.maximum(np.abs(r-c)-a*s, 0),
        'width_90': 2*a*s})
    for coverage in [.5, .6, .7, .8, .9, .95]:
        data[f'coverage_{int(coverage*100)}'] = (np.abs(z) <= norm.ppf((1+coverage)/2)).astype(float)
    g = data.groupby('cluster', sort=True).sum()
    values, n = g.drop(columns='n').to_numpy(), g.n.to_numpy()
    estimate = values.sum(axis=0)/n.sum()
    samples = []
    if len(g) >= minimum_clusters:
        rng = np.random.default_rng(42)
        for start in range(0, replicates, 250):
            ids = rng.integers(len(g), size=(min(250, replicates-start), len(g)))
            samples.append(values[ids].sum(axis=1)/n[ids].sum(axis=1)[:, None])
        lo, hi = np.quantile(np.concatenate(samples), [.025, .975], axis=0)
    else:
        lo = hi = np.full(len(estimate), np.nan)
    return pd.DataFrame({'metric': g.drop(columns='n').columns, 'estimate': estimate, 'ci_lower': lo, 'ci_upper': hi,
        'clusters': len(g), 'replicates': replicates, 'seed': 42, 'estimand': 'pooled_row_mean_score',
        'status': 'ok' if len(g) >= minimum_clusters else 'insufficient_clusters'})


class FinalReview:
    def __init__(self, root, args):
        self.root, self.args = Path(root).resolve(), args
        self.cfg = load_yaml(self.root / args.config)
        self.ess = load_yaml(self.root / self.cfg['ess_config'])
        self.smoke = args.smoke
        self.maximum = (args.max_profiles or 4) if self.smoke else None
        if args.max_profiles and not self.smoke:
            raise ValueError('Partial cohort requires --smoke')
        if self.smoke and not 2 <= self.maximum <= 10:
            raise ValueError('Smoke must contain 2--10 profiles, including both directions')
        self.replicates = args.bootstrap_replicates or (100 if self.smoke else 10000)
        if (self.smoke and not 1 <= self.replicates <= 200) or (not self.smoke and self.replicates != 10000):
            raise ValueError('Formal bootstrap=10000; smoke=1--200')
        if self.cfg['primary'] != 'ascending_trajectory' or self.cfg['seed'] != 42:
            raise ValueError('Primary analysis and seed are fixed')
        allowed = self.root / 'results_mapping/revision2_ess_review'
        self.out = (self.root / self.cfg['output']).resolve()
        if self.out != allowed:
            raise ValueError('Review output must remain isolated')
        self.code = [Path(__file__), self.root / 'scripts/revision2/run_ess_final_review.py',
                     self.root / 'scripts/revision2/collocate_external_profiles.py',
                     self.root / 'scripts/revision2/collocate_era5_igra.py']
        self.code += [self.root / 'src/igra_forecast' / name for name in ['revision2_data.py', 'revision2_baselines.py', 'revision2_confirmatory.py', 'revision2_ess.py', 'revision2_ess_repair.py']]
        self.fp = stable_fingerprint({'config': self.cfg, 'ess': self.ess, 'code': {str(p.relative_to(self.root)): sha256_file(p) for p in self.code}, 'maximum': self.maximum, 'replicates': self.replicates})
        if self.smoke:
            self.out = self.out / 'smoke' / self.fp[:12]
        sys.path.insert(0, str(self.root / 'scripts/revision2'))

    def path(self, key):
        return self.root / self.cfg[key]

    def write(self, name, frame):
        atomic_parquet(frame, self.out / name, self.fp)

    def csv(self, name, frame):
        p = self.out / 'statistics' / name
        p.parent.mkdir(parents=True, exist_ok=True)
        temp = p.with_suffix('.tmp')
        frame.to_csv(temp, index=False)
        os.replace(temp, p)

    def require(self, stage):
        p = self.out / 'stages' / f'{stage}.json'
        m = json.loads(p.read_text(encoding='utf-8'))
        if m['fingerprint'] != self.fp or not m['complete']:
            raise ValueError(f'Incomplete or stale stage {stage}')
        for rel, digest in m['outputs'].items():
            if sha256_file(self.out / rel) != digest:
                raise ValueError(f'Changed stage output: {rel}')

    def input_hashes(self):
        files = [self.path(k) for k in ['legacy_predictions', 'repair_freeze', 'ridge_model', 'training_context', 'data_config', 'ess_config']]
        files += [self.root / self.ess['frozen_runs'][k]['path'] / 'model.joblib' for k in ['hgb', 'hgb_probability']]
        files += [self.root / 'data/revision2/manifests/era5_archive_manifest.json']
        files += [self.root / item['path'] for item in self.ess['frozen_inputs'].values() if 'path' in item and item is not self.ess['frozen_inputs'].get('cosmic2_height')]
        files += [self.root / self.ess['frozen_inputs']['rapsodi']['platform_audit']]
        files += sorted((self.root / 'results_mapping/revision2_confirmatory').glob('*/summary/run_metrics.csv'))
        files += sorted(self.path('raw_store').rglob('*'))
        return {str(p.relative_to(self.root)): sha256_file(p) for p in tqdm(sorted(set(p for p in files if p.is_file())), desc='Verify source hashes', leave=False)}

    def run(self):
        stage = self.args.stage
        if stage != 'audit':
            self.require('audit')
            frozen = json.loads((self.out / 'inputs.json').read_text())
            if frozen != self.input_hashes():
                raise ValueError('Frozen inputs changed; refusing mixed evidence')
            self.require(STAGES[STAGES.index(stage)-1])
        marker = self.out / 'stages' / f'{stage}.json'
        if marker.exists():
            self.require(stage)
            if stage == 'audit' and json.loads((self.out / 'inputs.json').read_text()) != self.input_hashes():
                raise ValueError('Audit source hashes changed')
        if self.args.resume and marker.exists():
            self.require(stage)
            print(f'{stage}: verified complete; reused')
            return
        getattr(self, stage)()
        outputs = {str(p.relative_to(self.out)): sha256_file(p) for p in self.out.rglob('*') if p.is_file() and 'stages' not in p.parts and p.suffix != '.tmp'}
        atomic_json(marker, {'complete': True, 'fingerprint': self.fp, 'smoke': self.smoke, 'outputs': outputs})
        print(f'{stage}: complete -> {self.out}')

    def audit(self):
        hashes = self.input_hashes()
        if (self.out / 'inputs.json').exists():
            if json.loads((self.out / 'inputs.json').read_text()) != hashes:
                raise ValueError('Existing review has different inputs')
        # Verify models against pre-existing ESS fingerprints, not self-certified new hashes.
        for key in ['hgb', 'hgb_probability']:
            item = self.ess['frozen_runs'][key]
            if sha256_file(self.root / item['path'] / 'model.joblib') != item['model_sha256']:
                raise ValueError(f'Frozen model mismatch: {key}')
        freeze = json.loads(self.path('repair_freeze').read_text())
        for key in ['station_split', 'protocol_license_matrix', 'rapsodi']:
            item = self.ess['frozen_inputs'][key]
            if sha256_file(self.root / item['path']) != item['sha256']:
                raise ValueError(f'Upstream frozen input changed: {key}')
        old_stage = json.loads((self.path('ridge_model').parent.parent / 'stages/evaluate.json').read_text())
        old_records = {str((self.root / r['path']).resolve()): r['sha256'] for r in old_stage['artifacts']}
        if old_records.get(str(self.path('legacy_predictions').resolve())) != sha256_file(self.path('legacy_predictions')):
            raise ValueError('Legacy prediction differs from its completed evaluation stage')
        for key in ['ridge_model', 'training_context']:
            p = self.path(key)
            # These models were produced after the old freeze; bind to the completed fit stage.
            fit = json.loads((self.path('ridge_model').parent.parent / 'stages/fit-ridge.json').read_text())
            matches = [v for v in fit['artifacts'] if (self.root / v['path']).resolve() == p.resolve()]
            if not fit['complete'] or not matches or matches[0]['sha256'] != sha256_file(p):
                raise ValueError(f'Unverified repaired artifact: {p}')
        legacy = pd.read_parquet(self.path('legacy_predictions'))
        if legacy.duplicated(KEYS).any() or not legacy.groupby('profile_id').size().eq(6).all():
            raise ValueError('Legacy cohort must have six unique rows per profile')
        records, modes = [], set()
        with xr.open_zarr(self.path('raw_store'), consolidated=None) as ds:
            units = str(ds.p.attrs.get('units', '')).lower()
            if units not in ['pa', 'hpa']:
                raise ValueError(f'Unrecognized pressure units: {units}')
            flags = ds.ascent_flag.values
            if ds.ascent_flag.attrs.get('flag_meanings') != 'ascending descending':
                raise ValueError('Direction flag meanings changed')
            directions = {}
            for pid in legacy.profile_id.unique():
                i = int(pid.rsplit('_', 1)[1])
                flag = int(flags[i])
                if flag not in [0, 1]:
                    raise ValueError(f'Unknown ascent flag: {flag}')
                directions[pid] = 'ascending' if flag == 0 else 'descending'
            counts = pd.Series(directions).value_counts().to_dict()
            if counts != self.cfg['expected_profiles']:
                raise ValueError(f'Frozen cohort counts changed: {counts}')
            chosen = list(directions)
            if self.smoke:
                chosen = [next(k for k, v in directions.items() if v == d) for d in ['ascending', 'descending']]
                chosen += [k for k in directions if k not in chosen][:self.maximum-2]
            for pid in tqdm(chosen, desc='Audit direction and trajectory'):
                i = int(pid.rsplit('_', 1)[1])
                part = legacy.loc[legacy.profile_id.eq(pid), KEYS + ['time', 'latitude', 'longitude', 'level_mask', 'temperature_k', 'specific_humidity', 'height_m', 'observed_n_dry', 'observed_n_wet', 'observed_n']].copy()
                start = pd.Timestamp(ds.launch_time.isel(launch_time=i).item(), tz='UTC')
                if not pd.to_datetime(part.time, utc=True).eq(start).all():
                    raise ValueError('Raw index no longer matches legacy start time')
                p = np.asarray(ds.p.isel(launch_time=i), float) / (100 if units == 'pa' else 1)
                targets = part.pressure_hpa.to_numpy(float)
                seconds, mode = relative_seconds(ds.interpolated_time.isel(launch_time=i), start)
                modes.add(mode)
                dt = pressure_interp(p, seconds, targets)
                part['observation_time'] = start + pd.to_timedelta(dt, unit='s')
                part['observation_latitude'] = pressure_interp(p, ds.lat.isel(launch_time=i), targets)
                part['observation_longitude'] = pressure_interp(p, ds.lon.isel(launch_time=i), targets, circular=True)
                part['segment_start_time'] = pd.to_datetime(part.time, utc=True)
                part['direction'] = directions[pid]
                part['raw_index'] = i
                part['source_sonde_id'] = str(ds.sonde_id.isel(launch_time=i).values.item())
                part['trajectory_valid'] = part.observation_time.notna() & part.observation_latitude.between(-90, 90) & np.isfinite(part.observation_longitude)
                if np.any(np.isfinite(dt) & (np.abs(dt) > 86400)):
                    raise ValueError('Measurement time exceeds 24 hours from segment start; investigate encoding')
                part['mask_reason'] = np.where(part.trajectory_valid, 'available', 'missing_trajectory_no_fallback')
                records.append(part)
        frame = pd.concat(records, ignore_index=True)
        self.write('prepared.parquet', frame)
        atomic_json(self.out / 'inputs.json', hashes)
        atomic_json(self.out / 'audit.json', {'cohort_counts': counts, 'selected_profiles': frame.profile_id.nunique(), 'time_modes': sorted(modes), 'primary': self.cfg['primary'], 'probability': 'marginal only; 2025Q1-calibrated model transferred retrospectively to 2024', 'ascent_flag': {'0': 'ascending', '1': 'descending'}, 'units': 'pressure Pa converted to hPa; decoded measurement times UTC', 'legacy_freeze_version': freeze.get('fingerprint')})

    def collocate(self):
        from collocate_external_profiles import open_era5_variable, concatenate_time_boundaries, preload_interpolation_window, spatiotemporal_points
        config = load_data_config(self.path('data_config'))
        root = resolve_era5_root(config)
        source = pd.read_parquet(self.out / 'prepared.parquet')
        for variant in tqdm(['origin', 'trajectory'], desc='Collocation versions'):
            frame = source.copy()
            if variant == 'trajectory':
                frame['time'] = frame.observation_time
                frame['latitude'] = frame.observation_latitude
                frame['longitude'] = frame.observation_longitude
            good = frame.time.notna() & frame.latitude.between(-90, 90) & np.isfinite(frame.longitude)
      if variant == 'trajectory':
                good &= frame.trajectory_valid.astype(bool)
            frame['time'] = pd.to_datetime(frame.time, utc=True).dt.tz_localize(None)
            months = frame.loc[good, 'time'].dt.strftime('%Y%m')
            require_era5_audit(config, self.root / 'data/revision2/manifests/era5_archive_manifest.json', root, sorted(months.unique()), require_full_study=True)
            for month in tqdm(sorted(months.unique()), desc=f'{variant} months', leave=False):
                subset = frame.loc[good & frame.time.dt.strftime('%Y%m').eq(month)].sort_values(['time', 'profile_id', 'pressure_hpa'])
                # Only opened when there is an incomplete block; writes are atomic.
                arrays, handles = {}, []
                try:
                    for start in tqdm(range(0, len(subset), self.cfg['chunk_rows']), desc=f'{variant} {month} blocks', leave=False):
                        dest = self.out / 'collocated' / variant / f'{month}_{start:06d}.parquet'
                        if self.args.resume and completed_partition(dest, self.fp):
                            continue
                        if not arrays:
                            for logical in ['temperature', 'specific_humidity', 'geopotential', 'surface_pressure']:
                                current = open_era5_variable(root, pd.Period(month, freq='M'), config, logical)
                                handles.append(current)
                                previous = following = None
                                if subset.time.min() < pd.Timestamp(current.time.values.min()):
                                    previous = open_era5_variable(root, pd.Period(month, freq='M')-1, config, logical); handles.append(previous)
                                if subset.time.max() > pd.Timestamp(current.time.values.max()):
                                    following = open_era5_variable(root, pd.Period(month, freq='M')+1, config, logical); handles.append(following)
                                arrays[logical], _ = concatenate_time_boundaries(current, subset.time, previous, following)
                        part = subset.iloc[start:start+self.cfg['chunk_rows']].copy()
                        for logical, column in [('temperature', 'era5_temperature_k'), ('specific_humidity', 'era5_specific_humidity'), ('geopotential', 'era5_height_m'), ('surface_pressure', 'era5_surface_pressure_pa')]:
                            window, info = preload_interpolation_window(arrays[logical], part, logical != 'surface_pressure')
                            try:
                                part[column] = spatiotemporal_points(window, part, logical != 'surface_pressure')
                            finally:
                                if info['preloaded']: window.close()
                        part.era5_height_m /= 9.80665
                        part.era5_specific_humidity = part.era5_specific_humidity.clip(lower=0)
                        dry, wet, total, _ = refractivity_components(part.pressure_hpa.to_numpy(), part.era5_temperature_k.to_numpy(), part.era5_specific_humidity.to_numpy())
                        part['era5_n_dry'], part['era5_n_wet'], part['era5_n'] = dry, wet, total
                        part['below_ground_level'] = part.era5_surface_pressure_pa < part.pressure_hpa * 100
                        part['residual_total'] = part.observed_n - part.era5_n
                        part['residual_dry'] = part.observed_n_dry - part.era5_n_dry
                        part['residual_wet'] = part.observed_n_wet - part.era5_n_wet
                        part['evaluation_mask'] = part.level_mask.astype(bool) & ~part.below_ground_level.astype(bool)
                        part['variant'] = variant
                        part['mask_reason'] = np.select([~part.level_mask.astype(bool), part.below_ground_level], ['missing_observation_level', 'below_ground'], default='available')
                        self.write(str(dest.relative_to(self.out)), part)
                finally:
                    for da in list(arrays.values()) + handles: da.close()
            excluded = frame.loc[~good].copy()
            self.write(f'collocated/{variant}/excluded.parquet', excluded)

    def evaluate(self):
        hgb = joblib.load(self.root / self.ess['frozen_runs']['hgb']['path'] / 'model.joblib')
        probability = joblib.load(self.root / self.ess['frozen_runs']['hgb_probability']['path'] / 'model.joblib')
        ridge = joblib.load(self.path('ridge_model'))
        seasonal = joblib.load(self.path('training_context'))['seasonal_mean']
        versions = []
        for variant in ['origin', 'trajectory']:
            parts = []
            for p in tqdm(sorted((self.out / 'collocated' / variant).glob('20*.parquet')), desc=f'Frozen inference {variant}'):
                part = pd.read_parquet(p)
                good = part.evaluation_mask.astype(bool)
                if good.any():
                    predicted = marginal_predictions(part.loc[good].copy(), hgb, probability, ridge, seasonal)
                    columns = [c for c in predicted if '_correction_' in c or c == 'hgb_prediction_std']
                    part = part.merge(predicted[KEYS + columns], on=KEYS, how='left', validate='one_to_one')
                else:
                    for m in MODELS:
                        for component in ['dry', 'wet', 'total']:
                            part[f'{m}_correction_{component}'] = np.nan
                    part['hgb_prediction_std'] = np.nan
                parts.append(part)
            versions.append(pd.concat(parts, ignore_index=True))
        origin, trajectory = versions
        legacy = pd.read_parquet(self.path('legacy_predictions'))
        paired = origin.merge(legacy, on=KEYS, suffixes=('_new', '_old'), validate='one_to_one')
        valid = paired.evaluation_mask_new.astype(bool)
        if not paired.evaluation_mask_new.astype(bool).equals(paired.evaluation_mask_old.astype(bool)):
            raise ValueError('Origin mask does not reproduce legacy evaluation')
        checks = []
        for c in ['era5_temperature_k', 'era5_specific_humidity', 'era5_height_m', 'era5_surface_pressure_pa', 'era5_n', 'hgb_prediction_std'] + [f'{m}_correction_total' for m in MODELS]:
            delta = np.abs(paired.loc[valid, c+'_new']-paired.loc[valid, c+'_old']).max()
            checks.append({'field': c, 'max_abs_error': float(delta)})
            if not np.isfinite(delta) or delta > self.cfg['reproduction_tolerance']:
                raise ValueError(f'Origin reproduction failed: {c} {delta}')
        common = paired_validity(origin, trajectory)
        common = common.merge(origin[KEYS + ['direction']], on=KEYS, how='left', validate='one_to_one')
        self.csv('sample_changes.csv', common)
        for part in versions:
            part = part.merge(common[KEYS + ['common_mask']], on=KEYS, how='left', validate='one_to_one')
            part['common_mask'] = part.common_mask.eq(True)
            part['cluster_day'] = pd.to_datetime(part.segment_start_time, utc=True).dt.strftime('%Y-%m-%d')
            self.write(f'predictions/{part.variant.iloc[0]}.parquet', part)
        self.csv('legacy_reproduction.csv', pd.DataFrame(checks))

    def summarize(self):
        metrics, boots, shapes, attr, masks, probability_ci = [], [], [], [], [], []
        for variant in tqdm(['origin', 'trajectory'], desc='Summarize variants'):
            frame = pd.read_parquet(self.out / 'predictions' / f'{variant}.parquet')
            excluded = pd.read_parquet(self.out / 'collocated' / variant / 'excluded.parquet')
            for direction in tqdm(['ascending', 'descending', 'mixed'], desc=f'{variant} direction', leave=False):
                selected = frame if direction == 'mixed' else frame.loc[frame.direction.eq(direction)]
                for sample in ['own', 'common']:
                    selected_valid = selected.loc[selected['evaluation_mask' if sample == 'own' else 'common_mask'].astype(bool)]
                    for layer in ['all', 1000, 925, 850, 700, 500, 300]:
                        part = selected_valid if layer == 'all' else selected_valid.loc[selected_valid.pressure_hpa.eq(layer)]
                        meta = {'variant': variant, 'direction': direction, 'sample': sample, 'layer': layer, 'profiles': part.profile_id.nunique(), 'clusters': part.cluster_day.nunique(), 'bootstrap_unit': 'launch_day' if direction == 'ascending' else 'segment_start_day', 'primary': variant == 'trajectory' and direction == 'ascending' and sample == 'own'}
                        if part.empty:
                            metrics.append(dict(meta, model='all', n=0, status='no_samples'))
                            continue
                        for m in ['era5'] + MODELS:
                            c = np.zeros(len(part)) if m == 'era5' else part[f'{m}_correction_total'].to_numpy()
                            row = dict(meta, model=m, **point_metrics(part.residual_total, c))
                            if m == 'hgb': row.update(gaussian_metrics(part.residual_total, c, part.hgb_prediction_std))
                            metrics.append(row)
                        boot = paired_bootstrap(part, 'cluster_day', [f'{m}_correction_total' for m in MODELS], self.replicates, 42, minimum_clusters=10)
                        boots.append(boot.assign(**{k: v for k, v in meta.items() if k != 'clusters'}))
                        probability_ci.append(probability_bootstrap(part, self.replicates).assign(**{k: v for k, v in meta.items() if k != 'clusters'}))
                        shapes.append(dict(meta, **probability_shape(part.residual_total-part.hgb_correction_total, part.hgb_prediction_std)))
                        values = thermo_attribution(part.pressure_hpa.to_numpy(), part.temperature_k.to_numpy(), part.specific_humidity.to_numpy(), part.era5_temperature_k.to_numpy(), part.era5_specific_humidity.to_numpy())
                        for name, value in values.items():
                            v = np.asarray(value)
                            attr.append(dict(meta, term=name, mean=float(v.mean()), mse=float(np.mean(v*v)), max_abs=float(np.max(np.abs(v)))))
                        attr.append(dict(meta, term='twice_temperature_humidity_cross', mean=float(np.mean(2*values['phi_temperature']*values['phi_humidity'])), mse=np.nan, max_abs=np.nan))
                for reason, group in selected.groupby('mask_reason'):
                    masks.append({'variant': variant, 'direction': direction, 'reason': reason, 'rows': len(group)})
                excluded_group = excluded if direction == 'mixed' else excluded.loc[excluded.direction.eq(direction)]
                masks.append({'variant': variant, 'direction': direction, 'reason': 'missing_trajectory_no_fallback', 'rows': len(excluded_group)})
        for name, records in [('metrics', metrics), ('probability_shape', shapes), ('thermodynamic_attribution', attr), ('mask_counts', masks)]:
            self.csv(name+'.csv', pd.DataFrame(records))
        self.csv('bootstrap.csv', pd.concat(boots, ignore_index=True))
        self.csv('probability_bootstrap.csv', pd.concat(probability_ci, ignore_index=True))
        origin = pd.read_parquet(self.out / 'predictions/origin.parquet')
        traj = pd.read_parquet(self.out / 'predictions/trajectory.parquet')
        paired = origin.loc[origin.common_mask.astype(bool)].merge(traj.loc[traj.common_mask.astype(bool)], on=KEYS, suffixes=('_o', '_t'), validate='one_to_one')
        sensitivity = []
        for direction in ['ascending', 'descending', 'mixed']:
            pair = paired if direction == 'mixed' else paired.loc[paired.direction_o.eq(direction)]
            if pair.empty: continue
            for m in ['era5'] + MODELS:
                base = pair.era5_n_o + (0 if m == 'era5' else pair[f'{m}_correction_total_o'])
                new = pair.era5_n_t + (0 if m == 'era5' else pair[f'{m}_correction_total_t'])
                work = pd.DataFrame({'residual_total': (pair.observed_n_o-base).to_numpy(), 'change': (new-base).to_numpy(), 'cluster_day': pair.cluster_day_o.to_numpy()})
                result = paired_bootstrap(work, 'cluster_day', ['change'], self.replicates, 42, minimum_clusters=10)
                sensitivity.append(result.assign(model=m, direction=direction, contrast='trajectory_minus_origin', sample='common', bootstrap_unit='launch_day' if direction == 'ascending' else 'segment_start_day'))
        self.csv('collocation_sensitivity_bootstrap.csv', pd.concat(sensitivity, ignore_index=True))
        runs = []
        for p in sorted((self.root / 'results_mapping/revision2_confirmatory').glob('*/summary/run_metrics.csv')):
            r = pd.read_csv(p)
            r['source_file'] = str(p.relative_to(self.root))
            runs.append(r)
        run = pd.concat(runs, ignore_index=True)
        self.csv('probability_existing_runs.csv', run)
        cols = [c for c in run if c.startswith(('prob_', 'joint_', 'marginal_')) and pd.api.types.is_numeric_dtype(run[c])]
        summary = run.groupby(['protocol', 'model'])[cols].agg(['mean', 'std', 'count'])
        summary.columns = ['_'.join(c) for c in summary.columns]
        self.csv('probability_existing_summary.csv', summary.reset_index())
        paired_probability = []
        for protocol, p in run.groupby('protocol'):
            d = p.loc[p.model.eq('hgb_prob_hetero_diag')]
            s = p.loc[p.model.eq('hgb_prob_hetero_structured')]
            pair = d.merge(s, on='seed', suffixes=('_diagonal', '_structured'), validate='one_to_one')
            for c in cols:
                pair[c+'_difference'] = pair[c+'_structured']-pair[c+'_diagonal']
            pair['protocol'] = protocol
            paired_probability.append(pair)
        self.csv('probability_structured_minus_diagonal.csv', pd.concat(paired_probability, ignore_index=True))
        evidence = [
            ('rapsodi_primary_point', 'metrics.csv', 'trajectory;ascending;own;all;hgb', 'pooled row RMSE', 'bootstrap.csv;launch_day'),
            ('rapsodi_primary_probability', 'metrics.csv', 'trajectory;ascending;own;all;hgb', 'marginal coverage and proper scores', 'probability_bootstrap.csv;launch_day'),
            ('direction_sensitivity', 'metrics.csv', 'origin/trajectory;ascending/descending/mixed', 'descriptive composition contrast', 'no independent balloon pairing assumed'),
            ('collocation_sensitivity', 'collocation_sensitivity_bootstrap.csv', 'common sample', 'trajectory minus origin pooled error', 'launch_day for ascent'),
            ('vertical_correlation', 'probability_structured_minus_diagonal.csv', 'paired existing runs', 'joint score difference; no marginal correlation benefit', 'descriptive paired five runs'),
            ('thermodynamic_budget', 'thermodynamic_attribution.csv', 'trajectory;ascending;own', 'algebraic attribution, not causality', 'descriptive')]
        self.csv('claim_sources.csv', pd.DataFrame(evidence, columns=['claim', 'source', 'filter', 'estimand', 'inference']))
        atomic_json(self.out / 'review_status.json', {'status': 'smoke_only' if self.smoke else 'ready_for_scientific_review', 'submission_ready': False, 'primary': 'ascending_trajectory', 'no_training_or_calibration': True, 'probability_scope': 'marginal trajectory intervals only', 'next': 'Review direction and collocation sensitivity before updating manuscript numbers'})

    def validate(self):
        if self.input_hashes() != json.loads((self.out / 'inputs.json').read_text()):
            raise ValueError('Inputs were changed')
        for variant in ['origin', 'trajectory']:
            f = pd.read_parquet(self.out / 'predictions' / f'{variant}.parquet')
            if f.duplicated(KEYS).any(): raise ValueError('Duplicate profile-layer key')
            valid = f.loc[f.evaluation_mask.astype(bool)]
            for m in MODELS:
                if not np.isfinite(valid[f'{m}_correction_total']).all(): raise ValueError('Nonfinite prediction')
                if np.max(np.abs(valid[f'{m}_correction_dry'] + valid[f'{m}_correction_wet'] - valid[f'{m}_correction_total'])) > 1e-10: raise ValueError('Component identity')
            if not (valid.hgb_prediction_std > 0).all(): raise ValueError('Invalid SD')
            if valid.below_ground_level.astype(bool).any(): raise ValueError('Below-ground evaluation')
            if np.max(np.abs(valid.residual_total-valid.residual_dry-valid.residual_wet)) >= 1e-10:
                raise ValueError('Observed residual identity')
        metrics = pd.read_csv(self.out / 'statistics/metrics.csv')
        valid_metrics = metrics.loc[metrics.status.eq('ok')]
        if not np.isfinite(valid_metrics[['rmse', 'mae', 'bias']].to_numpy()).all():
            raise ValueError('Nonfinite point statistics')
        closure = valid_metrics.delta_mse-valid_metrics.correction_penalty-valid_metrics.alignment_term
        if np.max(np.abs(closure)) >= 1e-10:
            raise ValueError('Correction MSE decomposition does not close')
        attr = pd.read_csv(self.out / 'statistics/thermodynamic_attribution.csv')
        if attr.loc[attr.term.eq('closure'), 'max_abs'].max() >= 1e-10:
            raise ValueError('Thermodynamic decomposition closure')
        for name in ['bootstrap.csv', 'probability_bootstrap.csv']:
            b = pd.read_csv(self.out / 'statistics' / name)
            if not b.loc[b.clusters.lt(10), 'ci_lower'].isna().all():
                raise ValueError('CI incorrectly reported for insufficient clusters')
            if not np.isfinite(b.loc[b.status.eq('ok'), ['ci_lower', 'ci_upper']].to_numpy()).all():
                raise ValueError('Nonfinite supported confidence interval')
        atomic_json(self.out / 'validation_report.json', {'technical_checks_passed': True, 'smoke': self.smoke, 'submission_ready': False, 'fingerprint': self.fp, 'formal_bootstrap_replicates': self.replicates})
