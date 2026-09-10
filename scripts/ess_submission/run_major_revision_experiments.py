"""User-run, isolated point-objective/UTC and cross-fitted-scale sensitivities.

These are post-hoc development diagnostics, not fresh confirmatory tests.
No external sample is used for fitting, selection or calibration.
"""
from pathlib import Path
import argparse, json, sys, os, hashlib, time
os.environ.setdefault('OMP_NUM_THREADS','8')
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/revision2')]
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from run_revision2_baselines import load_protocol_data
from igra_forecast.revision2_baselines import make_row_features,row_training_mask,hgb_factory,station_crossfit_folds,rows_to_profiles
from igra_forecast.revision2_confirmatory import estimate_pairwise_correlation,predict_confirmatory_probability_hgb

PROTOCOLS=['spatial_station_disjoint','temporal_holdout','space_time_holdout']
SETTINGS=dict(learning_rate=.05,max_iter=300,max_leaf_nodes=31,min_samples_leaf=30,l2_regularization=.1,random_state=42)

def load(protocol,smoke):
    return load_protocol_data(ROOT/'data/revision2/processed/era5_igra_profiles',ROOT/'data/revision2/manifests/station_split.csv',ROOT/'data/revision2/manifests/protocol_license_matrix.csv',protocol,2 if smoke else None,None)

def features(f,no_utc):
    x=make_row_features(f)
    return np.delete(x,[14,15],axis=1) if no_utc else x

def score(f,c):
    e=np.asarray(c)-f.residual_total.to_numpy(); r=f.residual_total.to_numpy()
    g=pd.DataFrame(dict(station=f.station_id.to_numpy(),a=r*r,b=e*e)).groupby('station').mean()
    d=np.sqrt(g.b)-np.sqrt(g.a);rng=np.random.default_rng(42)
    boot=d.to_numpy()[rng.integers(len(d),size=(10000,len(d)))].mean(1)
    return dict(n=len(f),era5_rmse=np.sqrt(np.mean(r*r)),rmse=np.sqrt(np.mean(e*e)),mae=np.mean(abs(e)),bias=np.mean(e),station_delta=d.mean(),ci_lower=np.quantile(boot,.025),ci_upper=np.quantile(boot,.975))

def fit_hgb(train,early,no_utc,objective,smoke):
    x=features(train,no_utc);v=features(early,no_utc)
    cols=['residual_total'] if objective=='total' else ['residual_dry','residual_wet']
    # Same three candidate iteration budgets and same station-disjoint selection set.
    candidates=[2,3] if smoke else [100,200,300]
    best=None; records=[]
    for iterations in candidates:
        models=[HistGradientBoostingRegressor(**{**SETTINGS,'max_iter':iterations},early_stopping=False).fit(x,train[col]) for col in cols]
        pred=sum(m.predict(v) for m in models)
        sq=pd.DataFrame(dict(station=early.station_id.to_numpy(),sq=(pred-early.residual_total.to_numpy())**2))
        loss=float(np.sqrt(sq.groupby('station').sq.mean()).mean())
        records.append(dict(iterations=iterations,early_station_rmse=loss))
        if best is None or loss<best[0]:best=(loss,models,iterations)
    return best[1],records,best[2]

def point(args,out,data):
    train=data['train'];early=data['early_stopping']
    train=train.loc[row_training_mask(train)].reset_index(drop=True)
    early=early.loc[row_training_mask(early)].reset_index(drop=True)
    assert not set(train.station_id)&set(early.station_id)
    models={};selection={}
    for no_utc in [False,True]:
        for objective in ['components','total']:
            name=f'hgb_{objective}_'+('no_utc' if no_utc else 'utc')
            m,records,it=fit_hgb(train,early,no_utc,objective,args.smoke)
            models[name]=(m,no_utc);selection[name]=dict(candidates=records,chosen_iterations=it)
        name='ridge_'+('no_utc' if no_utc else 'utc')
        x=features(train,no_utc).astype(float)
        # Select training-derived near-constant columns, including the row mask.
        keep=x.std(0)>1e-12*np.maximum(1,abs(x.mean(0)))
        model=make_pipeline(StandardScaler(),Ridge(alpha=1.,solver='lsqr')).fit(x[:,keep],train.residual_total)
        models[name]=([model],no_utc,keep)
    joblib.dump(models,out/'point_models.joblib')
    (out/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    # Freeze selection before any evaluation data are scored.
    summary=[]
    for protocol in PROTOCOLS:
        d=data if protocol=='space_time_holdout' else load(protocol,args.smoke)
        f=d['evaluation'];f=f.loc[row_training_mask(f)].reset_index(drop=True)
        for name,item in models.items():
            x=features(f,item[1]).astype(float)
            if len(item)==3:x=x[:,item[2]]
            c=sum(m.predict(x) for m in item[0])
            summary.append(dict(protocol=protocol,model=name,**score(f,c)))
            f[['station_id','time','pressure_hpa','residual_total']].assign(correction=c).to_parquet(out/f'{protocol}_{name}.parquet',index=False)
        # Seasonal means use the same permitted training data; no selection budget.
        for no_utc in [False,True]:
            tr=train.copy();te=f.copy()
            for z in [tr,te]:
                dt=pd.to_datetime(z.time);z['month']=dt.dt.month;z['hour']=dt.dt.hour
            keys=['pressure_hpa','month']+([] if no_utc else ['hour'])
            lookup=tr.groupby(keys).residual_total.mean();fallback=tr.groupby('pressure_hpa').residual_total.mean()
            c=lookup.reindex(pd.MultiIndex.from_frame(te[keys])).to_numpy()
            missing=~np.isfinite(c);c[missing]=te.loc[missing,'pressure_hpa'].map(fallback)
            summary.append(dict(protocol=protocol,model='seasonal_'+('no_utc' if no_utc else 'utc'),**score(f,c)))
    pd.DataFrame(summary).to_csv(out/'point_summary.csv',index=False)
    # RAPSODI is evaluated only after the internal selection is saved.
    f=pd.read_parquet(ROOT/'results_mapping/revision2_ess_review/predictions/trajectory.parquet')
    f=f.loc[f.evaluation_mask & f.direction.eq('ascending')].copy()
    f['station_id']=f.profile_id.astype(str)
    ext=[]
    for name,item in models.items():
        x=features(f,item[1]).astype(float)
        if len(item)==3:x=x[:,item[2]]
        c=sum(m.predict(x) for m in item[0]);r=f.residual_total.to_numpy()
        ext.append(dict(model=name,n=len(f),rmse=np.sqrt(np.mean((c-r)**2)),era5_rmse=np.sqrt(np.mean(r*r)),status='descriptive_external_no_selection'))
    pd.DataFrame(ext).to_csv(out/'rapsodi_point_summary.csv',index=False)

def scale(args,out,data):
    """Hold fixed fitted means, marginal scales/calibration; change correlation only.

    Reconstruct original station OOF mean errors, then cross-fit the scale on
    those errors by station. This is a scale-fit sensitivity, not fully nested
    mean/scale cross-fitting; the residual targets share other-fold mean fits.
    """
    train=data['train']; train=train.loc[row_training_mask(train)].reset_index(drop=True)
    x=make_row_features(train);folds=station_crossfit_folds(train.station_id,5,42)
    oof=np.zeros(len(train));sfold=np.zeros(len(train))
    settings={**SETTINGS,'max_iter':3} if args.smoke else SETTINGS
    for fold in np.unique(folds):
        hold=folds==fold
        for component in ['dry','wet']:
            m=hgb_factory({**settings,'random_state':42+int(fold)}).fit(x[~hold],train.loc[~hold,'residual_'+component])
            oof[hold]+=m.predict(x[hold])
    err=train.residual_total.to_numpy()-oof
    target=np.log(np.maximum(abs(err),.05))
    for fold in np.unique(folds):
        hold=folds==fold
        m=hgb_factory({**settings,'random_state':1042}).fit(x[~hold],target[~hold])
        sfold[hold]=np.exp(np.clip(m.predict(x[hold]),-5,4))
    correlation,counts=estimate_pairwise_correlation(train,err/sfold,30)
    standardized=train[['station_id','time','pressure_hpa']].assign(z=err/sfold)
    pivot=standardized.pivot(index=['station_id','time'],columns='pressure_hpa',values='z').reindex(columns=[1000,925,850,700,500,300])
    raw=pivot.corr(min_periods=30).fillna(0).to_numpy();np.fill_diagonal(raw,1)
    model=joblib.load(ROOT/'results_mapping/revision2_confirmatory/space_time_holdout/hgb_prob_hetero_structured_run42/model.joblib')
    old=model['correlation_total'].copy();model['correlation_total']=correlation
    np.savetxt(out/'crossfit_scale_correlation.csv',correlation,delimiter=',')
    np.savetxt(out/'correlation_pair_counts.csv',counts,delimiter=',',fmt='%d')
    report=dict(projection_frobenius=float(np.linalg.norm(correlation-raw)),raw_min_eigenvalue=float(np.linalg.eigvalsh(raw).min()),correlation_change_frobenius=float(np.linalg.norm(correlation-old)),scope='station-crossfitted scale conditional on original OOF mean-error targets; not fully nested cross-fitting')
    arrays=rows_to_profiles(data['evaluation']);pred=predict_confirmatory_probability_hgb(model,arrays)
    error=arrays.target_total-pred.mean_total
    rows=[]
    for i in range(len(error)):
        mask=arrays.mask[i];e=error[i,mask];s=pred.std_total[i,mask]
        values=[]
        for R in [old,correlation]:
            cov=R[np.ix_(mask,mask)]*s[:,None]*s[None,:]
            sign,logdet=np.linalg.slogdet(cov);assert sign>0
            values.append(.5*(len(e)*np.log(2*np.pi)+logdet+e@np.linalg.solve(cov,e)))
        rows.append(dict(station_id=arrays.keys.iloc[i].station_id,old_nll=values[0],scale_crossfit_nll=values[1]))
    f=pd.DataFrame(rows);f.to_csv(out/'scale_crossfit_profile_nll.csv',index=False)
    d=f.groupby('station_id').mean(numeric_only=True);delta=d.scale_crossfit_nll-d.old_nll
    rng=np.random.default_rng(42);boot=delta.to_numpy()[rng.integers(len(delta),size=(10000,len(delta)))].mean(1)
    report.update(station_mean_nll_change=float(delta.mean()),ci_lower=float(np.quantile(boot,.025)),ci_upper=float(np.quantile(boot,.975)))
    (out/'scale_crossfit_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['point','scale'],required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    out=ROOT/'paper_outputs/ess_major_revision/experiments'/('smoke' if a.smoke else 'formal')/a.stage
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise SystemExit('Completed outputs exist; preserve them before a deliberate rerun.')
    start=time.time();data=load('space_time_holdout',a.smoke)
    (point if a.stage=='point' else scale)(a,out,data)
    (out/'complete.json').write_text(json.dumps(dict(stage=a.stage,smoke=a.smoke,seconds=time.time()-start,status='smoke_not_evidence' if a.smoke else 'posthoc_sensitivity',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),selection='2024 early-validation stations only; external data never selected')),encoding='utf-8')
    print(out)

if __name__=='__main__':main()
