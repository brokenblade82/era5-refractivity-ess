"""Frozen external inference and native-height background sensitivity; never fit externally."""
from pathlib import Path
import os,sys,json,hashlib,argparse
os.environ.setdefault('OMP_NUM_THREADS','8')
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/revision2'),str(ROOT/'scripts/ess_submission')]
import numpy as np
import pandas as pd
import joblib
import xarray as xr
import yaml
from scipy.stats import norm
from igra_forecast.revision2_baselines import make_row_features,LEVELS,row_training_mask
from igra_forecast.revision2_ess_repair import batch_weights,weighted_values,geometric_height
OUT=ROOT/'paper_outputs/ess_evidence_revision'
FORMAL=ROOT/'paper_outputs/ess_major_revision/experiments/formal'
REPAIR=ROOT/'results_mapping/revision2_ess_repair'

def save_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str),encoding='utf-8')

def grouped(frame,errors,group):
    rows=[]
    for name,e in errors.items():
        f=frame[group].copy();f['n']=1;f['sse']=e**2;f['ae']=abs(e);f['error']=e
        a=f.groupby(group,observed=True,dropna=False)[['n','sse','ae','error']].sum().reset_index();a['model']=name;rows.append(a)
    return pd.concat(rows,ignore_index=True)

def report_metrics(f):
    f=f.copy();f['rmse']=np.sqrt(f.sse/f.n);f['mae']=f.ae/f.n;f['bias']=f.error/f.n
    return f

def cosmic():
    from evaluate_ess_external_suite import _six_level_frame
    folder=OUT/'cosmic';folder.mkdir(parents=True,exist_ok=True)
    models={k:v for k,v in joblib.load(FORMAL/'point/point_models.joblib').items() if k.startswith('hgb_')}
    levels=np.array(yaml.safe_load((ROOT/'configs/revision2/ess.yaml').read_text())['cosmic2_height']['era5_pressure_levels_hpa'])
    ii=[np.flatnonzero(levels==l)[0] for l in LEVELS];files=sorted((REPAIR/'cosmic_cache').glob('*/*.parquet'))
    records=[];audit=[]
    for j,path in enumerate(files):
        op=folder/f'{path.parent.name}_{path.stem}.csv'
        predpath=folder/f'{path.parent.name}_{path.stem}.parquet'
        if op.exists() and predpath.exists():
            records.append(pd.read_csv(op));continue
        cached=pd.read_parquet(path)
        frozen=pd.read_parquet(REPAIR/'evaluation/cosmic2'/path.parent.name/path.name)
        frozen=frozen.loc[frozen.variant.eq('B')].reset_index(drop=True)
        keys=['profile_id','time','height_m','source_archive']
        pd.testing.assert_frame_equal(cached[keys].reset_index(drop=True),frozen[keys],check_dtype=False)
        c={k:np.stack(cached[k]) for k in ['temperature','humidity','n_dry','n_wet','n_total']}
        c['height']=np.stack(cached.era5_geopotential_height_m);c['surface_pressure']=cached.surface_pressure.to_numpy()
        six=_six_level_frame(cached,c,levels);x=make_row_features(six)
        # Repeated target heights share identical six-level features. Exact unique
        # rows reduce inference cost without changing the model inputs.
        unique,inverse=np.unique(x,axis=0,return_inverse=True)
        w,support=batch_weights(np.stack(cached.era5_geometric_height_m)[:,ii],cached.height_m.to_numpy(),LEVELS[None,:]*100.0<=c['surface_pressure'][:,None])
        old=weighted_values(w,np.stack(cached.hgb_dry6)+np.stack(cached.hgb_wet6),support)
        mask=frozen.evaluation_mask.to_numpy(bool)
        assert np.allclose(old[mask],frozen.hgb_correction_total.to_numpy()[mask],atol=1e-10)
        f=frozen.loc[mask].copy();r=f.residual_total.to_numpy()
        errors={'ERA5':-r,'original_hgb':f.hgb_correction_total.to_numpy()-r}
        predictions=f[keys+['residual_total','era5_n_height']].copy()
        for name,(mm,no_utc) in models.items():
            xx=np.delete(unique,[14,15],axis=1) if no_utc else unique
            pred=sum(m.predict(xx) for m in mm)[inverse].reshape(-1,6)
            correction=weighted_values(w,pred,support)[mask]
            assert np.isfinite(correction).all()
            errors[name]=correction-r
            predictions[name]=correction
        predictions.to_parquet(predpath,index=False)
        a=grouped(f,errors,['source_archive','height_band','utc_support'])
        a.to_csv(op,index=False);records.append(a)
        audit.append(dict(file=str(path.relative_to(ROOT)),rows=len(f),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if j%40==0:print(f'COSMIC {j+1}/{len(files)}',flush=True)
    allrows=pd.concat(records,ignore_index=True)
    for name,keys in [('overall',['model']),('archive',['source_archive','model']),('height',['height_band','model']),('utc',['utc_support','model'])]:
        t=allrows.groupby(keys)[['n','sse','ae','error']].sum().reset_index();report_metrics(t).to_csv(folder/f'{name}.csv',index=False)
    arch=allrows.groupby(['source_archive','model'])[['n','sse']].sum().reset_index();p=arch.pivot(index='source_archive',columns='model',values='sse');n=arch.query("model=='ERA5'").set_index('source_archive').n.reindex(p.index).to_numpy()
    rng=np.random.default_rng(42);idx=rng.integers(len(p),size=(10000,len(p)));den=n[idx].sum(1);contrasts=[]
    for a,b in [(k,'ERA5') for k in models]+[('hgb_components_no_utc','hgb_components_utc'),('hgb_total_no_utc','hgb_total_utc')]:
        d=np.sqrt(p[a].to_numpy()[idx].sum(1)/den)-np.sqrt(p[b].to_numpy()[idx].sum(1)/den)
        contrasts.append(dict(contrast=a+' minus '+b,estimate=np.sqrt(p[a].sum()/n.sum())-np.sqrt(p[b].sum()/n.sum()),lower=np.quantile(d,.025),upper=np.quantile(d,.975),archive_months=len(p)))
    pd.DataFrame(contrasts).to_csv(folder/'archive_paired_contrasts.csv',index=False)
    assert int(n.sum())==2044949
    save_json(folder/'complete.json',dict(blocks=len(files),rows=int(n.sum()),model_sha256=hashlib.sha256((FORMAL/'point/point_models.joblib').read_bytes()).hexdigest(),evaluation='fixed B background and exact original evaluation mask; archive-month paired bootstrap; no external fitting or selection',new_cache_audit=audit))
    print(pd.read_csv(folder/'overall.csv').to_string(index=False),flush=True)

def seasonal():
    from run_major_revision_experiments import load
    tr=load('space_time_holdout',False)['train'];tr=tr.loc[row_training_mask(tr)].copy();tr['month']=pd.to_datetime(tr.time).dt.month;tr['hour']=pd.to_datetime(tr.time).dt.hour
    f=pd.read_parquet(ROOT/'results_mapping/revision2_ess_review/predictions/trajectory.parquet');f=f.loc[f.evaluation_mask&f.direction.eq('ascending')].copy();f['month']=pd.to_datetime(f.time).dt.month;f['hour']=pd.to_datetime(f.time).dt.hour
    f['day']=pd.to_datetime(f.segment_start_time).dt.floor('D');r=f.residual_total.to_numpy();rows=[]
    for no_utc in [False,True]:
        keys=['pressure_hpa','month']+([] if no_utc else ['hour']);lookup=tr.groupby(keys).residual_total.mean();fallback=tr.groupby('pressure_hpa').residual_total.mean()
        c=lookup.reindex(pd.MultiIndex.from_frame(f[keys])).to_numpy();missing=~np.isfinite(c);c[missing]=f.loc[missing,'pressure_hpa'].map(fallback)
        a=f[['day']].assign(sse=(c-r)**2,sse0=r*r,n=1).groupby('day').sum();rng=np.random.default_rng(42);idx=rng.integers(len(a),size=(10000,len(a)));den=a.n.to_numpy()[idx].sum(1)
        d=np.sqrt(a.sse.to_numpy()[idx].sum(1)/den)-np.sqrt(a.sse0.to_numpy()[idx].sum(1)/den)
        rows.append(dict(model='seasonal_'+('no_utc' if no_utc else 'utc'),n=len(f),fallback_rows=int(missing.sum()),rmse=np.sqrt(np.mean((c-r)**2)),bias=np.mean(c-r),delta=np.sqrt(np.mean((c-r)**2))-np.sqrt(np.mean(r*r)),lower=np.quantile(d,.025),upper=np.quantile(d,.975),days=len(a)))
    pd.DataFrame(rows).to_csv(OUT/'rapsodi_seasonal.csv',index=False);print(pd.DataFrame(rows).to_string(index=False),flush=True)

def dense():
    from check_rapsodi_representation import weights
    from collocate_external_profiles import open_era5_variable,concatenate_time_boundaries
    from evaluate_cosmic2_height import preload_month_arrays,collocate_pressure_profiles
    folder=OUT/'dense_rapsodi';folder.mkdir(parents=True,exist_ok=True)
    f=pd.read_parquet(ROOT/'results_mapping/revision2_ess_review/predictions/trajectory.parquet');f=f.loc[f.evaluation_mask&f.direction.eq('ascending')].copy()
    R=joblib.load(ROOT/'results_mapping/revision2_confirmatory/space_time_holdout/hgb_prob_hetero_structured_run42/model.joblib')['correlation_total']
    cfg=yaml.safe_load((ROOT/'configs/revision2/data_pipeline.yaml').read_text());root=(ROOT/cfg['era5']['root']).resolve()
    levels=np.array(yaml.safe_load((ROOT/'configs/revision2/ess.yaml').read_text())['cosmic2_height']['era5_pressure_levels_hpa'])
    candidates=[]
    with xr.open_zarr(ROOT/'data/revision2/raw/rapsodi/rapsodi_level2.zarr') as ds:
        assert ds.height.attrs['standard_name']=='geopotential_height'
        z=geometric_height(np.asarray(ds.height,float))
        for pid,g in f.groupby('profile_id'):
            raw=int(g.raw_index.iloc[0]);g=g.sort_values('height_m');z6=geometric_height(g.height_m.to_numpy())
            T=np.asarray(ds.ta.isel(launch_time=raw),float);q=np.asarray(ds.q.isel(launch_time=raw),float);p=np.asarray(ds.p.isel(launch_time=raw),float)/100
            n=77.6*p/T+3.73e5*(p*q/(.622+.378*q))/T**2
            keep=np.isfinite(n)&np.isfinite(T)&np.isfinite(q)&(q>=0)&(T>153.15)&(T<333.15)&(z>=z6.min())&(z<=z6.max())
            w=weights(z[keep],z6);ii=[list(LEVELS).index(int(l)) for l in g.pressure_hpa];sd=g.hgb_prediction_std.to_numpy();cov=R[np.ix_(ii,ii)]*sd[:,None]*sd[None,:]
            times=pd.to_datetime(np.asarray(ds.interpolated_time.isel(launch_time=raw)))[keep]
            part=pd.DataFrame(dict(profile_id=pid,time=times,latitude=np.asarray(ds.lat.isel(launch_time=raw),float)[keep],longitude=np.asarray(ds.lon.isel(launch_time=raw),float)[keep],height_m=z[keep],observed_n=n[keep],obs6=w@g.observed_n.to_numpy(),background6=w@g.era5_n.to_numpy(),correction=w@g.hgb_correction_total.to_numpy(),sd=np.sqrt(np.einsum('ij,jk,ik->i',w,cov,w)),day=str(pd.Timestamp(g.segment_start_time.iloc[0]).date())))
            candidates.append(part)
    target=pd.concat(candidates,ignore_index=True);target['period']=target.time.dt.strftime('%Y%m');allout=[];audits=[]
    valid=np.isfinite(target.latitude)&np.isfinite(target.longitude)&target.time.notna();invalid=int((~valid).sum());target=target.loc[valid].copy()
    for month,part in target.groupby('period'):
        opened=[];arrays={};period=pd.Period(month,freq='M')
        try:
            for logical in ['temperature','specific_humidity','geopotential','surface_pressure']:
                current=open_era5_variable(root,period,cfg,logical);opened.append(current);previous=following=None
                if part.time.min()<pd.Timestamp(current.time.min().values):previous=open_era5_variable(root,period-1,cfg,logical);opened.append(previous)
                if part.time.max()>pd.Timestamp(current.time.max().values):following=open_era5_variable(root,period+1,cfg,logical);opened.append(following)
                arrays[logical],a=concatenate_time_boundaries(current,part.time,previous,following);audits.append(dict(month=month,variable=logical,available_levels=list(map(int,current.level.values)) if logical!='surface_pressure' else [],boundary=a))
            arrays,preload=preload_month_arrays(arrays,part,levels)
            for start in range(0,len(part),2500):
                chunk=part.iloc[start:start+2500].copy();c=collocate_pressure_profiles(chunk,arrays,levels);coord=geometric_height(c['height']);supported=(levels[None,:]*100.<=c['surface_pressure'][:,None])&np.isfinite(c['n_total'])&(c['n_total']>0)
                w,support=batch_weights(coord,chunk.height_m.to_numpy(),supported);chunk['background16']=weighted_values(w,c['n_total'],support,log=True);chunk['background16_linear']=weighted_values(w,c['n_total'],support);chunk['dense_support']=support
                ii=[np.flatnonzero(levels==l)[0] for l in LEVELS];w6,support6=batch_weights(coord[:,ii],chunk.height_m.to_numpy(),supported[:,ii]);chunk['background6_trajectory_log']=weighted_values(w6,c['n_total'][:,ii],support6,log=True);chunk['background6_trajectory_linear']=weighted_values(w6,c['n_total'][:,ii],support6);allout.append(chunk)
            print(f'Dense RAPSODI {month}: {len(part)} heights',flush=True)
        finally:
            for a in opened:a.close()
    result=pd.concat(allout,ignore_index=True);mask=result.dense_support&np.isfinite(result.background16);valid=result.loc[mask].copy();profiles=[]
    for kind,bg,obs in [('six_background_native','background6','observed_n'),('dense_background_native','background16','observed_n'),('six_background_sixobs','background6','obs6'),('dense_background_sixobs','background16','obs6')]:
        er=valid[bg]+valid.correction-valid[obs];a=valid[['profile_id','day']].assign(n=1,sse=er**2,error=er,covered=(abs(er)<=norm.ppf(.95)*valid.sd).astype(int),width=2*norm.ppf(.95)*valid.sd).groupby(['profile_id','day']).sum().reset_index();a['kind']=kind;profiles.append(a)
    profiles=pd.concat(profiles,ignore_index=True);profiles.to_csv(folder/'profile_metrics.csv',index=False)
    a=profiles.groupby('kind').sum(numeric_only=True);a['rmse']=np.sqrt(a.sse/a.n);a['bias']=a.error/a.n;a['coverage']=a.covered/a.n;a['mean_width']=a.width/a.n;a.to_csv(folder/'summary.csv')
    result.to_parquet(folder/'native_comparisons.parquet',index=False)
    common=valid.loc[np.isfinite(valid.background6_trajectory_log)].copy();factor=[]
    for bg in ['background6_trajectory_log','background6_trajectory_linear','background16','background16_linear']:
        e=common[bg]+common.correction-common.observed_n
        a=common[['profile_id','day']].assign(n=1,sse=e**2,error=e,covered=(abs(e)<=norm.ppf(.95)*common.sd).astype(int),width=2*norm.ppf(.95)*common.sd).groupby(['profile_id','day']).sum().reset_index();a['kind']=bg;factor.append(a)
    factor=pd.concat(factor,ignore_index=True);factor.to_csv(folder/'factorial_profile_metrics.csv',index=False)
    fa=factor.groupby('kind').sum(numeric_only=True);fa['rmse']=np.sqrt(fa.sse/fa.n);fa['coverage']=fa.covered/fa.n;fa['mean_width']=fa.width/fa.n;fa.to_csv(folder/'factorial_summary.csv')
    save_json(folder/'complete.json',dict(candidate_heights=sum(len(c) for c in candidates),invalid_trajectory=invalid,common_heights=len(valid),profiles=valid.profile_id.nunique(),days=valid.day.nunique(),unsupported_dense=int((~mask).sum()),levels=levels.tolist(),scope='Native 10 m observation trajectory; 16 pressure-level ERA5 N log-linearly interpolated in geometric height; original six-level correction and covariance fixed; not native-height recalibration or a retrieval forward operator',archive_audit=audits))
    print(fa[['n','rmse','coverage','mean_width']].to_string(),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['cosmic','seasonal','dense']);args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True);globals()[args.stage]()
