"""Controlled six-level representation check, without claiming a dense ERA5 test.

Keeps the frozen six-level ERA5 background, correction and covariance fixed,
and substitutes either the native 10-m RAPSODI N or six-level interpolated N.
This isolates observation-profile representation in this synthetic comparison.
It cannot reproduce a fully trajectory-collocated dense ERA5 comparison.
"""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import xarray as xr
import joblib
from scipy.stats import norm
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
OUT=ROOT/'paper_outputs/ess_major_revision/statistics'

def weights(z,levels):
    right=np.searchsorted(levels,z,side='right').clip(1,len(levels)-1);left=right-1
    f=(z-levels[left])/(levels[right]-levels[left]);w=np.zeros((len(z),len(levels)))
    w[np.arange(len(z)),left]=1-f;w[np.arange(len(z)),right]=f
    return w

def main():
    f=pd.read_parquet(ROOT/'results_mapping/revision2_ess_review/predictions/trajectory.parquet')
    f=f.loc[f.evaluation_mask & f.direction.eq('ascending')].copy()
    R=joblib.load(ROOT/'results_mapping/revision2_confirmatory/space_time_holdout/hgb_prob_hetero_structured_run42/model.joblib')['correlation_total']
    rows=[];order=[1000,925,850,700,500,300];radius=6371000.
    with xr.open_zarr(ROOT/'data/revision2/raw/rapsodi/rapsodi_level2.zarr') as d:
        if d.height.attrs.get('standard_name')!='geopotential_height':raise ValueError('Height definition changed')
        H=np.asarray(d.height,float);z=radius*H/(radius-H)
        for pid,g in f.groupby('profile_id'):
            raw=int(g.raw_index.iloc[0]);g=g.sort_values('height_m')
            # Raw RAPSODI height and saved height_m are geopotential (PTU) heights.
            h6=g.height_m.to_numpy();z6=radius*h6/(radius-h6)
            if not np.all(np.diff(z6)>0):raise ValueError('Nonmonotone standard heights')
            p=np.asarray(d.p.isel(launch_time=raw),float)/100
            T=np.asarray(d.ta.isel(launch_time=raw),float);q=np.asarray(d.q.isel(launch_time=raw),float)
            e=p*q/(.622+.378*q);n=77.6*p/T+3.73e5*e/T**2
            keep=np.isfinite(n)&np.isfinite(T)&np.isfinite(q)&(q>=0)&(T>153.15)&(T<333.15)&(z>=z6.min())&(z<=z6.max())
            zz=z[keep];nn=n[keep];w=weights(zz,z6)
            indices=[order.index(int(x)) for x in g.pressure_hpa]
            sig=g.hgb_prediction_std.to_numpy();cov=R[np.ix_(indices,indices)]*sig[:,None]*sig[None,:]
            sd=np.sqrt(np.einsum('ij,jk,ik->i',w,cov,w))
            background=w@g.era5_n.to_numpy();correction=w@g.hgb_correction_total.to_numpy()
            obs6=w@g.observed_n.to_numpy()
            native_error=nn-background-correction;interp_error=obs6-background-correction;eta=nn-obs6
            for name,er in [('native_10m_observation',native_error),('sixlevel_observation',interp_error)]:
                rows.append(dict(profile_id=pid,kind=name,n=len(er),sse=np.sum(er**2),covered=np.sum(abs(er)<=norm.ppf(.95)*sd),width_sum=np.sum(2*norm.ppf(.95)*sd),eta_sse=np.sum(eta**2),eta_sum=np.sum(eta),cross_sum=np.sum(interp_error*eta),error_sum=np.sum(er)))
            if not np.allclose(native_error,interp_error+eta,atol=1e-10):raise AssertionError('Representation closure failed')
    result=pd.DataFrame(rows);OUT.mkdir(parents=True,exist_ok=True);result.to_csv(OUT/'rapsodi_representation_profiles.csv',index=False)
    summary=[]
    for kind,g in result.groupby('kind'):
        a=g.sum(numeric_only=True);summary.append(dict(kind=kind,n=int(a.n),profiles=len(g),rmse=float(np.sqrt(a.sse/a.n)),coverage=float(a.covered/a.n),width=float(a.width_sum/a.n),eta_rmse=float(np.sqrt(a.eta_sse/a.n)),eta_mean=float(a.eta_sum/a.n),uncentered_cross=float(2*a.cross_sum/a.n)))
    (OUT/'rapsodi_representation_summary.json').write_text(json.dumps(dict(scope='controlled observation representation only; fixed six-level background, not dense ERA5 collocation; no native-height calibration claim',summaries=summary),indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
