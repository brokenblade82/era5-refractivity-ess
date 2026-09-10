"""Frozen diagnostics for reviewer revision. No model fitting except a tiny Ridge identity check."""
from pathlib import Path
import os,sys,json,hashlib,argparse,time
os.environ.setdefault('OMP_NUM_THREADS','8')
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/revision2'),str(ROOT/'scripts/ess_submission')]
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import joblib,yaml
from scipy.special import ndtr
from scipy.stats import norm
from igra_forecast.revision2_baselines import LEVELS,make_row_features,row_training_mask
from igra_forecast.revision2_confirmatory import predict_raw_scale,apply_level_scales
from igra_forecast.revision2_ess_repair import batch_weights,weighted_values,geometric_height
from run_revision2_baselines import phase_rows
from audit_ars_revision_evidence import profile_scores,masked_nll
O=ROOT/'paper_outputs/ess_reviewer_revision';O.mkdir(exist_ok=True)
B=ROOT/'results_mapping/revision2_confirmatory';F=ROOT/'paper_outputs/ess_major_revision/experiments/formal'
KEYS=['station_id','time','pressure_hpa'];PK=['station_id','time'];Z=norm.ppf(.95)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,x):
 (O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),encoding='utf-8')
def save(n,f):f.to_csv(O/n,index=False)
def valid(path):
 f=pd.read_parquet(path);f=f.loc[f.evaluation_mask].copy();assert not f.duplicated(KEYS).any();return f.sort_values(KEYS).reset_index(drop=True)
def phase(name):
 license=pd.read_csv(ROOT/'data/revision2/manifests/protocol_license_matrix.csv');r=license.loc[(license.protocol=='space_time_holdout')&(license.phase==name)].iloc[0]
 f=phase_rows(ds.dataset(ROOT/'data/revision2/processed/era5_igra_profiles',format='parquet',partitioning='hive'),pd.read_csv(ROOT/'data/revision2/manifests/station_split.csv'),r,None,None)
 return f.loc[row_training_mask(f)].sort_values(KEYS).reset_index(drop=True)
def ci(v):
 v=np.asarray(v,float);idx=np.random.default_rng(42).integers(len(v),size=(10000,len(v)));q=np.quantile(v[idx].mean(1),[.025,.975]);return dict(estimate=v.mean(),lower=q[0],upper=q[1],clusters=len(v))
def budget(d,w):
 d=np.asarray(d);w=np.asarray(w);e=d+w
 r=dict(n=len(e),dry_mse=np.mean(d*d),wet_mse=np.mean(w*w),cross=2*np.mean(d*w),mse=np.mean(e*e),rmse=np.sqrt(np.mean(e*e)),bias=np.mean(e),bias_squared=np.mean(e)**2,centered_variance=np.var(e),dry_bias_squared=np.mean(d)**2,wet_bias_squared=np.mean(w)**2,dry_variance=np.var(d),wet_variance=np.var(w),centered_cross=2*np.mean((d-d.mean())*(w-w.mean())))
 assert np.isclose(r['mse'],r['dry_mse']+r['wet_mse']+r['cross'],atol=1e-10)
 assert np.isclose(r['mse'],r['bias_squared']+r['centered_variance'],atol=1e-10)
 return r
def blocks():
 s=pd.read_csv(ROOT/'data/revision2/manifests/station_split.csv');sel=pd.read_csv(ROOT/'docs/revision2_confirmatory/spatial_block_manifest.csv')
 s['block']='lat'+np.clip(np.floor((s.latitude+90)/30).astype(int),0,5).astype(str)+'_lon'+np.clip(np.floor((s.longitude%360)/60).astype(int),0,5).astype(str)
 counts=s.groupby('block').size();s['block_stations']=s.block.map(counts);s['selected']=s.block.isin(sel.block);s['reason']=np.where(s.selected,'selected',np.where(s.block_stations<5,'below_minimum','outside_top20_after_southern_reservation'))
 save('station_block_membership.csv',s);g=s.groupby(['block','reason']).size().reset_index(name='stations');save('block_accounting.csv',g)
 assert len(s)==630 and s.selected.sum()==579
 rows=[];aud=[]
 for block in sel.block:
  inblock=s.block.eq(block)
  for role,mask in [('evaluation',inblock),('training',~inblock&s['split'].eq('train')),('early_validation',~inblock&s['split'].eq('early_val')),('calibration',~inblock&s['split'].eq('calibration'))]:
   ids=s.loc[mask,'station_id'];rows.extend(dict(block=block,role=role,station_id=i) for i in ids)
   aud.append(dict(block=block,role=role,stations=len(ids)))
  assert not set(s.loc[inblock,'station_id'])&set(s.loc[~inblock&s['split'].eq('train'),'station_id'])
 save('block_role_membership.csv',pd.DataFrame(rows));save('block_role_counts.csv',pd.DataFrame(aud))
 # Independent bootstrap grid is defined by the existing analysis, not the block-training grid.
 dump('block_rules.json',dict(grid='30 degree latitude by 60 degree longitude, longitude modulo 360',minimum_stations=5,target=20,southern_reservation=2,selected=579,below_minimum=21,other_omitted=30,early_validation='outside block intersect original early_val; HGB itself uses automatic row-level early stopping in its permitted training rows'))
def internal():
 ref=valid(B/'space_time_holdout/hgb_run42/predictions.parquet');f=phase('evaluation');pd.testing.assert_frame_equal(f[KEYS],ref[KEYS],check_dtype=False);assert np.allclose(f.residual_total,ref.residual_total)
 models=joblib.load(F/'point/point_models.joblib');rows=[];pr=f[KEYS+['residual_dry','residual_wet','residual_total']].copy()
 predictions={'ERA5':(np.zeros(len(f)),np.zeros(len(f))),'reference':(ref.prediction_dry.to_numpy(),ref.prediction_wet.to_numpy())}
 for name in ['hgb_components_utc','hgb_components_no_utc']:
  mm,no_utc=models[name];x=make_row_features(f);x=np.delete(x,[14,15],axis=1) if no_utc else x
  a,b=[m.predict(x.astype(float)) for m in mm];old=pd.read_parquet(F/'point'/f'space_time_holdout_{name}.parquet').sort_values(KEYS).reset_index(drop=True)
  pd.testing.assert_frame_equal(f[KEYS],old[KEYS],check_dtype=False);assert np.allclose(a+b,old.correction,atol=1e-9);predictions[name]=(a,b)
 for name,(a,b) in predictions.items():
  pr[name+'_dry']=a;pr[name+'_wet']=b;d=a-f.residual_dry.to_numpy();w=b-f.residual_wet.to_numpy()
  rows.append(dict(model=name,pressure='all',**budget(d,w)))
  for lev in LEVELS:
   m=f.pressure_hpa.eq(lev);rows.append(dict(model=name,pressure=str(lev),**budget(d[m],w[m])))
 pr.to_parquet(O/'component_predictions.parquet',index=False);save('component_budgets.csv',pd.DataFrame(rows))
 iterations=[]
 for p in [B/'space_time_holdout/hgb_run42/model.joblib']+sorted((B/'space_time_holdout').glob('hgb_prob_*/model.joblib')):
  m=joblib.load(p);objs=m.get('mean_models',m)
  for name,obj in objs.items():
   if hasattr(obj,'n_iter_'):iterations.append(dict(file=str(p.relative_to(ROOT)),part='mean_'+name,iterations=obj.n_iter_,maximum=obj.max_iter))
  for name,item in m.get('scale_models',{}).items():
   obj=item.get('model')
   if obj is not None:iterations.append(dict(file=str(p.relative_to(ROOT)),part='scale_'+name,iterations=obj.n_iter_,maximum=obj.max_iter))
 for name,item in models.items():
  if name.startswith('hgb_'):
   for j,obj in enumerate(item[0]):iterations.append(dict(file=str((F/'point/point_models.joblib').relative_to(ROOT)),part=f'{name}_{j}',iterations=obj.n_iter_,maximum=obj.max_iter))
 save('actual_iterations.csv',pd.DataFrame(iterations))
def probability():
 root=B/'space_time_holdout/hgb_prob_hetero_structured_run42';f=valid(root/'predictions.parquet');cov=pd.read_parquet(root/'profile_covariance.parquet');m=joblib.load(root/'model.joblib')
 scores=profile_scores(f,cov).rename(columns={'nll':'reference_nll'})
 values=f.assign(error=f.prediction_total-f.residual_total).pivot(index=PK,columns='pressure_hpa',values='error').reindex(columns=LEVELS);sd=f.pivot(index=PK,columns='pressure_hpa',values='std_total').reindex(columns=LEVELS).to_numpy();e=values.to_numpy();mask=np.isfinite(e);sd=np.nan_to_num(sd,nan=0.)
 rr=np.loadtxt(F/'scale/crossfit_scale_correlation.csv',delimiter=',');new=masked_nll(e,sd[:,:,None]*rr*sd[:,None,:],mask);diag=masked_nll(e,sd[:,:,None]*np.eye(6)*sd[:,None,:],mask)
 pd.testing.assert_frame_equal(scores[PK],values.index.to_frame(index=False));scores['crossfit_nll']=new;scores['diagonal_nll']=diag;scores['mask_bits']=(mask*(1<<np.arange(6))).sum(1)
 scores.to_parquet(O/'keyed_profile_scores.parquet',index=False)
 old=pd.read_csv(F/'scale/scale_crossfit_profile_nll.csv');a=scores.groupby('station_id')[['reference_nll','crossfit_nll']].mean();b=old.groupby('station_id')[['old_nll','scale_crossfit_nll']].mean();assert np.allclose(a.to_numpy(),b.reindex(a.index).to_numpy(),atol=1e-8)
 save('keyed_probability_contrasts.csv',pd.DataFrame([dict(contrast=x+' minus '+y,**ci(scores.assign(delta=scores[x]-scores[y]).groupby('station_id').delta.mean())) for x,y in [('crossfit_nll','reference_nll'),('reference_nll','diagonal_nll'),('crossfit_nll','diagonal_nll')]]))
 cal=phase('calibration');cs=apply_level_scales(cal,predict_raw_scale(m['scale_models']['total'],cal),m['calibration_scales']['total']);cuts={str(l):np.quantile(cs[cal.pressure_hpa.eq(l)],[.2,.4,.6,.8]).tolist() for l in LEVELS};dump('calibration_scale_quintiles.json',cuts)
 f['covered']=(abs(f.prediction_total-f.residual_total)<=Z*f.std_total).astype(float);f['width']=2*Z*f.std_total
 f['scale_bin']=[int(np.searchsorted(cuts[str(int(l))],s,side='right'))+1 for l,s in zip(f.pressure_hpa,f.std_total)]
 f[KEYS+['residual_total','prediction_total','std_total','covered','width','scale_bin']].to_parquet(O/'conditional_coverage_rows.parquet',index=False)
 records=[]
 for group in ['pressure_hpa','scale_bin']:
  groups=f.groupby(group) if group=='pressure_hpa' else f.groupby(['pressure_hpa','scale_bin'])
  for key,g in groups:
   t=g.groupby('station_id').agg(n=('covered','size'),hits=('covered','sum'),width=('width','sum'));idx=np.random.default_rng(42).integers(len(t),size=(10000,len(t)));den=t.n.to_numpy()[idx].sum(1);boot=t.hits.to_numpy()[idx].sum(1)/den
   q=np.quantile(boot,[.025,.975]);records.append(dict(group=group,key=str(key),n=len(g),stations=len(t),coverage=g.covered.mean(),lower=q[0],upper=q[1],width=g.width.mean()))
 save('conditional_coverage.csv',pd.DataFrame(records));save('station_coverage.csv',f.groupby('station_id').agg(n=('covered','size'),coverage=('covered','mean'),width=('width','mean')).reset_index())
def humidity():
 p=pd.read_parquet(O/'component_predictions.parquet');parts=[]
 for file in sorted((ROOT/'data/revision2/processed/igra_profiles').rglob('*.parquet')):
  f=pd.read_parquet(file,columns=KEYS+['humidity_source','pressure_flag','temperature_flag','height_flag']);g=f.merge(p[KEYS],on=KEYS,how='inner',validate='one_to_one')
  if len(g):parts.append(g)
 h=pd.concat(parts,ignore_index=True);assert not h.duplicated(KEYS).any();p=p.merge(h,on=KEYS,how='left',validate='one_to_one');assert p.humidity_source.notna().all()
 for col in ['pressure_flag','temperature_flag','height_flag']:p[col]=p[col].fillna('').astype(str).str.strip()
 save('humidity_sources.csv',p.groupby(['pressure_hpa','humidity_source']).size().reset_index(name='n'));save('humidity_flags.csv',p.groupby(['pressure_hpa','pressure_flag','temperature_flag','height_flag']).size().reset_index(name='n'))
 masks={'all':np.ones(len(p),bool),'pressure_temperature_checked':p.pressure_flag.isin(['A','B'])&p.temperature_flag.isin(['A','B'])}
 for source in p.humidity_source.unique():masks['source_'+source]=p.humidity_source.eq(source)
 rows=[]
 for subset,mask in masks.items():
  g=p.loc[mask];
  if not len(g):continue
  for model in ['ERA5','reference','hgb_components_utc','hgb_components_no_utc']:
   for label,gg in [('all',g)]+[(str(l),g.loc[g.pressure_hpa.eq(l)]) for l in LEVELS]:
    if not len(gg):continue
    d=gg[model+'_dry']-gg.residual_dry;w=gg[model+'_wet']-gg.residual_wet
    st=gg.assign(se=(d+w)**2,se0=gg.residual_total**2).groupby('station_id')[['se','se0']].mean();r=ci(np.sqrt(st.se)-np.sqrt(st.se0));rows.append(dict(subset=subset,model=model,pressure=label,**budget(d,w),station_delta=r['estimate'],lower=r['lower'],upper=r['upper'],stations=r['clusters']))
 result = pd.DataFrame(rows)
 result['interval_status'] = np.where(result.stations < 10, 'not_reported_fewer_than_10_stations', 'reported')
 result.loc[result.stations < 10, ['lower', 'upper']] = np.nan
 save('humidity_sensitivity.csv',result);p[KEYS+list(h.columns.difference(KEYS))].to_parquet(O/'humidity_row_audit.parquet',index=False)
def external():
 from check_rapsodi_representation import weights
 rap=pd.read_parquet(ROOT/'results_mapping/revision2_ess_review/predictions/trajectory.parquet');rap=rap.loc[rap.evaluation_mask&rap.direction.eq('ascending')].copy();rap['cluster']=pd.to_datetime(rap.segment_start_time,utc=True).dt.strftime('%Y-%m-%d')
 outputs=[];stats=[]
 def collect(f,e0,e1,sd0,sd1,dataset,group):
  for label,mask in [('all',np.ones(len(f),bool))]+[(str(k),f[group].eq(k).to_numpy()) for k in sorted(f[group].unique())]:
   g=f.loc[mask];a=e0[mask];b=e1[mask];
   for name,e in [('ERA5',a),('reference',b)]:
    t=pd.DataFrame(dict(cluster=g.cluster.to_numpy(),n=1,error=e,sse=e*e));t=t.groupby('cluster').sum().reset_index();t['dataset']=dataset;t['group']=label;t['model']=name;outputs.append(t)
   if sd0 is not None:
    for name,sd in [('diagonal',sd0[mask]),('reference',sd1[mask])]:
     t=pd.DataFrame(dict(cluster=g.cluster.to_numpy(),n=1,covered=(abs(b)<=Z*sd).astype(int),width=2*Z*sd));t=t.groupby('cluster').sum().reset_index();t['dataset']=dataset;t['group']=label;t['model']=name;stats.append(t)
 collect(rap,-rap.residual_total.to_numpy(),(rap.hgb_correction_total-rap.residual_total).to_numpy(),None,None,'RAPSODI_standard','pressure_hpa')
 native=pd.read_parquet(ROOT/'paper_outputs/ess_evidence_revision/dense_rapsodi/native_comparisons.parquet');native=native.loc[native.dense_support&np.isfinite(native.background16)].copy();native['cluster']=native.day.astype(str);native['height_band']=pd.cut(native.height_m,[-np.inf,2000,5000,np.inf],right=False,labels=['below2km','2to5km','5kmplus']).astype(str);native['sd_diagonal']=np.nan
 for pid,g in native.groupby('profile_id'):
  p=rap.loc[rap.profile_id.eq(pid)].sort_values('height_m');w=weights(g.height_m.to_numpy(),geometric_height(p.height_m.to_numpy()));sd=p.hgb_prediction_std.to_numpy();native.loc[g.index,'sd_diagonal']=np.sqrt((w*w)@(sd*sd))
 assert np.isfinite(native.sd_diagonal).all()
 collect(native,(native.background16-native.observed_n).to_numpy(),(native.background16+native.correction-native.observed_n).to_numpy(),native.sd_diagonal.to_numpy(),native.sd.to_numpy(),'RAPSODI_native','height_band')
 # Only covariance propagation is recomputed. No inference or background collocation.
 levels=np.array(yaml.safe_load((ROOT/'configs/revision2/ess.yaml').read_text())['cosmic2_height']['era5_pressure_levels_hpa']);ii=[np.flatnonzero(levels==l)[0] for l in LEVELS]
 files=sorted((ROOT/'results_mapping/revision2_ess_repair/cosmic_cache').glob('*/*.parquet'));assert len(files)==617
 for j,path in enumerate(files):
  c=pd.read_parquet(path);f=pd.read_parquet(ROOT/'results_mapping/revision2_ess_repair/evaluation/cosmic2'/path.parent.name/path.name);f=f.loc[f.variant.eq('B')].reset_index(drop=True);keys=['profile_id','time','height_m','source_archive'];pd.testing.assert_frame_equal(c[keys],f[keys],check_dtype=False)
  w,support=batch_weights(np.stack(c.era5_geometric_height_m)[:,ii],c.height_m.to_numpy(),LEVELS[None,:]*100<=c.surface_pressure.to_numpy()[:,None]);cov=np.stack(c.hgb_covariance6).reshape(-1,6,6);v=np.einsum('ni,nij,nj->n',w,cov,w);vd=np.sum(w*w*np.diagonal(cov,axis1=1,axis2=2),axis=1);mask=f.evaluation_mask.to_numpy(bool)
  assert np.allclose(np.sqrt(v[mask]),f.loc[mask,'hgb_prediction_std'],atol=1e-8)
  f=f.loc[mask].copy();f['cluster']=f.source_archive.astype(str);collect(f,-f.residual_total.to_numpy(),(f.hgb_correction_total-f.residual_total).to_numpy(),np.sqrt(vd[mask]),np.sqrt(v[mask]),'COSMIC2','height_band')
  if j%100==0:print('Frozen covariance blocks',j+1,'/',len(files),flush=True)
 a=pd.concat(outputs).groupby(['dataset','group','model','cluster'],as_index=False)[['n','error','sse']].sum();save('external_cluster_sums.csv',a);b=pd.concat(stats).groupby(['dataset','group','model','cluster'],as_index=False)[['n','covered','width']].sum();save('external_covariance_cluster_sums.csv',b)
 result=[]
 for keys,g in a.groupby(['dataset','group','model']):
  n=g.n.sum();bias=g.error.sum()/n;mse=g.sse.sum()/n;result.append(dict(dataset=keys[0],group=keys[1],model=keys[2],n=n,rmse=np.sqrt(mse),mse=mse,bias=bias,bias_squared=bias*bias,centered_variance=mse-bias*bias))
 save('external_error_decomposition.csv',pd.DataFrame(result));result=[]
 for keys,g in b.groupby(['dataset','group','model']):
  idx=np.random.default_rng(42).integers(len(g),size=(10000,len(g)));boot=g.covered.to_numpy()[idx].sum(1)/g.n.to_numpy()[idx].sum(1);q=np.quantile(boot,[.025,.975]);result.append(dict(dataset=keys[0],group=keys[1],model=keys[2],n=g.n.sum(),clusters=len(g),coverage=g.covered.sum()/g.n.sum(),width=g.width.sum()/g.n.sum(),lower=q[0],upper=q[1]))
 save('external_covariance_sensitivity.csv',pd.DataFrame(result))
def ridge_check():
 from sklearn.linear_model import Ridge
 from sklearn.preprocessing import StandardScaler
 rng=np.random.default_rng(42);x=StandardScaler().fit_transform(rng.normal(size=(256,18)));a=rng.normal(size=256);b=rng.normal(size=256);kw=dict(alpha=1.,solver='svd');p=Ridge(**kw).fit(x,a).predict(x)+Ridge(**kw).fit(x,b).predict(x);q=Ridge(**kw).fit(x,a+b).predict(x);delta=float(np.max(abs(p-q)));assert delta<1e-10;dump('ridge_identity.json',dict(max_absolute_difference=delta,rows=256,features=18,scope='algebraic numerical identity; not a new performance experiment'))
def main():
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['blocks','internal','probability','humidity','external','ridge','all']);args=parser.parse_args();cfg=ROOT/'configs/reviewer_revision.json';lock=dict(config_sha256=sha(cfg),code_sha256=sha(Path(__file__)))
 if (O/'run_lock.json').exists():assert json.loads((O/'run_lock.json').read_text())==lock,'Code/config changed: audit outputs before rerunning.'
 else:dump('run_lock.json',lock)
 for name in (['blocks','internal','probability','humidity','external','ridge'] if args.stage=='all' else [args.stage]):
  if (O/(name+'_complete.json')).exists():print('SKIP',name);continue
  start=time.time()
  try:
   {'blocks':blocks,'internal':internal,'probability':probability,'humidity':humidity,'external':external,'ridge':ridge_check}[name]();dump(name+'_complete.json',dict(status='complete',seconds=time.time()-start,**lock));print('COMPLETE',name,flush=True)
  except Exception:
   import traceback;dump(name+'_error.json',dict(traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
