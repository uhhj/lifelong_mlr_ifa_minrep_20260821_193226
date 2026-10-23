"""Independent CPU recomputation of complete A evidence. No writes, forwards or optimizer."""
import os,sys,json,hashlib,math,importlib.util
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
Q=Path('/data/mlr_ifa/experiment_workspaces/PHYSICAL_EFFECT_PROBE_OBJECT41_20261023_engineering03');O=Q/'output'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def js(p):return json.loads(Path(p).read_text())
c=js(Q/'CONTRACT.json');assert all(sha(p)==h for p,h in c['files'].items())
assert (Q/'controller.exit').read_text().strip()=='0'
raw=js(O/'STATE_PROBE_RESULTS.json');gate=js(O/'A_GATE.json');cost=js(O/'COST_AUDIT.json');splits=js(O/'DEMO_SPLITS.json')
binding=js(O/'INPUT_BINDING_AUDIT.json');native_manifest=js(O/'NATIVE_VIEW_MANIFEST.json');input_count=0
P=Path('/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed41_20261019')
for task in (6,7):
 b=js(P/'bindings'/('task%d.json'%task));assert sha(b['hdf5'])==b['hdf5_sha256']
 for name,record in b['windows'].items():assert sha(P/'features'/str(task)/name)==record['sha256'];input_count+=1
 assert sha(O/'private'/('labels%d.pt'%task))==binding['tasks'][str(task)]['labels_sha256']
 nm=next(x for x in native_manifest['tasks'] if x['task']==task);assert sha(O/'private'/('native%d.pt'%task))==nm['sha256']
 frames=torch.load(O/'private'/('native%d.pt'%task),map_location='cpu');assert len(frames)==len(nm['per_window_sha256'])
 assert all(hashlib.sha256(v.numpy().tobytes()).hexdigest()==h for v,h in zip(frames,nm['per_window_sha256']))
spec=importlib.util.spec_from_file_location('independent_readout',Q/'physical_effect_probe/readout_models.py');models=importlib.util.module_from_spec(spec);spec.loader.exec_module(models)
assert len(raw['results'])==30 and cost['probe_fits']==100 and cost['probe_epochs']==5000 and cost['policy_updates']==cost['environment_steps']==0
events=[json.loads(s) for s in (O/'EVENTS.jsonl').read_text().splitlines()];epochs={}
for e in events:
 if e['event']=='probe_epoch':epochs.setdefault(e['key'],[]).append(e)
assert len(epochs)==100
for key,es in epochs.items():assert [v['epoch'] for v in es]==list(range(50))
recomputed={};fingerprints={};observations=0
for row in raw['results']:
 task=row['task'];kind=row['model'];train=row['train_view'];test=row['test_view'];data=torch.load(O/'private'/('labels%d.pt'%task),map_location='cpu');meta=data['metadata']
 pp=O/'private/probes'/('%s_%s_%s_task%d_predictions.pth'%(train,test,kind,task));preds=torch.load(pp,map_location='cpu');ids=preds['ids'].numpy();valid=data['mask'][:,0].numpy()
 assert set(ids.tolist())==set(np.flatnonzero(valid).tolist()) and len(ids)==int(valid.sum())
 assert [v['fold'] for v in splits[str(task)]]==list(range(5))
 expected_ids=[]
 for f,sel in zip(splits[str(task)],row['folds']):
  es=epochs[sel['key']];best=min(es,key=lambda v:v['validation_position_m']);assert sel['selected_epoch']==best['epoch']
  expected_ids.extend(i for i,r in enumerate(meta) if r['state_valid'] and r['demo_id'] in f['test'])
  nt=sum(r['state_valid'] and r['demo_id'] in f['train'] for r in meta);assert sel['updates']==math.ceil(nt/128)*50==es[-1]['updates']
  checkpoint=O/'private/probes'/sel['key']/'best.pth';assert sha(checkpoint)==sel['checkpoint_sha256'];cp=torch.load(checkpoint,map_location='cpu')
  assert cp['epoch']==best['epoch'] and cp['fold']==f and cp['contract_sha256']==sha(Q/'CONTRACT.json')
  assert all(torch.isfinite(v).all() for v in cp['model'].values());fresh=models.build(kind);fresh.load_state_dict(cp['model'],strict=True);fingerprints[str(checkpoint)]=sha(checkpoint)
 assert ids.tolist()==expected_ids
 ds=np.array([meta[int(i)]['demo_id'] for i in ids]);dynamic=data['dynamic'][ids].numpy();y=data['q'][ids].numpy();yp=data['previous_q'][ids].numpy()
 errors={}
 for mode,p in preds['predictions'].items():
  p=p.numpy();assert p.shape==y.shape and np.isfinite(p).all()
  ee=np.stack([np.linalg.norm(p[:,:3]-y[:,:3],axis=1),np.linalg.norm(p[:,3:6]-y[:,3:6],axis=1),np.abs(p[:,6]-y[:,6])],axis=1)*1000
  pe=np.stack([np.linalg.norm(p[:,:3]-yp[:,:3],axis=1),np.linalg.norm(p[:,3:6]-yp[:,3:6],axis=1)],axis=1)*1000
  errors[mode]={}
  for j,name in enumerate(('object_eef','object_goal','aperture')):
   target=row['interventions'][mode][name]
   for stratum,mask in [('all',np.ones(len(ids),dtype=bool)),('dynamic',dynamic),('static',~dynamic)]:
    z=target[stratum];assert z['n']==int(mask.sum())
    if mask.any():
     assert np.isclose(z['median_mm'],np.median(ee[mask,j]),rtol=1e-5,atol=1e-4)
     assert np.isclose(z['p90_mm'],np.quantile(ee[mask,j],.9),rtol=1e-5,atol=1e-4)
    for demo,v in z['by_demo'].items():
     dm=mask&(ds==demo);assert v['n']==int(dm.sum());assert np.isclose(v['median_mm'],np.median(ee[dm,j]),rtol=1e-5,atol=1e-4)
     if j<2:assert np.isclose(v['previous_target_median_mm'],np.median(pe[dm,j]),rtol=1e-5,atol=1e-4)
   errors[mode][name]=dict(median=float(np.median(ee[:,j])),p90=float(np.quantile(ee[:,j],.9)),by_demo={dd:float(np.median(ee[ds==dd,j])) for dd in sorted(set(ds))},dynamic_current={dd:float(np.median(ee[dynamic&(ds==dd),j])) for dd in sorted(set(ds[dynamic]))},dynamic_previous={dd:float(np.median(pe[dynamic&(ds==dd),j])) for dd in sorted(set(ds[dynamic]))} if j<2 else {})
  observations+=len(ids)
 if train==test=='native':recomputed[task,kind]=errors
 fingerprints[str(pp)]=sha(pp)
checks={}
for kind in ('P2','P3'):
 rows=[]
 for task in (6,7):
  for name in ('object_eef','object_goal'):
   n=recomputed[task,kind]['normal'][name];base=recomputed[task,'P0']['normal'][name];keys=sorted(n['by_demo'])
   dif=np.array([base['by_demo'][k]-n['by_demo'][k] for k in keys]);rng=np.random.RandomState(410023);boot=dif[rng.randint(0,10,(10000,10))].mean(1);ci=np.quantile(boot,[.025,.975])
   dyn=n['dynamic_current'];alt=recomputed[task,kind]['latest_visual_previous'][name]['dynamic_current'];lag=float(np.mean([n['dynamic_previous'][k]-v for k,v in dyn.items()]));vis=float(np.mean([alt[k]-v for k,v in dyn.items()]))
   passed=bool(ci[0]>0 and n['median']<=10 and n['p90']<=25 and len(dyn)>0 and lag>0 and vis>0)
   rows.append(dict(task=task,target=name,median_mm=n['median'],p90_mm=n['p90'],ci95_improvement_mm=ci.tolist(),dynamic_lag_advantage_mm=lag,dynamic_visual_advantage_mm=vis,passed=passed))
  
 checks[kind]=dict(passed=all(x['passed'] for x in rows),checks=rows)
selected=next((k for k in ('P2','P3') if checks[k]['passed']),None);assert selected==gate['selected']
assert len(fingerprints)==130
costs=cost['costs'];assert sum(r['updates'] for r in costs)==cost['probe_updates']
lengths=[v['frames'] for task in binding['tasks'].values() for v in task['physical'].values()]
actual_cost=dict(probe_training_forward_batches=sum(r['updates'] for r in costs),probe_validation_forward_batches=sum(50*math.ceil(r['validation_windows']/128) for r in costs),probe_selected_test_forward_batches=sum((6 if r['key'].startswith('original_') else 3)*math.ceil(r['test_windows']/128) for r in costs),native_unique_observations=sum(lengths),native_repeat_check_observations=sum(min(n,32) for n in lengths),native_forward_batches=sum(math.ceil(n/32)+1 for n in lengths),frozen_temporal_embedding_batches=6*sum(math.ceil(v['population']/128) for v in binding['tasks'].values()),frozen_temporal_embedding_windows=6*sum(v['population'] for v in binding['tasks'].values()),visibility_temporal_forward_batches=14,visibility_light_forward_batches=9,visibility_input_backward_calls=3,policy_updates=0,environment_episodes=0)
print(json.dumps(dict(status='PASS_COMPLETE_A_INDEPENDENT_RECOMPUTATION',selected=selected,gate_status=gate['status'],checks=checks,metrics_recomputed=30,intervention_predictions_checked=observations,probe_fits=100,probe_epochs=5000,probe_updates=cost['probe_updates'],original_feature_files_checked=input_count,actual_cost=actual_cost,files=fingerprints,source_results_sha256=sha(O/'STATE_PROBE_RESULTS.json'),contract_sha256=sha(Q/'CONTRACT.json'),cpu_only=not torch.cuda.is_initialized(),model_forwards=0,optimizer_calls=0,environment_steps=0,server_writes=0)))
