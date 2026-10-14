"""CPU-only post-hoc audit; reads existing payloads, never imports model/env or saves payloads."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
import json,hashlib,importlib.util,time,sys
from pathlib import Path
import numpy as np
import torch,h5py
torch.set_num_threads(4)
DEST=Path(sys.argv[1]);DEST.mkdir(parents=True,exist_ok=True)
assert not (DEST/'coverage.json').exists()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def stat(a): return dict(mean=float(a.mean()),p50=float(np.median(a)),p95=float(np.percentile(a,95)),maximum=float(a.max()))
results=[];hashes={};begin=time.time()
for seed in (42,43,44):
 root=Path(f'/data/mlr_ifa/experiments/Coreset_paper_OBJECT_seed{seed}_20261013')
 src=Path(f'/data/mlr_ifa/experiment_workspaces/Coreset_paper_OBJECT_seed{seed}_20261013/source/libero/lifelong/selectors/greedy_kcenter.py')
 hashes[str(src)]=sha(src)
 spec=importlib.util.spec_from_file_location('frozen_selector',src);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 fm=read(root/'FEATURES.json');tasks=read(root/'TASKS.json')
 for p in (root/'FEATURES.json',root/'TASKS.json',root/'CONTRACT.json',root/'MODAL_CONTRACT.json'):hashes[str(p)]=sha(p)
 for task in range(6,10):
  files=sorted((root/'features'/str(task)).glob('*.pt'));names=[p.name for p in files];vv=[];aa=[]
  for p in files:
   h=sha(p);assert h==fm['hashes'][f'{task}/{p.name}'];hashes[str(p)]=h
   x=torch.load(p,map_location='cpu');vv.append(x['feature'][0,:,1:3,:].numpy());aa.append(x['actions'][0,-1,:].numpy())
  hdf=Path(tasks[str(task)]['paths'][0]);hashes[str(hdf)]=sha(hdf)
  with h5py.File(hdf,'r') as f: population=np.concatenate([f['data'][d]['actions'][:] for d in fm['demos'][str(task)]])
  distance=mod.Distance(vv,aa,population,seed=42)
  row={'seed':seed,'task':task,'N':len(files),'arms':{}}
  for arm in ('random','coreset'):
   mp=root/arm/'storage'/f'task{task}.json';hashes[str(mp)]=sha(mp);m=read(mp)
   chosen=[names.index(Path(v['source']).name) for v in m['selected']]
   assert len(set(chosen))==m['K']
   for e in m['selected']:
    assert hashes[e['source']]==e['sha256']
    bp=root/arm/'buffer'/e['destination'];h=sha(bp);assert h==e['sha256'];hashes[str(bp)]=h
   if arm=='coreset':
    saved=m['selection'];assert np.allclose(saved['action_scale'],distance.action_scale,rtol=0,atol=1e-12)
    assert abs(saved['visual_scale']-distance.hs)<1e-12 and abs(saved['action_distance_scale']-distance.as_)<1e-12
   nearest=np.full(len(files),np.inf);near_v=np.full(len(files),np.inf);near_a=np.full(len(files),np.inf)
   for start in range(0,len(chosen),64):
    ids=np.asarray(chosen[start:start+64]);nearest=np.minimum(nearest,distance.distances(ids).min(1))
    v=np.maximum(0,1-distance.v@distance.v[ids].T/16)/distance.hs
    a=np.sqrt(np.mean((distance.a[:,None,:]-distance.a[ids][None,:,:])**2,axis=-1))/distance.as_
    v[ids,np.arange(len(ids))]=0.;a[ids,np.arange(len(ids))]=0.
    near_v=np.minimum(near_v,v.min(1));near_a=np.minimum(near_a,a.min(1))
   unselected=np.ones(len(files),dtype=bool);unselected[chosen]=False
   row['arms'][arm]={'K':len(chosen),'joint_all':stat(nearest),'joint_unselected':stat(nearest[unselected]),'visual_only_all':stat(near_v),'action_only_all':stat(near_a)}
   if arm=='coreset':assert abs(nearest.max()-saved['covering_radius'][-1])<1e-10
  row['C_vs_R_max_reduction_fraction']=1-row['arms']['coreset']['joint_all']['maximum']/row['arms']['random']['joint_all']['maximum']
  row['C_vs_R_mean_reduction_fraction']=1-row['arms']['coreset']['joint_all']['mean']/row['arms']['random']['joint_all']['mean']
  results.append(row);print(json.dumps(row),flush=True)
assert all(sha(Path(p))==h for p,h in hashes.items())
out={'status':'PASS','rows':results,'inputs_before_after_unchanged':True,'checked_files':len(hashes),'seconds':time.time()-begin,'scope':'Existing features, action demonstrations, storage manifests and retained buffer only. No model checkpoint/environment/training loaded. Task9 storage is post-training and cannot explain within-run improvement.'}
(DEST/'coverage.json').write_text(json.dumps(out,indent=2)+'\n')
(DEST/'input_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='rows'}),flush=True)
