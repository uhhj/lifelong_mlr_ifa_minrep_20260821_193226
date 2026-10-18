"""Existing latent distance diagnostics on CPU; no policy forward/training/environment."""
from pathlib import Path
import json,time,hashlib,importlib.util
import numpy as np,torch
torch.set_num_threads(2)
O=Path('/root/mlr_ifa/experiments/SCv2_Coreset_GOAL_seed42_20261018')
SRC=Path('/root/mlr_ifa/experiment_workspaces/SCv2_Coreset_GOAL_seed42_20261018/source/libero/lifelong/selectors/greedy_kcenter.py')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p):return json.loads(p.read_text())
spec=importlib.util.spec_from_file_location('frozen_distance',SRC);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
proof={str(SRC):sha(SRC)};out={};start=time.time()
for t in [6,7,8]:
 mp=O/'sc_coreset/storage'/f'task{t}.json';bp=O/'bindings'/f'task{t}.json';proof[str(mp)]=sha(mp);proof[str(bp)]=sha(bp)
 m=js(mp)['selection'];b=js(bp)['windows'];paths=sorted((O/'features'/str(t)).glob('*.pt'),key=lambda p:p.name);names=[p.name for p in paths];idx={n:i for i,n in enumerate(names)}
 vv=[];aa=[]
 for p in paths:
  h=sha(p);assert h==b[p.name]['sha256'];proof[str(p)]=h
  x=torch.load(p,map_location='cpu');vv.append(x['feature'][0,:,1:3,:].numpy());aa.append(x['actions'][0,-1,:].numpy())
 v=np.asarray(vv,dtype=np.float64);a=np.asarray(aa,dtype=np.float64);s=m['original_selection']
 # Reuse exact frozen all-task scales. No fitting/recalibration or selector execution.
 d=mod.Distance.__new__(mod.Distance);d.v=(v/np.linalg.norm(v,axis=-1,keepdims=True)).reshape(len(v),-1)
 d.a=a/np.asarray(s['action_scale']);d.hs=s['visual_scale'];d.as_=s['action_distance_scale'];d.block=256
 before=d.distances([idx[n] for n in m['original_ids']]).min(axis=1)
 after=d.distances([idx[n] for n in m['selected_ids']]).min(axis=1)
 assert abs(before.max()-m['before']['radius'])<1e-12 and abs(before.mean()-m['before']['mean'])<1e-12
 assert abs(after.max()-m['after']['radius'])<1e-12 and abs(after.mean()-m['after']['mean'])<1e-12
 delta=after-before;tol=1e-12
 def summary(ix):
  x=delta[ix];return dict(n=len(ix),worse=int((x>tol).sum()),better=int((x< -tol).sum()),same=int((abs(x)<=tol).sum()),
   mean_before=float(before[ix].mean()),mean_after=float(after[ix].mean()),max_increase=float(x.max()),sum_increase=float(np.maximum(x,0).sum()),sum_decrease=float(np.minimum(x,0).sum()))
 groups={}
 for i,n in enumerate(names):
  r=b[n];g=min(3,int(4*r['target']/max(1,r['length']-1)))*3+r['group']%3
  groups.setdefault(str(g),[]).append(i)
 removed=set(m['original_ids'])-set(m['selected_ids']);kept=set(m['original_ids'])&set(m['selected_ids'])
 out[t]=dict(all=summary(list(range(len(names)))),not_removed_centers=summary([i for i,n in enumerate(names) if n not in removed]),
  groups={g:summary(ix) for g,ix in groups.items()},
  changed_center_metadata=[dict(sample_id=n,kind='removed' if n in removed else 'added',demo=b[n]['demo_id'],target=b[n]['target'],length=b[n]['length'],nearest_before=float(before[idx[n]]),nearest_after=float(after[idx[n]])) for n in sorted(removed|(set(m['selected_ids'])-kept))])
 del vv,aa,v,a,d
assert all(sha(Path(p))==h for p,h in proof.items())
result=dict(status='PASS',cpu_only=True,model_forwards=0,optimizer_updates=0,environment_calls=0,input_sha256=proof,files_unchanged=True,coverage=out,seconds=time.time()-start)
dest=Path('/tmp/scv2_local_coverage_20261018.json');dest.write_text(json.dumps(result,indent=2)+'\n');print(dest)
