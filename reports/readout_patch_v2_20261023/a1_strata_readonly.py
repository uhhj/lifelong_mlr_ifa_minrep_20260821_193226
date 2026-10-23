"""Existing heldout P3 predictions only; scalar errors, no model forwards or writes."""
import os,sys,json,hashlib
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/PHYSICAL_EFFECT_PROBE_OBJECT41_20261023_engineering03')
os.environ['PEP_ROOT']=str(Q);sys.path.insert(0,str(Q))
from physical_effect_probe.common import verify,sha
import torch,numpy as np
contract=verify();cases={}
def summary(e):
 return dict(n=len(e),median_mm=np.median(e,axis=0).tolist(),p90_mm=np.quantile(e,.9,axis=0).tolist(),mean_mm=e.mean(0).tolist()) if len(e) else dict(n=0)
for task in (6,7):
 O=Q/'output';d=torch.load(O/'private'/('labels%d.pt'%task),map_location='cpu');pf=O/'private/probes'/('native_native_P3_task%d_predictions.pth'%task);p=torch.load(pf,map_location='cpu');ids=p['ids'];v=p['predictions']['normal'];err=(v-d['q'][ids]).numpy()*1000
 e=np.stack([np.linalg.norm(err[:,:3],axis=1),np.linalg.norm(err[:,3:6],axis=1),abs(err[:,6])],-1)
 meta=[d['metadata'][int(i)] for i in ids];demo=[r['demo_id'] for r in meta]
 # Fixed temporal thirds proxy, not physical contact/grasp annotation.
 maxidx={dd:max(r['last_observation_index'] for r in d['metadata'] if r['demo_id']==dd) for dd in set(demo)}
 phase=np.array([min(2,int(3*r['last_observation_index']/max(1,maxidx[r['demo_id']]+1))) for r in meta])
 cases['task%d'%task]=dict(task=task,object='butter' if task==6 else 'milk',target='basket',prediction_sha256=sha(pf),labels_sha256=sha(O/'private'/('labels%d.pt'%task)),pooled=summary(e),axis_absolute=summary(abs(err)),axis_signed_mean_mm=err.mean(0).tolist(),temporal_thirds={str(k):summary(e[phase==k]) for k in range(3)},by_demo={dd:summary(e[np.array(demo)==dd]) for dd in sorted(set(demo))},dynamic={str(k):summary(e[d['dynamic'][ids].numpy()==k]) for k in (False,True)},row_count=len(ids))
assert verify()==contract
print(json.dumps(dict(status='PASS_READONLY_STRATIFIED_EXISTING_P3',cases=cases,contract_sha256=contract,units='mm',model_forwards=0,optimizer_updates=0,environment_steps=0,server_writes=0,limitations=['Temporal thirds are progress proxies, not semantic operation/contact stages.','Per-demo differences combine layout, trajectory and occlusion; cannot attribute them to layout alone.','No validated visibility/occlusion labels exist in this diagnostic; no occlusion-stratified claim.','Absolute object and goal positions are not independently identified by relative-coordinate targets.'])))
