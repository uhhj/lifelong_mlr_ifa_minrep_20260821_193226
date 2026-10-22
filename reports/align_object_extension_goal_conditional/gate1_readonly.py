import os,json,time,sys,hashlib,subprocess
from pathlib import Path
# bc_labels.py is injected by the local wrapper; it is not installed on the host.
torch.set_num_threads(1)
inventory={**{'OBJECT_'+str(s):'/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed%d_20261019'%s for s in (41,42,43)},
 'GOAL_41':'/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed41_20261019',
 'GOAL_42':'/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed42_20261018',
 'GOAL_43':'/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed43_20261019'}
result=dict(status='PASS',runs={},observed_unix=time.time(),optimizer_calls=0,model_forwards=0,environment_steps=0,server_writes=0)
for key,path in inventory.items():
 P=Path(path);cfg=js(P/'resolved_ra_coreset.json');counts={};tasks={};protected={str(P/'CONTRACT.json'):sha(P/'CONTRACT.json')}
 for t in range(6,10):
  rows,proof=derive(P,t);tasks[str(t)]=proof
  st=js(P/'ra_coreset/storage'/('task%d.json'%t));assert st['K']==len(st['selected'])
  for x in st['selected']:assert sha(x['source'])==sha(P/'ra_coreset/buffer'/x['destination'])==x['sha256']
  counts[str(t)]=st['K'];protected[str(P/'bindings'/('task%d.json'%t))]=proof['binding_sha256']
 result['runs'][key]=dict(status='PASS',parent=path,seed=cfg['seed'],benchmark_name=cfg['benchmark_name'],gamma=cfg['gamma'],pretrain=cfg['pretrain'],tasks=tasks,quotas=counts,protected=protected)
assert not torch.cuda.is_initialized()
result['cuda_initialized']=False
result['gpu']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
assert not result['gpu'].strip()
print(json.dumps(result))
