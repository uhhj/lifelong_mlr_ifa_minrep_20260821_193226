"""Export only error magnitudes for independent local recomputation, never state or predictions."""
import json,hashlib
from pathlib import Path
import torch,numpy as np
torch.set_num_threads(1)
Q=Path('/data/mlr_ifa/experiment_workspaces/PHYSICAL_EFFECT_PROBE_OBJECT41_20261023_engineering03');O=Q/'output'
r=json.loads((O/'STATE_PROBE_RESULTS.json').read_text());cases={}
for row in r['results']:
 task=row['task'];kind=row['model'];tr=row['train_view'];te=row['test_view'];data=torch.load(O/'private'/('labels%d.pt'%task),map_location='cpu')
 pp=O/'private/probes'/('%s_%s_%s_task%d_predictions.pth'%(tr,te,kind,task));raw=torch.load(pp,map_location='cpu');ids=raw['ids'];y=data['q'][ids].numpy();yp=data['previous_q'][ids].numpy();modes={}
 for mode,p in raw['predictions'].items():
  p=p.numpy();err=np.stack([np.linalg.norm(p[:,:3]-y[:,:3],axis=1),np.linalg.norm(p[:,3:6]-y[:,3:6],axis=1),np.abs(p[:,6]-y[:,6])],-1)*1000
  prev=np.stack([np.linalg.norm(p[:,:3]-yp[:,:3],axis=1),np.linalg.norm(p[:,3:6]-yp[:,3:6],axis=1)],-1)*1000
  modes[mode]=dict(error_mm=err.tolist(),previous_target_error_mm=prev.tolist())
 key='%s_%s_%s_task%d'%(tr,te,kind,task)
 cases[key]=dict(task=task,model=kind,train_view=tr,test_view=te,ids=ids.tolist(),demo_ids=[data['metadata'][int(i)]['demo_id'] for i in ids],dynamic=data['dynamic'][ids].tolist(),modes=modes,prediction_file_sha256=hashlib.sha256(pp.read_bytes()).hexdigest())
print(json.dumps(dict(status='ERROR_MAGNITUDES_ONLY',cases=cases,source_results_sha256=hashlib.sha256((O/'STATE_PROBE_RESULTS.json').read_bytes()).hexdigest(),model_forwards=0,optimizer_calls=0,environment_steps=0,server_writes=0),separators=(',',':')))
