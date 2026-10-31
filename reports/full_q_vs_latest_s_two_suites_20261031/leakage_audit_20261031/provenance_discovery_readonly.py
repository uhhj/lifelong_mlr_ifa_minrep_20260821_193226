PARENTS={'GOAL': {'output': '/data/mlr_ifa/experiments/ALIGN_GOAL_seed41_20261022', 'source': '/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/GOAL_41/source'}, 'OBJECT': {'output': '/data/mlr_ifa/experiments/CAETv3_ALIGN_OBJECT_seed41_20261022_collate01', 'source': '/data/mlr_ifa/experiment_workspaces/CAETv3_TAC_EFTR_ALIGN_OBJECT41_20261022_collate01/ALIGN/source'}}
import os,json,hashlib,time
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
out={}
for suite,p in PARENTS.items():
 q=Path(p['output']);cfg=json.loads((q/'resolved_ra_coreset.json').read_text());paths={}
 def scan(x,key=''):
  if isinstance(x,dict):
   for k,v in x.items():scan(v,key+'/'+k)
  elif isinstance(x,(list,tuple)):
   for i,v in enumerate(x):scan(v,key+'/'+str(i))
  elif isinstance(x,str) and (x.startswith('/') or any(z in key.lower() for z in ['suite','task','pretrain','demo','train'])):paths[key]=x
 scan(cfg)
 audits={}
 for t in range(6,10):
  a=q/('effects/current/task%d/AUDIT.json'%t);j=json.loads(a.read_text());audits[str(t)]=dict(path=str(a),sha256=sha(a),metadata=j)
 out[suite]=dict(output=str(q),source=p['source'],config_paths=paths,audits=audits)
print(json.dumps(dict(status='READ_ONLY_PROVENANCE_DISCOVERY',observed=time.time(),suites=out,optimizer_calls=0,environment_calls=0)))
