import os,json,time,hashlib,subprocess
from pathlib import Path
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
parents={**{'OBJECT_'+str(s):'/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed%d_20261019'%s for s in (41,42,43)},
 'GOAL_41':'/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed41_20261019',
 'GOAL_42':'/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed42_20261018',
 'GOAL_43':'/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed43_20261019'}
out=dict(observed_unix=time.time(),parents={},processes=[],source_files={},new_training=0,model_forwards=0,environment_steps=0,server_writes=0)
for key,path in parents.items():
 p=Path(path);d={'path':path,'exists':p.is_dir()}
 if p.is_dir():
  for n in ('AUDIT.json','AUTONOMOUS_EXIT_ACCEPTANCE.json','PROCESS_EXIT_ACCEPTANCE.json','resolved_ra_coreset.json','resolved_eval.json','CONTRACT.json','TASKS.json'):
   if (p/n).is_file():
    v=js(p/n);d[n]=v if n.startswith('resolved') or n=='TASKS.json' else {k:v[k] for k in ('status','method','source_sha256','epochs','optimizer_steps','formal_records','metrics','baseline') if k in v}
  d['storage']={str(t):dict(K=js(p/'ra_coreset/storage'/('task%d.json'%t))['K'],count=len(js(p/'ra_coreset/storage'/('task%d.json'%t))['selected'])) for t in range(6,10)}
  d['epochs']=js(p/'ra_coreset/epochs.json')[-1]
  d['feature_bytes']=sum(f.stat().st_size for f in (p/'features').rglob('*.pt'))
  d['buffer_bytes']=sum(f.stat().st_size for f in (p/'ra_coreset/buffer').glob('*.pt'))
  d['binding']={str(t):{'keys':list(js(p/'bindings'/('task%d.json'%t)).keys()),'window_count':len(js(p/'bindings'/('task%d.json'%t))['windows']),
    'example_sanitized':{k:v for k,v in next(iter(js(p/'bindings'/('task%d.json'%t))['windows'].values())).items() if k not in ('last_action','actions','states')}} for t in range(6,10)}
 out['parents'][key]=d
S=Path('/data/mlr_ifa/experiment_workspaces/CAETv3_TAC_EFTR_ALIGN_OBJECT41_20261022_collate01/ALIGN/source')
for n in ('coreset_exp/common.py','coreset_exp/model.py','coreset_exp/train.py','coreset_exp/evaluate.py','ra_exp/audit.py','ra_exp/storage.py','caet_v3_exp/variant.py'):
 out['source_files'][n]=(S/n).read_text()
out['source_root']=str(S)
out['source_sha256']=js('/data/mlr_ifa/experiments/CAETv3_ALIGN_OBJECT_seed41_20261022_collate01/CONTRACT.json')['source_sha256']
out['disk']={m:dict(ordinary=(v:=os.statvfs(m)).f_bavail*v.f_frsize,root=v.f_bfree*v.f_frsize) for m in ('/data','/')}
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True)
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace');cwd=str((p/'cwd').resolve())
  if int(p.name)!=os.getpid() and 'python -c' not in cmd and ('/experiment_workspaces/' in cmd or '/experiment_workspaces/' in cwd):out['processes'].append(dict(pid=int(p.name),command=cmd,cwd=cwd))
 except OSError:pass
print(json.dumps(out))
