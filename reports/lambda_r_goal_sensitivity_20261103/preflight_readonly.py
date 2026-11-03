"""Read-only bindings/resource/scheduler inspection; no model or environment calls."""
import os,sys,json,time,hashlib,subprocess,inspect,contextlib,io
from pathlib import Path
NAME='S_LAMBDA_R_GOAL41_42_43_20261103_v1'
Q=Path('/data/mlr_ifa/experiment_workspaces')/NAME
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text())
assert not Q.exists()
object_root=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1')
assert (object_root/'engineering_cross_lambda_20261102_v1/controller.exit').read_text().strip()=='0'
roots={s:Path('/data/mlr_ifa/experiment_workspaces')/n for s,n in [
 ('GOAL','SUPPORT_GATE_FULL_S_GOAL41_42_43_20261029_v1')]}
completed_c=Path('/data/mlr_ifa/experiment_workspaces/C_MATCH_FULL_S_BUDGET_GOAL_OBJECT_20261030_v1')
assert (completed_c/'engineering_suite_order_20261030_v1/controller.exit').read_text().strip()=='0'
completed_q=Path('/data/mlr_ifa/experiment_workspaces/FULL_Q_NO_GATE_GOAL_OBJECT_20261031_v2')
assert (completed_q/'controller.exit').read_text().strip()=='0'
qc=js(completed_q/'CONTRACT.json');assert all(sha(p)==h for p,h in qc['files'].items())
bindings={};previous={};checked={};diagnostics={};existing_sizes={}
for suite,root in roots.items():
 private=Path('/data/mlr_ifa/experiments')/root.name
 existing_sizes[suite]=[dict(path=str(p.relative_to(private)),bytes=p.stat().st_size) for p in private.rglob('*') if p.is_file() and not p.is_symlink()]

for suite,root in roots.items():
 exit_path=root/('controller.exit' if suite=='GOAL' else 'engineering_copy_20261029_v1/controller.exit')
 assert exit_path.read_text().strip()=='0'
 contract=js(root/'CONTRACT.json')
 assert all(sha(p)==h for p,h in contract['files'].items())
 previous[suite]=dict(root=str(root),contract_sha256=sha(root/'CONTRACT.json'),exit_sha256=sha(exit_path))
 for seed in (41,42,43):
  p=root/('seed%d'%seed)/'PARENT_BINDING.json';b=js(p)
  for path,h in b['files'].items():
   if path not in checked: checked[path]=sha(path)
   assert checked[path]==h,path
  for n,h in b['source_sha256'].items():
   path=str(Path(b['source'])/n)
   if path not in checked:checked[path]=sha(path)
   assert checked[path]==h,path
  cfg=js(Path(b['output'])/'resolved_ra_coreset.json')
  ec=js(Path(b['output'])/'resolved_eval.json')
  assert cfg['seed']==seed and ec['seed']==seed
  assert ec['benchmark_name'].lower()=='libero_'+suite.lower()
  case=suite+'_'+str(seed)
  records=[]
  for t in range(6,10):
   for i,row in enumerate(js(Path(b['output'])/('ra_coreset/dataset_manifests/task%d.json'%t))['rows']):
    records.append(dict(stage=t,index=i,task=row['task'],bc_valid=row['bc_valid'],sha256=row['sha256']))
  diagnostics[case+'_manifest_metadata']=records
  ref={}
  for n in ['INITIAL_S.json','TRAIN_S.json','EPOCHS_S.json','STAGE_S_6.json','OUTCOMES.json','train_S.exit','evaluate.exit']:
   fp=root/('seed%d'%seed)/n;ref[n]=dict(sha256=sha(fp),path=str(fp))
  assert (root/('seed%d'%seed)/'train_S.exit').read_text().strip()=='0'
  assert (root/('seed%d'%seed)/'evaluate.exit').read_text().strip()=='0'
  st=js(root/('seed%d'%seed)/'STAGE_S_6.json');assert sha(st['model'])==st['sha256']
  ref['stage6_checkpoint']=dict(path=st['model'],sha256=st['sha256'])
  bindings[case]=dict(parent_binding_path=str(p),parent_binding_sha256=sha(p),output=b['output'],source=b['source'],stages=b['stages'],train_config=cfg['train'],eval_benchmark=ec['benchmark_name'],accepted_S_reference=ref)
sys.path.insert(0,str(roots['GOAL']/'seed41'))
with contextlib.redirect_stdout(io.StringIO()):
 import runtime
 import libero.lifelong.algos.mlr as module
scheduler_sources={}
for name,value in vars(module).items():
 if inspect.isclass(value) and hasattr(value,'start_task'):
  method=value.start_task
  scheduler_sources[name]=dict(path=inspect.getsourcefile(method),source=inspect.getsource(method),file_sha256=sha(inspect.getsourcefile(method)))
processes=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if int(p.name)!=os.getpid() and any(x in cmd for x in [str(Q),*(str(x) for x in roots.values()),'SUPPORT_GATE_FORKS_20261027_v1',str(completed_c),str(completed_q),str(object_root)]):processes.append(dict(pid=int(p.name),command=cmd))
 except OSError:pass
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
baseline_bytes=sum(x['bytes'] for x in existing_sizes['GOAL']);largest_file=max(x['bytes'] for x in existing_sizes['GOAL'])
new_data_bound=max(12*1024**3,2*baseline_bytes+2*largest_file+1024**3)
capacity={}
for mount,bound,reserve,rootfree in [('/root',128*1024**2,1024**3,False),('/data',new_data_bound,2*1024**3,True)]:
 v=os.statvfs(mount);free=(v.f_bfree if rootfree else v.f_bavail)*v.f_frsize
 capacity[mount]=dict(free=free,new_peak_bound=bound,reserve=reserve,headroom=free-bound-reserve)
print(json.dumps(dict(status='PASS_PREDEPLOY_READONLY' if not gpu and not processes and all(x['headroom']>=0 for x in capacity.values()) else 'REVIEW_RESOURCE_REQUIRED',observed=time.time(),bindings=bindings,previous=previous,verified_unique_files=len(checked),scheduler_sources=scheduler_sources,processes=processes,gpu=gpu,capacity=capacity,hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True),optimizer_calls=0,environments=0,remote_writes=0,diagnostics=diagnostics,existing_sizes=existing_sizes,capacity_basis=dict(baseline_private_bytes=baseline_bytes,largest_private_file=largest_file,bound_formula='max(12GiB,2*baseline_total+2*largest_file+1GiB)'))))
