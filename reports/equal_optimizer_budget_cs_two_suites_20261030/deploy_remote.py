"""Unique authorized C-only deployment; no existing run is modified or resumed."""
import os,json,hashlib,subprocess,time,fcntl
from pathlib import Path
NAME='C_MATCH_FULL_S_BUDGET_GOAL_OBJECT_20261030_v1'
Q=Path('/data/mlr_ifa/experiment_workspaces')/NAME
O=Path('/data/mlr_ifa/experiments')/NAME
CASES=[s+'_'+str(k) for s in ['GOAL','OBJECT'] for k in [41,42,43]]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not Q.exists() and not O.exists(),'No redeployment/output reuse'
pre=json.loads(BUNDLE['PREDEPLOY.json']);assert 0<=time.time()-pre['observed']<3600
for suite,ref in pre['previous'].items():
 root=Path(ref['root']);assert sha(root/'CONTRACT.json')==ref['contract_sha256']
 assert all(sha(p)==h for p,h in json.loads((root/'CONTRACT.json').read_text())['files'].items())
 ep=root/('controller.exit' if suite=='GOAL' else 'engineering_copy_20261029_v1/controller.exit')
 assert ep.read_text().strip()=='0' and sha(ep)==ref['exit_sha256']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
for p in Path('/proc').glob('[0-9]*'):
 try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
 except OSError:continue
 assert int(p.name)==os.getpid() or not any(x in cmd for x in [str(Q),'SUPPORT_GATE_FORKS_20261027_v1',*(x['root'] for x in pre['previous'].values())])
for mount,bound,reserve,rootfree in [('/root',0,1024**3,False),('/data',18*1024**3,2*1024**3,True)]:
 v=os.statvfs(mount);assert (v.f_bfree if rootfree else v.f_bavail)*v.f_frsize-bound>=reserve
checked={}
for case in CASES:
 parent=json.loads(BUNDLE[case+'/PARENT_BINDING.json'])
 for path,h in {**parent['files'],**{str(Path(parent['source'])/n):h for n,h in parent['source_sha256'].items()}}.items():
  if path not in checked:checked[path]=sha(path)
  assert checked[path]==h,path
 for x in json.loads(BUNDLE[case+'/S_REFERENCE.json']).values():assert sha(x['path'])==x['sha256']
for n,s in BUNDLE.items():
 assert not Path(n).is_absolute() and '..' not in Path(n).parts and Path(n).suffix in ['.py','.md','.json']
 if n.endswith('.py'):compile(s,n,'exec')
lock=open('/data/mlr_ifa/'+NAME+'.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
Q.mkdir();files={}
for n,s in BUNDLE.items():
 target=Q/n;target.parent.mkdir(exist_ok=True);target.write_text(s);files[str(target)]=sha(target)
children={}
for case in CASES:
 child=Q/case;parent=json.loads((child/'PARENT_BINDING.json').read_text());total=142200 if parent['suite']=='GOAL' else 195300
 cf={p:h for p,h in files.items() if p.startswith(str(child)+'/')}
 contract=dict(files=cf,parent_source=parent['source_sha256'],source_root=parent['source'],official_initialization=parent['official_initialization'],seed=parent['seed'],suite=parent['suite'],method='C_matched_existing_full_S_physical_optimizer_budget',budget=dict(epochs=700,stage_epochs={'6':100,'7':200,'8':200,'9':200},updates=total,physical_optimizer_calls=total,admission_optimizer_calls=91,admission_extra_autograd=12,direction_extra_autograd=48,collection_episodes=0,formal_episodes=200),created=time.time())
 (child/'CONTRACT.json').write_text(json.dumps(contract,indent=2));children[case]=sha(child/'CONTRACT.json');files[str(child/'CONTRACT.json')]=children[case]
 proc=subprocess.run(['/data/mlr_ifa/env/bin/python','-B','-c',"import runtime,updates,train,evaluate; import synthetic_checks; print(synthetic_checks.run()); assert runtime.CALLS=={'production':0,'admission':0}"],cwd=child,capture_output=True,text=True)
 (child/'IMPORT_REVIEW.json').write_text(json.dumps(dict(exit_code=proc.returncode,stdout=proc.stdout,stderr=proc.stderr)))
 assert proc.returncode==0,(case,proc.stdout,proc.stderr)
(Q/'CONTRACT.json').write_text(json.dumps(dict(files=files,children=children,accepted_S_roots=pre['previous'],created=time.time(),budget=dict(epochs=4200,updates=1012500,physical_optimizer_calls=1012500,admission_optimizer_calls=546,training_episodes=0,formal_episodes=1200,new_S_runs=0)),indent=2))
fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
with (Q/'controller.log').open('x') as f:
 proc=subprocess.Popen(['/data/mlr_ifa/env/bin/python','-B',str(Q/'controller.py')],cwd=Q,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(status='DISPATCHED_C_ONLY_PENDING_OWN_GPU_ADMISSION',root=str(Q),controller=proc.pid,contract=sha(Q/'CONTRACT.json'),children=children,frozen_files=len(files),hardware=pre['hardware'],observed=time.time(),new_S_runs=0)))
