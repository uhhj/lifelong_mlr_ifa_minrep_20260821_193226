"""One fresh finite C extension dispatch; no restart or resume entry."""
import os,json,hashlib,subprocess,time,fcntl
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/CCBP_C_GOAL42_43_OBJECT41_42_43_20261027_v1')
O=Path('/data/mlr_ifa/experiments/CCBP_C_GOAL42_43_OBJECT41_42_43_20261027_v1')
P=Path('/data/mlr_ifa/experiment_workspaces/CCBP_V3_FULL_GOAL41_20261025_v1_engineering02')
CASES=['GOAL_42','GOAL_43','OBJECT_41','OBJECT_42','OBJECT_43']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not Q.exists() and not O.exists(),'No redeploy or output reuse'
pre=json.loads(BUNDLE['PREDEPLOY.json']);assert time.time()-pre['observed']<3600
assert sha(P/'CONTRACT.json')==pre['c_contract'] and (P/'controller.exit').read_text()=='0'
assert all(sha(p)==h for p,h in json.loads((P/'CONTRACT.json').read_text())['files'].items())
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
for p in Path('/proc').glob('[0-9]*'):
 try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
 except OSError:continue
 assert int(p.name)==os.getpid() or not any(str(r) in cmd for r in [Q,P])
for mount,bound,reserve,root in [('/root',0,1024**3,False),('/data',6*1024**3,2*1024**3,True)]:
 v=os.statvfs(mount);assert (v.f_bfree if root else v.f_bavail)*v.f_frsize-bound>=reserve
for case in CASES:
 p=json.loads(BUNDLE[case+'/PARENT_BINDING.json']);assert all(sha(n)==h for n,h in p['files'].items())
 assert all(sha(Path(p['source'])/n)==h for n,h in p['source_sha256'].items())
for n,s in BUNDLE.items():
 assert not Path(n).is_absolute() and '..' not in Path(n).parts and Path(n).suffix in ['.py','.md','.json']
 if n.endswith('.py'):compile(s,n,'exec')
lock=open('/data/mlr_ifa/CCBP_C_GOAL42_43_OBJECT41_42_43_20261027.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
Q.mkdir();files={}
for n,s in BUNDLE.items():
 target=Q/n;target.parent.mkdir(exist_ok=True);target.write_text(s);files[str(target)]=sha(target)
children={}
for case in CASES:
 child=Q/case;parent=json.loads((child/'PARENT_BINDING.json').read_text())
 cf={p:h for p,h in files.items() if p.startswith(str(child)+'/')}
 contract=dict(files=cf,parent_source=parent['source_sha256'],source_root=parent['source'],official_initialization=parent['official_initialization'],seed=parent['seed'],suite=parent['suite'],method='C',budget=dict(epochs=400,updates=76100 if parent['suite']=='GOAL' else 105500,collection_episodes=0,formal_episodes=200),created=time.time())
 (child/'CONTRACT.json').write_text(json.dumps(contract,indent=2));children[case]=sha(child/'CONTRACT.json');files[str(child/'CONTRACT.json')]=children[case]
 proc=subprocess.run(['/data/mlr_ifa/env/bin/python','-B','-c','import runtime,updates,train,evaluate; import synthetic_checks; print(synthetic_checks.run())'],cwd=child,capture_output=True,text=True)
 (child/'IMPORT_REVIEW.json').write_text(json.dumps(dict(exit_code=proc.returncode,stdout=proc.stdout,stderr=proc.stderr)))
 assert proc.returncode==0,(case,proc.stdout,proc.stderr)
(Q/'CONTRACT.json').write_text(json.dumps(dict(files=files,children=children,parent_C_contract=pre['c_contract'],created=time.time(),budget=dict(epochs=2000,updates=468700,training_episodes=0,formal_episodes=1000)),indent=2))
fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
with (Q/'controller.log').open('x') as f:
 proc=subprocess.Popen(['/data/mlr_ifa/env/bin/python','-B',str(Q/'controller.py')],cwd=Q,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(status='DISPATCHED_PENDING_OWN_GPU_ADMISSION',root=str(Q),controller=proc.pid,contract=sha(Q/'CONTRACT.json'),children=children,frozen_files=len(files),hardware=pre['hardware'],observed=time.time())))
