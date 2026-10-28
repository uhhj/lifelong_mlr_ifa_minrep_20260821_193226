"""One finite dispatch after text/import/asset review. Never a resume path."""
import os,json,hashlib,subprocess,time,fcntl
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/SUPPORT_GATE_FORKS_20261027_v1');P=Path('/data/mlr_ifa/experiments/SUPPORT_GATE_FORKS_20261027_v1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not Q.exists() and not P.exists(),'Existing root: no redeploy'
pre=json.loads(BUNDLE['PREDEPLOY.json']);assert time.time()-pre['observed']<7200
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
cases=json.loads(BUNDLE['CASES.json'])
for name,c in cases.items():
    b=json.loads(BUNDLE[name+'/BINDING.json']);assert all(sha(p)==h for p,h in b['files'].items())
    assert all(sha(Path(b['source'])/n)==h for n,h in b['parent_source_sha256'].items())
    assert (Path(c['controller_root'])/'controller.exit').read_text().strip()=='0'
for mount,bound,reserve,root in [('/root',0,1024**3,False),('/data',8*1024**3,2*1024**3,True)]:
    v=os.statvfs(mount);assert (v.f_bfree if root else v.f_bavail)*v.f_frsize-bound>=reserve
for n,s in BUNDLE.items():
    assert not Path(n).is_absolute() and '..' not in Path(n).parts and Path(n).suffix in ['.py','.md','.json']
    if n.endswith('.py'):compile(s,n,'exec')
lock=open('/data/mlr_ifa/SUPPORT_GATE_FORKS_20261027.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
Q.mkdir();files={}
for n,s in BUNDLE.items():
    p=Q/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s);files[str(p)]=sha(p)
children={}
for name in cases:
    child=Q/name;cf={p:h for p,h in files.items() if p.startswith(str(child)+'/')}
    (child/'CONTRACT.json').write_text(json.dumps(dict(files=cf,created=time.time()),indent=2));children[name]=sha(child/'CONTRACT.json');files[str(child/'CONTRACT.json')]=children[name]
    check="import runtime,updates,legacy_updates,support_memory,fork_tools,train,evaluate; import torch; b=runtime.BINDING; cp=torch.load(b['checkpoint'],map_location='cpu'); assert len(cp['state_dict'])==100; assert all(torch.isfinite(v).all() for v in cp['state_dict'].values()); print('PASS_IMPORT_CPU_CP_SCHEMA_NO_FORWARD')"
    p=subprocess.run(['/data/mlr_ifa/env/bin/python','-B','-c',check],cwd=child,capture_output=True,text=True)
    (child/'IMPORT_REVIEW.json').write_text(json.dumps(dict(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)));assert p.returncode==0,(name,p.stderr)
(Q/'CONTRACT.json').write_text(json.dumps(dict(files=files,children=children,created=time.time(),budget=dict(epochs=1700,accepted_updates=403000,physical_training_optimizer_calls=494700,formal_episodes=1120,collection_episodes=0)),indent=2))
fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
with (Q/'controller.log').open('x') as log:p=subprocess.Popen(['/data/mlr_ifa/env/bin/python','-B',str(Q/'controller.py')],cwd=Q,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(status='DISPATCHED_PENDING_OWN_GPU_ADMISSION',root=str(Q),controller=p.pid,contract=sha(Q/'CONTRACT.json'),children=children,frozen_files=len(files),observed=time.time())))
