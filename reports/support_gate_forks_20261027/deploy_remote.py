"""Deploy only reviewed recovery helpers into a new engineering directory; no bootstrap."""
import os,json,hashlib,subprocess,time,fcntl
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/SUPPORT_GATE_FORKS_20261027_v1')
P=Path('/data/mlr_ifa/experiments/SUPPORT_GATE_FORKS_20261027_v1')
E=Q/'engineering_resume_20261028_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text())
assert not E.exists(),'No repeated deployment'
assert sha(Q/'CONTRACT.json')=='9b53df5a8fbe894e7b27887462b7b25b4e8477a9ad12820deb8df162632fb978'
c=js(Q/'CONTRACT.json');assert all(sha(p)==h for p,h in c['files'].items())
assert sha(P/'OBJECT_41/S/recovery.pth')=='df2c9101552a50f0478b36b1df942166c614b598a6d11af0d595fb4c9da7bb97'
assert len(js(Q/'OBJECT_41/EPOCHS_S.json'))==7
assert not (Q/'controller.exit').exists() and not (Q/'OBJECT_41/train_S.exit').exists()
for p in Path('/proc').glob('[0-9]*'):
    try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    except OSError:continue
    assert int(p.name)==os.getpid() or str(Q) not in cmd,(p.name,cmd)
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip();assert gpu
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
allocated=0;seen=set()
for p in P.rglob('*'):
    if p.is_file() and not p.is_symlink():
        st=p.stat();key=(st.st_dev,st.st_ino)
        if key not in seen:seen.add(key);allocated+=st.st_blocks*512
capacity={}
for mount,bound,reserve,root in [('/root',0,1024**3,False),('/data',max(0,8*1024**3-allocated)+(P/'OBJECT_41/S/recovery.pth').stat().st_size,2*1024**3,True)]:
    v=os.statvfs(mount);free=(v.f_bfree if root else v.f_bavail)*v.f_frsize;assert free-bound>=reserve
    capacity[mount]=dict(free=free,remaining_bound_including_extra_archive=bound,reserve=reserve,spare=free-bound-reserve)
assert set(BUNDLE)=={'resume_train_S.py','controller_resume.py','BUILD_AUDIT.json','resume_train_S.diff'}
for n,s in BUNDLE.items():
    if n.endswith('.py'):compile(s,n,'exec')
lock=open('/data/mlr_ifa/SUPPORT_GATE_FORKS_20261027.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
E.mkdir();files={}
for n,s in BUNDLE.items():
    p=E/n;p.write_text(s);files[n]=sha(p)
for n in ['OBJECT_41/train_S.log','OBJECT_41/EPOCHS_S.json','OBJECT_41/GRADIENTS_S.json','controller.log','STATUS.json','STARTED.json']:
    p=Q/n;(E/('original_'+n.replace('/','_'))).write_bytes(p.read_bytes())
(E/'CONTRACT.json').write_text(json.dumps(dict(files=files,parent_contract=sha(Q/'CONTRACT.json'),checkpoint_sha256=sha(P/'OBJECT_41/S/recovery.pth'),capacity=capacity,gpu=gpu,isolated_extra_planned=31,autograd_extra_planned=11,unlogged_interrupted_cost='UNKNOWN',created=time.time()),indent=2))
assert all(sha(p)==h for p,h in c['files'].items())
fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
with (E/'controller.log').open('x') as log:
    p=subprocess.Popen(['/data/mlr_ifa/env/bin/python','-B',str(E/'controller_resume.py')],cwd=Q,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(status='RESUME_DISPATCHED_GPU_ADMISSION_PENDING',root=str(Q),engineering=str(E),controller=p.pid,engineering_contract_sha256=sha(E/'CONTRACT.json'),parent_contract_sha256=sha(Q/'CONTRACT.json'),gpu=gpu,capacity=capacity,observed=time.time(),scientific_parameters_changed=False)))
