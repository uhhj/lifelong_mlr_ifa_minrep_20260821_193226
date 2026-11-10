"""Frozen new-root bookkeeping; never changes science or old experiments."""
from pathlib import Path
import hashlib,json,os,time
Q=Path(__file__).resolve().parent
NAME='REVERSE_MAIN_MLR_DEBR_OBJECT_GOAL_20261110_v1'
P=Path('/data/mlr_ifa/experiments')/NAME
PY='/data/mlr_ifa/env/bin/python'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp.%d'%os.getpid())
    t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def verify_case(case_root):
    r=Path(case_root);c=read(r/'CONTRACT.json')
    for p,h in c['files'].items():assert sha(p)==h,p
    return sha(r/'CONTRACT.json')
def verify(full=False):
    c=read(Q/'CONTRACT.json');assert c['name']==NAME
    for p,h in c['files'].items():assert sha(Q/p)==h,p
    for name,row in c['cases'].items():
        assert sha(Q/row['contract_file'])==row['contract_sha256']
        verify_case(Q/'cases'/name)
        if full:
            registry=read(Q/'cases'/name/'EXTERNAL_FILES.json')
            for p,h in registry.items():assert sha(p)==h,p
    return c
def capacity():
    seen=set();allocated=0
    for base in [Q,P]:
        for p in base.rglob('*') if base.exists() else []:
            if not p.is_file() or p.is_symlink():continue
            s=p.stat();key=(s.st_dev,s.st_ino)
            if key not in seen:seen.add(key);allocated+=s.st_blocks*512
    v=os.statvfs('/data');free=v.f_bavail*v.f_frsize;remaining=max(0,14*1024**3-allocated)
    assert free-remaining>=2*1024**3,('data free/remaining peak',free,remaining)
    v=os.statvfs('/');root=v.f_bavail*v.f_frsize
    assert root>=1024**3+64*1024**2,('ordinary root free',root)
    return dict(data_available=free,unique_allocated=allocated,remaining_peak=remaining,root_available=root)
def processes():
    out=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:
            cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            cwd=os.readlink(str(p/'cwd'))
            if str(Q) in cmd or str(P) in cmd or cwd.startswith(str(Q)) or cwd.startswith(str(P)):
                out.append(dict(pid=int(p.name),cmd=cmd,cwd=cwd,state=(p/'stat').read_text().split(') ',1)[1].split()[0]))
        except (OSError,ProcessLookupError,PermissionError):pass
    return out
def event(kind,**kwargs):
    with (Q/'EVENTS.jsonl').open('a') as f:f.write(json.dumps(dict(kind=kind,time=time.time(),**kwargs))+'\n');f.flush()
def environment():
    e=os.environ.copy();e.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',LIBERO_CONFIG_PATH='/data/mlr_ifa/config',HF_HOME='/data/mlr_ifa/cache/huggingface')
    return e
