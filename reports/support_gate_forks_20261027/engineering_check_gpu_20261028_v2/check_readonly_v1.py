"""Read-only lightweight progress snapshot; never touches checkpoint payloads."""
import json,hashlib,os,subprocess,time
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/SUPPORT_GATE_FORKS_20261027_v1');P=Path('/data/mlr_ifa/experiments/SUPPORT_GATE_FORKS_20261027_v1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text())
contract=js(Q/'CONTRACT.json');hashes={p:sha(p) for p in contract['files']};assert hashes==contract['files']
light={};tails={};procs=[]
for p in Q.rglob('*'):
    if not p.is_file():continue
    n=p.relative_to(Q).as_posix()
    if p.suffix in ['.json','.jsonl','.exit'] and p.name not in ['BINDING.json','CASES.json','PREDEPLOY.json']:
        assert p.stat().st_size<12*1024**2
        raw=p.read_bytes();light[n]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),text=raw.decode())
    elif p.suffix=='.log':tails[n]=p.read_text(errors='replace')[-3000:]
for p in Path('/proc').glob('[0-9]*'):
    try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    except OSError:continue
    if str(Q) in cmd and int(p.name)!=os.getpid():procs.append(dict(pid=int(p.name),command=cmd))
capacity={};seen=set();allocated=0
for p in P.rglob('*') if P.exists() else []:
    if p.is_file() and not p.is_symlink():
        st=p.stat();key=(st.st_dev,st.st_ino)
        if key not in seen:seen.add(key);allocated+=st.st_blocks*512
for mount,bound,reserve,root in [('/root',0,1024**3,False),('/data',max(0,8*1024**3-allocated),2*1024**3,True)]:
    st=os.statvfs(mount);free=(st.f_bfree if root else st.f_bavail)*st.f_frsize;capacity[mount]=dict(free=free,remaining_bound=bound,reserve=reserve,spare=free-bound-reserve)
print(json.dumps(dict(status='READONLY_SNAPSHOT_NOT_FINAL',observed=time.time(),contract=sha(Q/'CONTRACT.json'),frozen=hashes,processes=procs,capacity=capacity,light=light,log_tails=tails,gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True))))
