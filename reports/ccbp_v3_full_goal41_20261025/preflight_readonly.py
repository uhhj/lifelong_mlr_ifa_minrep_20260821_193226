"""Read-only parent closure, source/input and volume inventory; no GPU/optimizer/env."""
import os, json, hashlib, subprocess, time, stat
from pathlib import Path
P=Path('/data/mlr_ifa/experiment_workspaces/CCBP_V2_FULL_GOAL41_20261024_v1_engineering02')
Q=Path('/data/mlr_ifa/experiment_workspaces/CCBP_V3_FULL_GOAL41_20261025_v1')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
assert sha(P/'CONTRACT.json')=='56ec3ca98bde2767e3b0a27170b2e8a71a633fbee2ba028b8a0630e7ff47ff71'
c=json.loads((P/'CONTRACT.json').read_text())
assert all(sha(p)==h for p,h in c['files'].items())
assert all(sha(Path(c['source_root'])/n)==h for n,h in c['parent_source'].items())
assert (P/'controller.exit').read_text()=='0' and (P/'evaluate.exit').read_text()=='0'
assert all((P/('train_'+a+'.exit')).read_text()=='0' for a in 'BCNV')
assert not Q.exists(), 'New root already exists; inspect, never redeploy'
procs=[]
for p in Path('/proc').glob('[0-9]*'):
    try:
        cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        if int(p.name)!=os.getpid() and any(x in cmd for x in [str(P),str(Q)]):procs.append(dict(pid=int(p.name),command=cmd))
    except OSError:pass
assert not procs
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
volumes={}
for mount,bound,reserve,rootfree in [('/root',0,1024**3,False),('/data',int(2.50*1024**3),2*1024**3,True)]:
    s=os.statvfs(mount);free=(s.f_bfree if rootfree else s.f_bavail)*s.f_frsize
    volumes[mount]=dict(free=free,new_peak_bound=bound,reserve=reserve,headroom=free-bound-reserve)
# Metadata only. Being listed is NOT authorization to delete a final or active asset.
inventory=[]
for base in [Path('/root/mlr_ifa'),Path('/data/mlr_ifa/experiment_workspaces'),Path('/data/mlr_ifa/experiments')]:
    for p in base.rglob('*'):
        if not ('recovery' in p.name.lower() or p.name.endswith('.tmp')):continue
        try:
            s=p.lstat()
            if stat.S_ISREG(s.st_mode) and s.st_size>1024**2:
                inventory.append(dict(path=str(p),bytes=s.st_size,allocated=s.st_blocks*512,device=s.st_dev,inode=s.st_ino,nlink=s.st_nlink,mtime=s.st_mtime,classification='UNCLASSIFIED_DO_NOT_DELETE'))
        except FileNotFoundError:pass
stage={}
for a in 'BCNV':
    for t in range(6,10):
        x=json.loads((P/('STAGE_%s_%d.json'%(a,t))).read_text());assert sha(x['model'])==x['sha256'];stage[a+str(t)]=x
hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip()
print(json.dumps(dict(status='PASS_PREDEPLOY_READONLY' if not gpu and all(v['headroom']>=0 for v in volumes.values()) else 'REVIEW_RESOURCE_REQUIRED_NOT_DEPLOYED',observed=time.time(),parent_contract=sha(P/'CONTRACT.json'),parent_stage_checkpoints=stage,parent_frozen_files=len(c['files']),gpu=gpu,hardware=hardware,processes=procs,volumes=volumes,unclassified_intermediate_name_inventory=inventory,model_forwards=0,optimizer_updates=0,environments=0,server_writes=0)))
