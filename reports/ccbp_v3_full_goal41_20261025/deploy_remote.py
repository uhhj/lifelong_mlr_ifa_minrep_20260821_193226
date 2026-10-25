"""ONE new version dispatch; requires actual fresh preflight, never a restart."""
import os, subprocess, json, time, hashlib, fcntl
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/CCBP_V3_FULL_GOAL41_20261025_v1')
PR=Path('/root/mlr_ifa/experiments/CCBP_V3_FULL_GOAL41_20261025_v1')
PD=Path('/data/mlr_ifa/experiments/CCBP_V3_FULL_GOAL41_20261025_v1')
P=Path('/data/mlr_ifa/experiment_workspaces/CCBP_V2_FULL_GOAL41_20261024_v1_engineering02')
O=Path('/data/mlr_ifa/experiments/ALIGN_GOAL_seed41_20261022')
S=Path('/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/GOAL_41/source')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
assert not any(p.exists() for p in (Q,PR,PD)), 'No redeployment'
pre=json.loads(BUNDLE['PREDEPLOY_ACCEPTANCE.json'])
assert pre['status']=='PASS_LOCAL_PREDEPLOY_ACCEPTANCE' and time.time()-pre['remote_observed']<1800
assert sha(P/'CONTRACT.json')==pre['parent_contract']=='56ec3ca98bde2767e3b0a27170b2e8a71a633fbee2ba028b8a0630e7ff47ff71'
pc=json.loads((P/'CONTRACT.json').read_text());assert all(sha(p)==h for p,h in pc['files'].items())
assert all(sha(S/n)==h for n,h in pc['parent_source'].items())
assert (P/'controller.exit').read_text()=='0' and (P/'evaluate.exit').read_text()=='0'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
for p in Path('/proc').glob('[0-9]*'):
    try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    except OSError:continue
    assert int(p.name)==os.getpid() or not any(str(q) in cmd for q in (P,Q))
cap={}
for mount,bound,reserve,rootfree in [('/root',int(1.10*1024**3),1024**3,False),('/data',int(1.40*1024**3),2*1024**3,True)]:
    v=os.statvfs(mount);free=(v.f_bfree if rootfree else v.f_bavail)*v.f_frsize
    assert free-bound>=reserve,(mount,free,bound,reserve)
    cap[mount]=dict(free=free,new_peak_bound=bound,reserve=reserve,headroom=free-bound-reserve)
lock=open('/data/mlr_ifa/CCBP_V3_FULL_GOAL41_20261025.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for n,s in BUNDLE.items():
    assert Path(n).name==n and Path(n).suffix in ['.py','.md','.txt','.json']
    if n.endswith('.py'):compile(s,n,'exec')
Q.mkdir();PR.mkdir();PD.mkdir();files={}
for n,s in BUNDLE.items():(Q/n).write_text(s);files[str(Q/n)]=sha(Q/n)
# Bind unchanged parent metadata/input files without copying any payload.
for p,h in pc['files'].items():
    if p.startswith(str(O)+'/'):files[p]=h
for n in ['CONTRACT.json','controller.exit','evaluate.exit']:
    files[str(P/n)]=sha(P/n)
accept=json.loads(BUNDLE['PARENT_LOCAL_ACCEPTANCE.json'])
for a in 'BCNV':
    for t in range(6,10):
        m=json.loads((P/('STAGE_%s_%d.json'%(a,t))).read_text())
        assert sha(m['model'])==m['sha256']==accept['strict'][a][str(t)]['sha256']
        files[str(P/('STAGE_%s_%d.json'%(a,t)))]=sha(P/('STAGE_%s_%d.json'%(a,t)))
official=Path(json.loads((O/'resolved_ra_coreset.json').read_text())['pretrain'])
assert sha(official)==pc['official_initialization']['sha256']
(Q/'CONTRACT.json').write_text(json.dumps(dict(files=files,source_root=str(S),parent_source=pc['parent_source'],official_initialization=dict(path=str(official),sha256=sha(official)),parent_v2_contract=sha(P/'CONTRACT.json'),created=time.time(),budget=dict(arms=list('CRG'),seed=41,epochs=1200,steps=228300,episodes=600),capacity=cap),indent=2))
fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
with (Q/'controller.log').open('x') as f:
    p=subprocess.Popen(['/data/mlr_ifa/env/bin/python','-B',str(Q/'controller.py')],cwd=Q,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(status='DISPATCHED_NOT_YET_ADMITTED',root=str(Q),controller=p.pid,contract_sha256=sha(Q/'CONTRACT.json'),frozen_files=len(files),capacity=cap,observed=time.time())))
