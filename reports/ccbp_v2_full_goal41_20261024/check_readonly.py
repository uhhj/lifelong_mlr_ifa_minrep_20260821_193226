import json,os,hashlib,time,subprocess
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/CCBP_V2_FULL_GOAL41_20261024_v1_engineering02')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
contract=json.loads((Q/'CONTRACT.json').read_text());assert all(sha(p)==h for p,h in contract['files'].items())
assert all(sha(Path(contract['source_root'])/n)==h for n,h in contract['parent_source'].items())
procs=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if int(p.name)!=os.getpid() and str(Q) in cmd:procs.append(dict(pid=int(p.name),cmd=cmd))
 except OSError:pass
light={}
for p in Q.rglob('*'):
 if p.is_file() and p.suffix in ['.json','.jsonl','.exit'] and p.stat().st_size<8*1024**2 and p.name!='CONTRACT.json':light[str(p.relative_to(Q))]=dict(text=p.read_text(),sha256=sha(p),bytes=p.stat().st_size)
logs={p.name:p.read_text()[-6000:] for p in Q.glob('*.log')}
free={}
for n in ['/root','/data']:
 v=os.statvfs(n);free[n]=dict(bavail=v.f_bavail*v.f_frsize,bfree=v.f_bfree*v.f_frsize)
print(json.dumps(dict(status='READ_ONLY_SNAPSHOT_NOT_FINAL',observed=time.time(),source_files=len(contract['files']),contract_sha256=sha(Q/'CONTRACT.json'),processes=procs,gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True),capacity=free,light=light,log_tails=logs)))
