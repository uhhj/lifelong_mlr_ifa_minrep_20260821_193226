import json,os,hashlib,time,subprocess
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/CCBP_V4_S_GOAL42_43_20261026_v1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
c=json.loads((Q/'CONTRACT.json').read_text());assert all(sha(p)==h for p,h in c['files'].items())
for seed in (42,43):
 q=Q/('seed%d'%seed);cc=json.loads((q/'CONTRACT.json').read_text());assert sha(q/'CONTRACT.json')==c['children'][str(seed)]
 assert all(sha(Path(cc['source_root'])/n)==h for n,h in cc['parent_source'].items())
procs=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if int(p.name)!=os.getpid() and str(Q) in cmd:procs.append(dict(pid=int(p.name),cmd=cmd,state=(p/'stat').read_text().split(')')[-1].split()[0]))
 except OSError:pass
light={}
for p in Q.rglob('*'):
 if p.is_file() and p.suffix in ['.json','.jsonl','.exit'] and p.stat().st_size<8*1024**2 and p.name not in ['CONTRACT.json','PARENT_BINDING.json','PREDEPLOY.json']:
  raw=p.read_bytes();light[str(p.relative_to(Q))]=dict(text=raw.decode(),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
free={}
for n in ['/root','/data']:
 v=os.statvfs(n);free[n]=dict(bavail=v.f_bavail*v.f_frsize,bfree=v.f_bfree*v.f_frsize)
print(json.dumps(dict(status='READ_ONLY_SNAPSHOT_NOT_FINAL',observed=time.time(),contract_sha256=sha(Q/'CONTRACT.json'),children=c['children'],source_files=len(c['files']),processes=procs,gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True),capacity=free,light=light,log_tails={str(p.relative_to(Q)):p.read_text()[-6000:] for p in Q.rglob('*.log')})))
