import os,json,hashlib,subprocess,time
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/CCBP_GOAL41_20261024_v1');P=Path('/root/mlr_ifa/experiments/CCBP_GOAL41_20261024_v1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(Q/'CONTRACT.json')=='4e921bf269b364c20758fb947305d9aceac484048ba7d8e3e0d9ecb26957ae24'
c=json.loads((Q/'CONTRACT.json').read_text());assert all(sha(p)==h for p,h in c['files'].items());assert all(sha(Path(c['source_root'])/n)==h for n,h in c['parent_source'].items())
assert (Q/'controller.exit').read_text()=='0'
processes=[]
for d in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(d/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if str(Q) in cmd:processes.append(dict(pid=int(d.name),cmd=cmd))
 except OSError:pass
assert not processes
cap={}
for root in ['/root','/data']:
 s=os.statvfs(root);cap[root]=dict(bavail=s.f_bavail*s.f_frsize,bfree=s.f_bfree*s.f_frsize)
sizes={str(p.relative_to(P)):dict(bytes=p.stat().st_size,allocated=p.stat().st_blocks*512) for p in P.rglob('*') if p.is_file() and not p.is_symlink()}
print(json.dumps(dict(status='PASS_READONLY_PREFLIGHT',observed=time.time(),gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True),capacity=cap,prior_private_sizes=sizes,prior_contract=sha(Q/'CONTRACT.json'),prior_exit=0,prior_residual=processes,forwards=0,updates=0,environments=0)))
