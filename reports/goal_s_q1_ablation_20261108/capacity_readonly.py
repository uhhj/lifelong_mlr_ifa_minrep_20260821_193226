import os,json,time
from pathlib import Path
roots=['SUPPORT_GATE_FULL_S_GOAL41_42_43_20261029_v1','S_LAMBDA_R_GOAL41_42_43_20261103_v1']
out={}
for name in roots:
 base=Path('/data/mlr_ifa/experiments')/name
 rows=[]
 for p in base.rglob('*'):
  if p.is_file() and not p.is_symlink():
   s=p.stat();rows.append(dict(path=str(p.relative_to(base)),bytes=s.st_size,allocated=s.st_blocks*512,inode=s.st_ino))
 out[name]=rows
print(json.dumps(dict(status='READ_ONLY_CAPACITY_FILES',observed=time.time(),files=out,optimizer_calls=0,environments=0)))
