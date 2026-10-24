"""Inventory proposed obsolete recovery copies only; never deletes anything."""
import json,os,hashlib,time
from pathlib import Path
roots=[Path('/data/mlr_ifa/experiment_workspaces/CAET_EM_OBJECT41_20261023_v1_engineering05/provenance'),Path('/data/mlr_ifa/experiment_workspaces/CAET_EM_OBJECT41_20261024_recovery06/provenance')]
paths=[]
for root in roots:
 if root.exists():paths.extend(p for p in root.rglob('*') if p.is_file() and p.suffix=='.pth')
paths.append(Path('/root/mlr_ifa/experiments/CCBP_V2_GOAL41_20261024_v1/B/recovery.pth'))
rows=[]
for p in paths:
 assert p.is_file() and not p.is_symlink()
 st=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 rows.append(dict(path=str(p.resolve()),bytes=st.st_size,allocated=st.st_blocks*512,nlink=st.st_nlink,device=st.st_dev,inode=st.st_ino,sha256=h.hexdigest()))
print(json.dumps(dict(status='CANDIDATES_ONLY_NO_DELETION',observed=time.time(),rows=rows,server_writes=0,forwards=0,updates=0,environments=0)))
