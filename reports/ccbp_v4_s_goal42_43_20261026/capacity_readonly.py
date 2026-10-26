import os,json,time
from pathlib import Path
base=Path('/data/mlr_ifa/experiments/CCBP_V4_S_GOAL42_43_20261026_v1');seen=set();allocated=0
for p in base.rglob('*') if base.exists() else []:
 if not p.is_file() or p.is_symlink():continue
 st=p.stat();key=(st.st_dev,st.st_ino)
 if key not in seen:seen.add(key);allocated+=st.st_blocks*512
rows={}
for mount,bound,reserve,credit in [('/root',0,1024**3,0),('/data',4*1024**3,2*1024**3,allocated)]:
 v=os.statvfs(mount);free=(v.f_bfree if mount=='/data' else v.f_bavail)*v.f_frsize;remaining=max(0,bound-credit)
 rows[mount]=dict(free=free,allocated_unique=credit,remaining=remaining,reserve=reserve,headroom=free-remaining-reserve)
print(json.dumps(dict(status='PASS_NOT_FINAL' if all(x['headroom']>=0 for x in rows.values()) else 'REVIEW_CAPACITY',observed=time.time(),capacity=rows,payload_reads=0,server_writes=0,forwards=0,optimizer_updates=0,environments=0)))
