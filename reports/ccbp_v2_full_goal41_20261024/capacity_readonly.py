"""External stat-only storage check; no payload reads or remote writes."""
import os,json,time,stat
from pathlib import Path
def inventory(base):
 out={}
 for p in base.rglob('*'):
  try:
   s=p.lstat()
   if stat.S_ISREG(s.st_mode) and not p.name.endswith('.tmp'):
    out[str(p)]=(s.st_dev,s.st_ino,s.st_size,s.st_blocks,s.st_mtime_ns,s.st_nlink)
  except FileNotFoundError:pass
 return out
result={}
for mount,bound,reserve,rootfree in [('/root',int(1.07*1024**3),1024**3,False),('/data',int(1.88*1024**3),2*1024**3,True)]:
 base=Path(mount)/'mlr_ifa/experiments/CCBP_V2_FULL_GOAL41_20261024_v1'
 first=inventory(base);second=inventory(base);seen=set();allocated=0
 for p,s in first.items():
  if second.get(p)!=s:continue
  key=s[:2]
  if key in seen:continue
  # Credit hardlinks only when all names are accounted for within this experiment.
  if sum(v[:2]==key for v in second.values())!=s[5]:continue
  seen.add(key);allocated+=s[3]*512
 v=os.statvfs(mount);free=(v.f_bfree if rootfree else v.f_bavail)*v.f_frsize
 remaining=max(0,bound-allocated)
 result[mount]=dict(free=free,stable_unique_allocated=allocated,remaining_bound=remaining,reserve=reserve,headroom=free-remaining-reserve)
print(json.dumps(dict(status='PASS_NOT_FINAL' if all(v['headroom']>=0 for v in result.values()) else 'REVIEW_CAPACITY',observed=time.time(),volumes=result,payload_reads=0,forwards=0,updates=0,environments=0,server_writes=0)))
