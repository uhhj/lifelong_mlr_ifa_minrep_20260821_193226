"""Remaining additional allocation, credit only stable private already allocated files."""
import os,json,time
from pathlib import Path
A=Path('/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022')
N=Path('/data/mlr_ifa/experiment_workspaces/TACR_TACST_OBJECT_conditional_20261023')
bound=int(1.6*1024**3);rows=[];need={'/':0,'/data':0}
for Q in (A,N):
 for e in json.loads((Q/'QUEUE.json').read_text()):
  O=Path(e['output']);accepted=(Q/(e['key']+'.receipt.json')).is_file();vol=e.get('volume','/data');allocated=0;seen=set();unstable=0
  if O.is_dir():
   candidates=[]
   for base,dirs,files in os.walk(O,followlinks=False):
    dirs[:]=[n for n in dirs if not (Path(base)/n).is_symlink()]
    for name in files:
     p=Path(base)/name
     if p.is_symlink() or name.endswith('.tmp'):continue
     try:
      st=p.stat();key=(st.st_dev,st.st_ino)
      if st.st_nlink!=1 or key in seen:continue
      seen.add(key);candidates.append((p,st))
     except OSError:pass
   for p,st in candidates:
    try:
     after=p.stat()
     if (st.st_ino,st.st_size,st.st_blocks,st.st_mtime_ns)==(after.st_ino,after.st_size,after.st_blocks,after.st_mtime_ns):allocated+=st.st_blocks*512
     else:unstable+=1
    except OSError:unstable+=1
  remaining=0 if accepted else max(0,bound-allocated);need[vol]+=remaining
  rows.append(dict(key=e['key'],volume=vol,server_accepted=accepted,exists=O.exists(),stable_private_allocated_bytes=allocated,unstable_files_not_credited=unstable,planning_bound_bytes=bound,additional_bound_bytes=remaining))
disk={}
for vol in ('/','/data'):
 st=os.statvfs(vol);free=(st.f_bfree if vol=='/data' else st.f_bavail)*st.f_frsize;reserve=(2 if vol=='/data' else 1)*1024**3
 disk[vol]=dict(free=free,reserve=reserve,remaining_additional_bound=need[vol],margin=free-reserve-need[vol])
 assert free>=reserve+need[vol],('RESOURCE_CAPACITY',vol,disk[vol])
print(json.dumps(dict(status='REMAINING_ALLOCATION_BOUND_PASS_NOT_FINAL_ACCEPTANCE',observed_unix=time.time(),entries=rows,volumes=disk,model_forwards=0,optimizer_updates=0,environment_steps=0,server_writes=0)))
