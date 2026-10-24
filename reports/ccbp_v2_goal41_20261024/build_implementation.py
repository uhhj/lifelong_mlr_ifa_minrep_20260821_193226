"""Local one-time derivation of the preregistered four-arm implementation."""
from pathlib import Path
import json,hashlib,ast
R=Path(__file__).resolve().parent;OLD=R.parent/'20261024_ccbp'
oldq='/data/mlr_ifa/experiment_workspaces/CCBP_GOAL41_20261024_v1'
q='/data/mlr_ifa/experiment_workspaces/CCBP_V2_GOAL41_20261024_v1'
name='CCBP_V2_GOAL41_20261024_v1'
def put(n,s):ast.parse(s) if n.endswith('.py') else None;(R/n).write_text(s,encoding='utf8')
put('legacy_losses.py',(OLD/'losses.py').read_text())
put('layout.py',f'''from pathlib import Path
ARMS='BCNV'
ROOT_PRIVATE=Path('/root/mlr_ifa/experiments/{name}')
DATA_PRIVATE=Path('/data/mlr_ifa/experiments/{name}')
OUTPUT={{a:(ROOT_PRIVATE if a in 'BN' else DATA_PRIVATE)/a for a in ARMS}}
TEACHER_CACHE=Path('/root/mlr_ifa/experiments/CCBP_GOAL41_20261024_v1/teacher_cache.pth')
PRIOR_ROOT=Path('{oldq}')
PER_VOLUME_BOUND=int(1.04*1024**3)
''')
s=(OLD/'runtime.py').read_text();s=s.replace("P=Path('/root/mlr_ifa/experiments/CCBP_GOAL41_20261024_v1')","from layout import *")
start=s.index('def capacity():');end=s.index('class Dataset',start)
s=s[:start]+'''def capacity():
 out={}
 for mount,base,reserve,root_free in [('/root',ROOT_PRIVATE,1024**3,False),('/data',DATA_PRIVATE,2*1024**3,True)]:
  seen=set();allocated=0
  for p in base.rglob('*') if base.exists() else []:
   if not p.is_file() or p.is_symlink():continue
   st=p.stat();key=(st.st_dev,st.st_ino)
   if key not in seen:seen.add(key);allocated+=st.st_blocks*512
  v=os.statvfs(mount);free=(v.f_bfree if root_free else v.f_bavail)*v.f_frsize
  remaining=max(0,PER_VOLUME_BOUND-allocated);assert free-remaining>=reserve,(mount,free,remaining)
  out[mount]=dict(free=free,unique_inode_allocated=allocated,remaining_bound=remaining,reserve=reserve)
 return out
''' +s[end:]
s=s.replace("str(P/arm/'models')","str(OUTPUT[arm]/'models')").replace("str(P/arm)","str(OUTPUT[arm])")
put('runtime.py',s)
s=(OLD/'evaluate.py').read_text().replace('160','320').replace('16,cells=8','32,cells=16').replace("for arm in 'ABCD'","for arm in ARMS").replace("load('A'","load('B'").replace('P/arm', 'OUTPUT[arm]').replace('for taskid in [8,9]','for taskid in [6,7,8,9]')
put('evaluate.py',s)
s=(OLD/'check_readonly.py').read_text().replace(oldq,q);put('check_readonly.py',s)
s=(OLD/'audit_progress.py').read_text().replace("for arm in 'ABCD'","for arm in 'BCNV'").replace("for arm in 'BCD'","for arm in 'CNV'").replace("EPOCHS_A.json","EPOCHS_B.json");put('audit_progress.py',s)
s=(OLD/'publish.py').read_text().replace('reports/ccbp_goal41_20261024','reports/ccbp_v2_goal41_20261024').replace('CCBP finite four-arm Task8 to Task9 validation and reviewed evidence','CCBP-v2 finite GOAL41 four-arm validation and reviewed evidence');put('publish.py',s)
s=(OLD/'controller.py').read_text().replace('CCBP_GOAL41_20261024.lock','CCBP_V2_GOAL41_20261024.lock').replace("('prepare_A',['prepare','A'])","('prepare_B',['prepare','B'])").replace("for arm in 'ABCD'","for arm in 'BCNV'")
put('controller.py',s)
print('Derived runtime/layout/legacy/evaluation/controller/local wrappers; no deployment.')
