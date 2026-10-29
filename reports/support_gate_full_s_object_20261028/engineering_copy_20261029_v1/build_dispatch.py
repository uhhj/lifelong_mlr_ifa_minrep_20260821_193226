"""Build versioned source-only diagnostic dispatch; no scientific source edits."""
from pathlib import Path
import json,hashlib,ast
E=Path(__file__).resolve().parent;N=E.parent
files={'candidate.py':(N/'performance_copy_candidate.py').read_text(encoding='utf8'),'validate_training.py':(E/'validate_training.py').read_text(encoding='utf8')}
for s in files.values():ast.parse(s,feature_version=(3,8))
script='''import os,json,time,hashlib,subprocess,sys
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/SUPPORT_GATE_FULL_S_OBJECT41_42_43_20261028_v1');E=Q/'engineering_copy_20261029_v1'
files=FILES
g=json.loads((E/'GUARD.json').read_text());st=Path('/proc/3013/stat').read_text().split(') ',1)[1].split()
assert st[0]=='T' and st[19]==g['controller']['start']
assert not (Q/'seed43/train_S.log').exists()
assert time.time()<g['deadline']-1200
for name,source in files.items():
 p=E/name;assert not p.exists();p.write_text(source)
receipt=dict(time=time.time(),files={name:hashlib.sha256((E/name).read_bytes()).hexdigest() for name in files})
with (E/'validation.log').open('x') as f:
 p=subprocess.Popen([sys.executable,'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=1',str(E/'validate_training.py')],cwd=E,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1'))
receipt['pid']=p.pid;(E/'VALIDATION_DISPATCH.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''.replace('FILES',repr(files))
(E/'dispatch_validation.py').write_text(script,encoding='utf8')
print(json.dumps(dict(status='STATIC_ONLY',files={n:hashlib.sha256(s.encode()).hexdigest() for n,s in files.items()})))
