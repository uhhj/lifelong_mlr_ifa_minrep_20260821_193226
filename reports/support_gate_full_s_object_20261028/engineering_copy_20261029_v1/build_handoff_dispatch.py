from pathlib import Path
import json,hashlib,ast
E=Path(__file__).resolve().parent
files={n:(E/n).read_text(encoding='utf8') for n in ['controller.py','owner_watchdog.py','train43_optimized.py']}
for source in files.values():ast.parse(source,feature_version=(3,8))
existing={'candidate.py':hashlib.sha256((E.parent/'performance_copy_candidate.py').read_bytes()).hexdigest(),'validate_training.py':hashlib.sha256((E/'validate_training.py').read_bytes()).hexdigest()}
script='''import os,json,time,hashlib,subprocess,sys
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/SUPPORT_GATE_FULL_S_OBJECT41_42_43_20261028_v1');E=Q/'engineering_copy_20261029_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
files=FILES
existing=EXISTING
assert not (E/'CONTRACT.json').exists() and not (E/'STARTED.json').exists()
v=json.loads((E/'VALIDATION.json').read_text());assert v['status']=='PASS_FULL_TRAINING_PATH_EXACT' and v['optimizer_calls_completed']==v['optimizer_calls_started']==3914 and v['environments']==0
for p,h in existing.items():assert sha(E/p)==h
for p,h in json.loads((Q/'CONTRACT.json').read_text())['files'].items():assert sha(p)==h
g=json.loads((E/'GUARD.json').read_text());s=Path('/proc/3013/stat').read_text().split(') ',1)[1].split()
assert s[0]=='T' and s[19]==g['controller']['start'] and time.time()<g['deadline']-60
assert not (Q/'seed43/train_S.log').exists()
for p in Path('/proc').glob('[0-9]*'):
 try:cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
 except OSError:continue
 assert str(E/'validate_training.py') not in cmd,'Validation process must exit first'
for name,source in files.items():
 p=E/name;assert not p.exists();p.write_text(source)
contract=dict(version='engineering_copy_20261029_v1',scientific_contract_sha256=sha(Q/'CONTRACT.json'),seed43_contract_sha256=sha(Q/'seed43/CONTRACT.json'),validation_sha256=sha(E/'VALIDATION.json'),files={str(E/n):sha(E/n) for n in list(files)+list(existing)},change='Only CUDA-source CPU snapshot redundant clone removal; frozen sources unchanged',jobs=['wait42_natural_exit','train43_optimized','evaluate41','evaluate42','evaluate43'],diagnostic_optimizer_calls=3914,diagnostic_extra_autograd_calls=v['extra_autograd_calls'],diagnostic_environments=0)
(E/'CONTRACT.json').write_text(json.dumps(contract,indent=2))
with (E/'controller.log').open('x') as f:
 p=subprocess.Popen([sys.executable,'-B',str(E/'controller.py')],cwd=E,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1'))
receipt=dict(time=time.time(),pid=p.pid,contract_sha256=sha(E/'CONTRACT.json'),contract=contract)
(E/'DISPATCH.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''.replace('FILES',repr(files)).replace('EXISTING',repr(existing))
ast.parse(script,feature_version=(3,8));(E/'dispatch_handoff.py').write_text(script,encoding='utf8')
print('Handoff dispatch built; not executed')
