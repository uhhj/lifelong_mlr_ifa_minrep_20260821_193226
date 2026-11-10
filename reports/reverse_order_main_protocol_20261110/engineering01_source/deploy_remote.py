"""One text-only staging upload. It never starts a scientific controller."""
from pathlib import Path
import ast,hashlib,json,os,time
Q=Path('/data/mlr_ifa/experiment_workspaces/REVERSE_MAIN_MLR_DEBR_OBJECT_GOAL_20261110_v1')
P=Path('/data/mlr_ifa/experiments/REVERSE_MAIN_MLR_DEBR_OBJECT_GOAL_20261110_v1')
assert not Q.exists() and not P.exists(),'New staging must be unique'
assert len(BUNDLE)>1000
v=os.statvfs('/data');assert v.f_bavail*v.f_frsize>=2*1024**3+256*1024**2
v=os.statvfs('/');assert v.f_bavail*v.f_frsize>=1024**3+64*1024**2
for relative,text in BUNDLE.items():
    rel=Path(relative);assert not rel.is_absolute() and '..' not in rel.parts
    assert rel.suffix in ['.py','.json','.yaml','.yml']
    assert '\x00' not in text
    if rel.suffix=='.py':ast.parse(text,filename=relative,feature_version=(3,8))
    if rel.suffix=='.json':json.loads(text)
Q.mkdir(parents=True)
files={}
for relative,text in BUNDLE.items():
    path=Q/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(text.encode('utf-8'))
    files[relative]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
receipt=dict(status='STAGED_TEXT_ONLY_NOT_CONTROLLER_LAUNCH',observed=time.time(),root=str(Q),files=files,new_optimizer=0,new_environments=0,new_model_forward=0)
(Q/'STAGED.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
