import json,os
from pathlib import Path
import numpy as np

def _safe(value):
    if isinstance(value,dict):return {k:_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_safe(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return {'nonfinite':repr(float(value))}
    return value

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:json.dump(_safe(value),f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def dump(path,arrays,metadata):
    if path is None:raise RuntimeError('Diagnostic path required for failed QP capture')
    path=Path(path);path.mkdir(parents=True,exist_ok=True);target=path/'QP_RESULT.npz';tmp=path/'QP_RESULT.tmp.npz'
    with tmp.open('wb') as f:np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,target)
    with np.load(target,allow_pickle=False) as z:assert set(z.files)==set(arrays)
    write_json(path/'QP_RESULT.json',metadata)
