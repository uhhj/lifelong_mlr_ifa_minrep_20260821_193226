import json, os
from pathlib import Path
import numpy as np

def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def checkpoint(diag_dir, stage, K, f, u_norm, valid, scale, A, b, x, metadata):
    if diag_dir is None:return
    out=Path(diag_dir);out.mkdir(parents=True,exist_ok=True)
    primal=b-A@x
    parts={'primal_positive':float(np.maximum(primal,0).max(initial=0)), 'nonnegative_violation':float(np.maximum(-x,0).max(initial=0)), 'complementarity':float(np.abs(x*primal).max(initial=0))}
    parts['kkt']=max(parts.values())
    path=out/(stage+'.npz'); assert not path.exists(),path
    tmp=path.with_name(path.name+'.tmp.npz')
    with tmp.open('wb') as fh:
        np.savez(fh,K=K,f=f,u_norm=u_norm,valid=valid,scale=scale,A=A,b=b,x=x,primal=primal,metadata_json=json.dumps(metadata,allow_nan=False),**parts)
        fh.flush();os.fsync(fh.fileno())
    os.replace(tmp,path)
    with np.load(path,allow_pickle=False) as z:
        assert np.array_equal(z['x'],x,equal_nan=True)
    write_json(out/(stage+'.json'),dict(stage=stage,parts=parts,metadata=metadata,formal_training_resumed=False))
    if stage=='after_fallback' and parts['kkt']>1e-8:
        write_json(out/'FAILURE_CONTEXT.json',dict(status='FAILED_STOP',parts=parts,metadata=metadata,formal_training_resumed=False))
