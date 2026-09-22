import hashlib,json,os
from decimal import Decimal,localcontext
from pathlib import Path
import numpy as np

def parts(A,b,x):
    p=b-A@x
    d=dict(primal_positive=float(np.maximum(p,0).max(initial=0)),
           nonnegative_violation=float(np.maximum(-x,0).max(initial=0)),
           complementarity=float(np.abs(x*p).max(initial=0)))
    d['kkt']=max(d.values());return d

def mp_parts(A,b,x):
    with localcontext() as ctx:
        ctx.prec=80;cv=lambda a:Decimal.from_float(float(a));zero=Decimal(0)
        xd=[cv(v) for v in x];bd=[cv(v) for v in b]
        r=[bd[i]-sum((cv(A[i,j])*xd[j] for j in range(len(x))),zero) for i in range(len(x))]
        d=dict(primal_positive=max([zero]+r),nonnegative_violation=max([zero]+[-v for v in xd]),
               complementarity=max([zero]+[abs(xd[i]*r[i]) for i in range(len(x))]))
        d['kkt']=max(d.values())
        return {k:str(v) for k,v in d.items()}

def exact_residual(A,b,x):
    with localcontext() as ctx:
        ctx.prec=80;cv=lambda v:Decimal.from_float(float(v));zero=Decimal(0)
        xd=[cv(v) for v in x]
        return np.array([float(cv(b[i])-sum((cv(A[i,j])*xd[j] for j in range(len(x))),zero))
                         for i in range(len(x))],dtype=np.float64)

def strict(A,b,x):
    p=parts(A,b,x);hp=mp_parts(A,b,x)
    return bool(p['kkt']<=1e-8 and Decimal(hp['kkt'])<=Decimal('1e-8')),p,hp

def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)

def save(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.npz')
    with tmp.open('wb') as f:np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    with np.load(path,allow_pickle=False) as z:assert set(z.files)==set(arrays)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def array_sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
