from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np

TOLS = [1e-8, 1.1e-8, 2e-8, 5e-8]

def load_fixture(path):
    z=np.load(path, allow_pickle=True)
    return {k:z[k] for k in z.files}

def residual_parts(A,b,x):
    x=np.asarray(x,dtype=np.float64); A=np.asarray(A,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    primal=b-A@x
    return {"primal_positive":float(max(0.,float(np.max(primal)))) if primal.size else 0.,
            "nonnegative_violation":float(max(0.,float(np.max(-x)))) if x.size else 0.,
            "complementarity":float(np.max(np.abs(x*primal))) if x.size else 0.,
            "kkt":float(max(np.max(np.maximum(primal,0)),np.max(np.maximum(-x,0)),np.max(np.abs(x*primal)))) if x.size else 0.,
            "primal_min":float(np.min(primal)) if primal.size else 0.,
            "primal_max":float(np.max(primal)) if primal.size else 0.}

def polish(A,b,x,budget):
    A=np.asarray(A,np.float64); b=np.asarray(b,np.float64); x=np.asarray(x,np.float64).copy()
    d=np.diag(A); hist=[]; t=time.perf_counter()
    for sweep in range(int(budget)):
        for i in range(len(x)):
            x[i]=max(0., x[i]+(b[i]-A[i]@x)/d[i])
        q=residual_parts(A,b,x); q["sweep"]=sweep+1; hist.append(q)
    q=residual_parts(A,b,x); q["seconds"]=time.perf_counter()-t; q["x"]=x; q["history"]=hist
    return q

def mp_residual(A,b,x,dps=80):
    try:
        import mpmath as mp
    except Exception:
        # Decimal is a dependency-free arbitrary precision fallback.  Values are
        # converted from the binary arrays through repr, never from printed logs.
        from decimal import Decimal, getcontext
        getcontext().prec=dps
        AA=[[Decimal(repr(float(v))) for v in row] for row in np.asarray(A)]
        bb=[Decimal(repr(float(v))) for v in np.asarray(b)]
        xx=[Decimal(repr(float(v))) for v in np.asarray(x)]
        pp=[bb[i]-sum((AA[i][j]*xx[j] for j in range(len(xx))),Decimal(0)) for i in range(len(xx))]
        z=Decimal(0)
        pos=max([z]+[max(z,p) for p in pp]); neg=max([z]+[-v for v in xx]); comp=max([z]+[abs(xx[i]*pp[i]) for i in range(len(xx))])
        return {"available":True,"engine":"decimal","dps":dps,"primal_positive":float(pos),"nonnegative_violation":float(neg),"complementarity":float(comp),"kkt":float(max(pos,neg,comp))}
    mp.mp.dps=dps
    AA=[[mp.mpf(str(v)) for v in row] for row in np.asarray(A)]
    bb=[mp.mpf(str(v)) for v in np.asarray(b)]
    xx=[mp.mpf(str(v)) for v in np.asarray(x)]
    pp=[bb[i]-sum(AA[i][j]*xx[j] for j in range(len(xx))) for i in range(len(xx))]
    return {"available":True,"dps":dps,"primal_positive":float(max([mp.mpf(0)]+[max(mp.mpf(0),p) for p in pp])),
            "nonnegative_violation":float(max([mp.mpf(0)]+[-v for v in xx])),
            "complementarity":float(max([abs(xx[i]*pp[i]) for i in range(len(xx))] or [mp.mpf(0)])),
            "kkt":float(max([mp.mpf(0)]+[max(mp.mpf(0),p) for p in pp]+[-v for v in xx]+[abs(xx[i]*pp[i]) for i in range(len(xx))]))}

def run_fixture(name, data):
    A=data.get('A'); b=data.get('b'); x=data.get('x')
    if A is None or b is None or x is None: raise ValueError('fixture lacks A,b,x')
    out={"fixture":name,"shape":list(A.shape),"original":residual_parts(A,b,x),"high_precision":mp_residual(A,b,x)}
    for budget in (0,1000,10000,100000):
        q=polish(A,b,x,budget); out.setdefault("polish",{})[str(budget)]={k:v for k,v in q.items() if k not in ('x','history')}
        if q['kkt']<=1e-10: break
    out['acceptance']={str(t): bool(out['original']['kkt']<=t) for t in TOLS}
    return out
