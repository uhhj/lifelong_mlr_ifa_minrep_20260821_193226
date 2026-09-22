from pathlib import Path
import numpy as np
from qp_workingset_refinement_diag_v1.run import FIXTURES as REAL_PATHS
ROOT=Path(__file__).resolve().parents[1]
SEEDS=(2026092201,2026092202,2026092203)
KINDS=('low','high','mixed')

def unit(x):return x/np.linalg.norm(x,axis=0,keepdims=True)

def synthetic(m,kind,seed):
    rng=np.random.RandomState(seed);p=128
    if kind=='low':N=unit(rng.normal(size=(p,m)))
    else:
        nc=m//4 if kind=='high' else m//8
        Q,R=np.linalg.qr(rng.normal(size=(p,nc)),mode='reduced')
        centers=Q*np.where(np.diag(R)>=0,1.,-1.)
        cols=[]
        for j in range(nc):
            for _ in range(4):
                noise=unit(rng.normal(size=(p,1)))[:,0]
                cols.append(unit((centers[:,j]+.01*noise)[:,None])[:,0])
        if kind=='mixed':
            for j in range(nc):
                for _ in range(2):
                    noise=unit(rng.normal(size=(p,1)))[:,0]
                    cols.append(unit((-centers[:,j]+.01*noise)[:,None])[:,0])
            cols.extend(unit(rng.normal(size=(p,m//4))).T)
        N=np.column_stack(cols)
    v=unit(rng.normal(size=(p,1)))[:,0]
    assert N.shape==(p,m)
    return dict(A=N.T@N,b=N.T@v,N=N,v=v,valid=np.arange(m),scale=np.ones(m),u_norm=np.asarray(1.))

def fixture_specs():
    result=[]
    for seed in SEEDS[:2]:
        for kind in KINDS:
            for m in (32,64,96):
                result.append(dict(id='S_%s_m%d_s%d'%(kind,m,seed),kind=kind,m=m,seed=seed,
                    source='synthetic',split='implementation'))
    for name,path in REAL_PATHS.items():
        result.append(dict(id=name,source='real',split='stress',kind='real',m=64,path=str(path)))
    for kind in KINDS:
        for m in (32,64,96):
            result.append(dict(id='S_%s_m%d_s%d'%(kind,m,SEEDS[2]),kind=kind,m=m,seed=SEEDS[2],
                source='synthetic',split='confirmation'))
    return result

def real(path):
    # No x, support or solver outcome enters the comparison-method interface.
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in ('A','b','valid','scale','u_norm')}
