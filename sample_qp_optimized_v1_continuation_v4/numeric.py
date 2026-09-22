import numpy as np
def parts(A,b,x):
    p=b-A@x
    r=dict(primal_positive=float(np.maximum(p,0).max(initial=0)),nonnegative_violation=float(np.maximum(-x,0).max(initial=0)),complementarity=float(np.abs(x*p).max(initial=0)))
    r['kkt']=max(r.values())
    return r
