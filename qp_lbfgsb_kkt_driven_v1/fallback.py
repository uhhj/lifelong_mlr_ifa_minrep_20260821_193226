import time
import numpy as np
from scipy.optimize import minimize

OPTIONS=dict(maxiter=100000,maxfun=200000,ftol=0.0,gtol=0.0,maxls=50)
KKT_TOL=1e-8

def parts(A,b,x):
    p=b-A@x
    r=dict(primal_positive=float(np.maximum(p,0).max(initial=0)),nonnegative_violation=float(np.maximum(-x,0).max(initial=0)),complementarity=float(np.abs(x*p).max(initial=0)))
    r['kkt']=max(r.values())
    return r

def run(A,b,x_standard):
    start=time.perf_counter();x_standard=np.asarray(x_standard,dtype=np.float64).copy()
    objective=lambda x:.5*float(x@A@x)-float(b@x)
    jac=lambda x:A@x-b
    trace=[];first=[]
    def callback(xk):
        x=np.asarray(xk,dtype=np.float64).copy()
        r=parts(A,b,x)
        trace.append(dict(iteration=len(trace)+1,objective=objective(x),dx_norm=float(np.linalg.norm(x-x_standard)),**r))
        if np.isfinite(x).all() and all(np.isfinite(v) for v in r.values()) and r['kkt']<=KKT_TOL and not first:
            first.append((len(trace),x.copy(),r))
    fit=minimize(objective,x_standard,jac=jac,bounds=[(0.0,None)]*len(x_standard),method='L-BFGS-B',options=dict(OPTIONS,maxcor=len(x_standard)),callback=callback)
    final=np.maximum(np.asarray(fit.x,dtype=np.float64),0.0)
    candidate=first[0][1].copy() if first else final.copy()
    selected=parts(A,b,candidate)
    accepted=bool(np.isfinite(candidate).all() and all(np.isfinite(v) for v in selected.values()) and selected['kkt']<=KKT_TOL)
    meta=dict(name='LBFGSB_KKT_DRIVEN_V1',options=dict(OPTIONS,maxcor=len(x_standard)),success=bool(fit.success),message=str(fit.message),nit=int(fit.nit),nfev=int(fit.nfev),njev=int(fit.njev),callback_count=len(trace),seconds=time.perf_counter()-start,standard=parts(A,b,x_standard),final=parts(A,b,final),first_pass_iteration=first[0][0] if first else None,first_pass_kkt=first[0][2]['kkt'] if first else None,selected_source='first_pass_x' if first else 'final_x',selected=selected,accepted=accepted,objective=objective(candidate))
    return candidate,final,trace,meta
