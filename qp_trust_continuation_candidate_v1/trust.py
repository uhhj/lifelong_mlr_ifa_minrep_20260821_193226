"""Frozen second equivalent solver. Original A/b, bounds and KKT are unchanged."""
import time
import numpy as np
from scipy.optimize import minimize,Bounds
from .numeric import parts
OPTIONS=dict(gtol=1e-12,xtol=1e-12,barrier_tol=1e-12,maxiter=5000,verbose=0)

def run(A,b,x_standard):
    x_standard=np.asarray(x_standard,dtype=np.float64).copy();trace=[];first=[];trajectory=[];started=time.perf_counter()
    objective=lambda x:.5*float(x@A@x)-float(b@x)
    def callback(x,state=None):
        x=np.asarray(x,dtype=np.float64).copy();trajectory.append(x)
        r=parts(A,b,x);iteration=int(getattr(state,'nit',len(trace)+1))
        trace.append(dict(iteration=iteration,objective=objective(x),dx_norm=float(np.linalg.norm(x-x_standard)),optimality=float(state.optimality) if state is not None else None,constr_violation=float(state.constr_violation) if state is not None else None,**r))
        if not first and np.isfinite(x).all() and all(np.isfinite(v) for v in r.values()) and r['kkt']<=1e-8:first.append((iteration,x.copy(),r,objective(x)))
        return False
    fit=minimize(objective,x_standard.copy(),method='trust-constr',jac=lambda x:A@x-b,hess=lambda x:A,bounds=Bounds(np.zeros(len(x_standard),dtype=np.float64),np.full(len(x_standard),np.inf,dtype=np.float64)),options=OPTIONS.copy(),callback=callback)
    final=np.asarray(fit.x,dtype=np.float64).copy();candidate=first[0][1].copy() if first else final.copy();r=parts(A,b,candidate)
    meta=dict(options=OPTIONS.copy(),success=bool(fit.success),status=int(fit.status),message=str(fit.message),nit=int(fit.nit),nfev=int(fit.nfev),njev=int(fit.njev),nhev=int(fit.nhev),optimality=float(fit.optimality),constr_violation=float(fit.constr_violation),seconds=time.perf_counter()-started,first_pass_iteration=first[0][0] if first else None,first_pass_kkt=first[0][2]['kkt'] if first else None,first_pass_objective=first[0][3] if first else None,standard=parts(A,b,x_standard),final=parts(A,b,final),selected=r,selected_source='first_pass_x' if first else 'final_x',objective=objective(candidate),accepted=bool(np.isfinite(candidate).all() and all(np.isfinite(v) for v in r.values()) and r['kkt']<=1e-8),best_seen_kkt=min(v['kkt'] for v in trace))
    return candidate,final,trace,meta,np.stack(trajectory)
