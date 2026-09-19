"""Small dual convex QP, without truncating the parameter-space constraints."""
import numpy as np
import json
import os, csv, time
from pathlib import Path
from scipy.optimize import minimize

def solve(K,f,u_norm):
    diag_dir = os.environ.get("QP_DIAG_DIR")
    diag_dir = Path(diag_dir) if diag_dir else None
    if diag_dir: diag_dir.mkdir(parents=True, exist_ok=True)
    K=np.asarray(K,dtype=np.float64);f=np.asarray(f,dtype=np.float64)
    n=len(f);assert K.shape==(n,n) and np.isfinite(K).all() and np.isfinite(f).all()
    assert np.isfinite(u_norm) and u_norm>=0
    assert np.max(np.abs(K-K.T),initial=0)<=1e-9*max(1,float(np.abs(K).max(initial=0)))
    K=(K+K.T)/2;diag=np.diag(K);valid=np.flatnonzero(diag>1e-12)
    lam=np.zeros(n);nit=0
    if len(valid) and u_norm>0:
        scale=np.sqrt(diag[valid]);A=K[np.ix_(valid,valid)]/scale[:,None]/scale[None,:]
        b=f[valid]/scale/u_norm
        if np.max(b)>0:
            objective=lambda x:.5*float(x@A@x)-float(b@x)
            jac=lambda x:A@x-b
            fit=minimize(objective,np.zeros(len(valid)),jac=jac,bounds=[(0,None)]*len(valid),method='SLSQP',options={'ftol':1e-14,'maxiter':10000,'disp':False})
            x=np.maximum(np.asarray(fit.x),0);nit=int(fit.nit)
            # Polish the same convex objective; no ridge or relaxed constraints.
            # SLSQP's objective tolerance does not guarantee a KKT tolerance.
            for sweep in range(10000):
                for i in range(len(x)):
                    x[i]=max(0.,x[i]+(b[i]-float(A[i]@x))/A[i,i])
                residual=b-A@x
                err=max(float(np.maximum(residual,0).max()),float(np.abs(x*residual).max()))
                if err<=1e-11:break
            if err>1e-8:
                for extra in range(20000):
                    for i in range(len(x)):
                        x[i]=max(0.,x[i]+(b[i]-float(A[i]@x))/A[i,i])
                    residual=b-A@x
                    err=max(float(np.maximum(residual,0).max()),float(np.abs(x*residual).max()))
                    if err<=1e-11:break
                sweep += extra+1
            nit+=sweep+1
            primal=b-A@x
            kkt=max(float(np.maximum(primal,0).max()),float(np.maximum(-x,0).max()),float(np.abs(x*primal).max()))
            if kkt>1e-8:
                out=diag_dir
                if out is not None:
                    np.savez(out/'before_extended.npz.tmp.npz',K=K,f=f,u_norm=np.asarray(u_norm),valid=valid,scale=scale,A=A,b=b,x=x.copy(),primal=primal,kkt=np.asarray(kkt),base_sweeps=np.asarray(sweep),continuation_sweeps=np.asarray(sweep),slsqp_iterations=np.asarray(nit)); os.replace(out/'before_extended.npz.tmp.npz',out/'before_extended.npz')
                points={0,100,1000,5000,10000,20000,50000,100000}
                rows=[]
                x_before=x.copy(); t0=time.perf_counter()
                def row(done):
                    rr=b-A@x; pos=float(np.maximum(rr,0).max()); nonneg=float(np.maximum(-x,0).max()); comp=float(np.abs(x*rr).max()); kk=max(pos,nonneg,comp)
                    return {'additional_sweeps':int(done),'primal_positive':pos,'nonnegative_violation':nonneg,'complementarity':comp,'kkt':kk,'objective':float(.5*x@A@x-b@x),'dx_norm':float(np.linalg.norm(x-x_before)),'elapsed_seconds':float(time.perf_counter()-t0)}
                if diag_dir is not None: rows.append(row(0))
                done=0
                for ext in range(100000):
                    for i in range(len(x)): x[i]=max(0.,x[i]+(b[i]-float(A[i]@x))/A[i,i])
                    done=ext+1
                    if diag_dir is not None and done in points: rows.append(row(done))
                    residual=b-A@x; err=max(float(np.maximum(residual,0).max()),float(np.abs(x*residual).max()))
                    if err<=1e-8: break
                sweep += done
                if diag_dir is not None:
                    with (out/'convergence.csv').open('w',newline='') as fh:
                        w=csv.DictWriter(fh,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
            primal=b-A@x
            kkt=max(float(np.maximum(primal,0).max()),float(np.maximum(-x,0).max()),float(np.abs(x*primal).max()))
            if diag_dir is not None and 'out' in locals() and out is not None:
                np.savez(out/'after_extended.npz.tmp.npz',K=K,f=f,u_norm=np.asarray(u_norm),valid=valid,scale=scale,A=A,b=b,x=x.copy(),primal=primal,kkt=np.asarray(kkt),extended_sweeps=np.asarray(sweep),slsqp_iterations=np.asarray(nit)); os.replace(out/'after_extended.npz.tmp.npz',out/'after_extended.npz')
            assert kkt<=1e-8,('dual QP did not meet frozen KKT after bounded extended polish',fit.message,kkt)
            lam[valid]=x*u_norm/scale
        else:kkt=0.
    else:kkt=0.
    return dict(lam=lam.tolist(),valid=valid.tolist(),rates=(f-K@lam).tolist(),
                active=np.flatnonzero(lam>0).tolist(),kkt_error=kkt,iterations=nit,
                correction_norm=float(np.sqrt(max(float(lam@K@lam),0))))
