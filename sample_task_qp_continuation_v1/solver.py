"""Small dual convex QP, without truncating the parameter-space constraints."""
import numpy as np
from scipy.optimize import minimize

def solve(K,f,u_norm):
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
            assert kkt<=1e-8,('dual QP did not meet frozen KKT',fit.message,kkt)
            lam[valid]=x*u_norm/scale
        else:kkt=0.
    else:kkt=0.
    return dict(lam=lam.tolist(),valid=valid.tolist(),rates=(f-K@lam).tolist(),
                active=np.flatnonzero(lam>0).tolist(),kkt_error=kkt,iterations=nit,
                correction_norm=float(np.sqrt(max(float(lam@K@lam),0))))
