"""Unchanged v3 standard path and first L-BFGS-B fallback.

Only if both fail the frozen KKT gate, trust-constr starts from the original
30k iterate and uses the first certified callback iterate. No QP, scaling,
constraint, lambda-map or acceptance-tolerance changes. Atomic failure capture
retains both trajectories and the pre-update batch context.
"""
import time

import numpy as np
from scipy.optimize import minimize
from .trust import run as trust_run
from .numeric import parts
from .io import dump

# Frozen deterministic configuration for the dual fallback. Do not tune per
# instance; the same options apply to every dual QP, in regression and in
# formal training. maxcor is completed with len(x) at solve time (full-memory
# BFGS; the dual dimension never exceeds 96, so the memory cost is trivial
# and the result is deterministic).
FALLBACK_OPTIONS = dict(maxiter=100000, maxfun=200000, ftol=1e-15,
                        gtol=1e-12, maxls=50)

KKT_TOL = 1e-8


def dual_kkt(A, b, x):
    """Project KKT checker: feasibility, nonnegativity, complementarity."""
    primal = b - A @ x
    primal_positive = float(np.maximum(primal, 0).max(initial=0.0))
    nonnegative_violation = float(np.maximum(-x, 0).max(initial=0.0))
    complementarity = float(np.abs(x * primal).max(initial=0.0))
    return primal_positive, nonnegative_violation, complementarity


def solve(K, f, u_norm, diag_dir=None, context=None):
    K = np.asarray(K, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    n = len(f)
    assert K.shape == (n, n) and np.isfinite(K).all() and np.isfinite(f).all()
    assert np.isfinite(u_norm) and u_norm >= 0
    assert np.max(np.abs(K - K.T), initial=0) <= 1e-9 * max(1, float(np.abs(K).max(initial=0)))
    K = (K + K.T) / 2
    diag = np.diag(K)
    valid = np.flatnonzero(diag > 1e-12)
    lam = np.zeros(n)
    nit = 0
    base_done = 0
    continuation_done = 0
    slsqp_nit = 0
    fallback = dict(triggered=False)
    second_fallback = dict(triggered=False)
    normalized_x = np.zeros(len(valid))
    standard_kkt = 0.
    if len(valid) and u_norm > 0:
        scale = np.sqrt(diag[valid])
        A = K[np.ix_(valid, valid)] / scale[:, None] / scale[None, :]
        b = f[valid] / scale / u_norm
        if np.max(b) > 0:
            objective = lambda x: .5 * float(x @ A @ x) - float(b @ x)
            jac = lambda x: A @ x - b
            fit = minimize(objective, np.zeros(len(valid)), jac=jac,
                           bounds=[(0, None)] * len(valid), method='SLSQP',
                           options={'ftol': 1e-14, 'maxiter': 10000, 'disp': False})
            x = np.maximum(np.asarray(fit.x), 0)
            nit = int(fit.nit)
            slsqp_nit = int(fit.nit)
            # Polish the same convex objective; no ridge or relaxed constraints.
            # SLSQP's objective tolerance does not guarantee a KKT tolerance.
            for sweep in range(10000):
                for i in range(len(x)):
                    x[i] = max(0., x[i] + (b[i] - float(A[i] @ x)) / A[i, i])
                base_done = sweep + 1
                residual = b - A @ x
                err = max(float(np.maximum(residual, 0).max()),
                          float(np.abs(x * residual).max()))
                if err <= 1e-11:
                    break
            if err > KKT_TOL:
                for extra in range(20000):
                    for i in range(len(x)):
                        x[i] = max(0., x[i] + (b[i] - float(A[i] @ x)) / A[i, i])
                    continuation_done = extra + 1
                    residual = b - A @ x
                    err = max(float(np.maximum(residual, 0).max()),
                              float(np.abs(x * residual).max()))
                    if err <= 1e-11:
                        break
            nit += base_done + continuation_done
            primal_positive, nonneg, comp = dual_kkt(A, b, x)
            kkt = max(primal_positive, nonneg, comp)
            standard_kkt = kkt
            if kkt > KKT_TOL:
                # Deterministic L-BFGS-B fallback from the 30k-polish iterate.
                x_standard = x.copy()
                t0 = time.perf_counter()
                lbfgsb_trace=[];lbfgsb_points=[]
                def lbfgsb_callback(v):
                    q=np.asarray(v,dtype=np.float64).copy();lbfgsb_points.append(q)
                    lbfgsb_trace.append(dict(iteration=len(lbfgsb_trace)+1,objective=objective(q),**parts(A,b,q)))
                fb = minimize(objective, x_standard, jac=jac,
                              bounds=[(0, None)] * len(x_standard),
                              method='L-BFGS-B',
                              options=dict(FALLBACK_OPTIONS,
                                           maxcor=len(x_standard)),callback=lbfgsb_callback)
                seconds = time.perf_counter() - t0
                candidate = np.maximum(np.asarray(fb.x, dtype=np.float64), 0.0)
                fp, fn, fc = dual_kkt(A, b, candidate)
                fkkt = max(fp, fn, fc)
                fallback = dict(triggered=True, success=bool(fb.success),
                                message=str(fb.message), nit=int(fb.nit),
                                nfev=int(fb.nfev), njev=int(fb.njev),
                                seconds=float(seconds), standard_kkt=float(kkt),
                                fallback_kkt=float(fkkt),
                                primal_positive=float(fp),
                                nonnegative_violation=float(fn),
                                complementarity=float(fc),
                                candidate_objective=float(objective(candidate)),
                                standard_x=x_standard.tolist(),
                                accepted=bool(fkkt <= KKT_TOL))
                if fkkt <= KKT_TOL:
                    x = candidate
                    kkt = fkkt
                    nit += int(fb.nit)
                else:
                    tc_candidate,tc_final,tc_trace,tc_meta,tc_points=trust_run(A,b,x_standard)
                    tp,tn,tc=dual_kkt(A,b,tc_candidate);tkkt=max(tp,tn,tc)
                    second_fallback=dict(tc_meta,triggered=True)
                    finite=np.isfinite(tc_candidate).all() and np.isfinite([tp,tn,tc]).all()
                    accepted=bool(finite and tkkt<=KKT_TOL)
                    arrays=dict(K=K,f=f,u_norm=np.asarray(u_norm),valid=valid,scale=scale,A=A,b=b,x_standard=x_standard,lbfgsb_candidate=candidate,trust_candidate=tc_candidate,trust_final=tc_final,lbfgsb_iterates=np.asarray(lbfgsb_points).reshape(-1,len(x_standard)),trust_iterates=tc_points)
                    metadata=dict(context=context,base_sweeps=base_done,continuation_sweeps=continuation_done,slsqp_iterations=slsqp_nit,standard=parts(A,b,x_standard),lbfgsb=fallback,trust=second_fallback,lbfgsb_trace=lbfgsb_trace,trust_trace=tc_trace,accepted=accepted,kkt_tolerance=KKT_TOL)
                    if diag_dir is not None or not accepted:dump(diag_dir,arrays,metadata)
                    if not accepted:raise AssertionError('dual QP failed frozen KKT after both fallbacks',tkkt)
                    x=tc_candidate;kkt=tkkt;nit+=int(fb.nit)+int(tc_meta['nit'])
            normalized_x=x.copy()
            lam[valid] = x * u_norm / scale
        else:
            kkt = 0.
    else:
        kkt = 0.
    return dict(lam=lam.tolist(), valid=valid.tolist(), rates=(f - K @ lam).tolist(),
                active=np.flatnonzero(lam > 0).tolist(), kkt_error=kkt, iterations=nit,
                correction_norm=float(np.sqrt(max(float(lam @ K @ lam), 0))),
                slsqp_iterations=slsqp_nit,
                base_sweeps=int(base_done), continuation_sweeps=int(continuation_done),
                fallback=fallback,second_fallback=second_fallback,standard_kkt=standard_kkt,normalized_x=normalized_x.tolist())
