"""Small dual convex QP with a deterministic L-BFGS-B fallback.

Relative to the v2 solver, the ONLY algorithmic change is:
when the original SLSQP + 10k coordinate polish + 20k continuation polish
path still yields KKT > 1e-8, a frozen-configuration L-BFGS-B solve from the
30k-polish iterate replaces the former 100k extended coordinate polish.

The QP, constraints, normalization, dual scaling, lambda mapping, correction
construction and the frozen 1e-8 KKT threshold are all unchanged. The
optimizer's own success flag is never used for acceptance; the project KKT
checker is the only gate.
"""
import time
from .capture import checkpoint

import numpy as np
from scipy.optimize import minimize

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


def solve(K, f, u_norm, diag_dir=None):
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
            if kkt > KKT_TOL:
                # Deterministic L-BFGS-B fallback from the 30k-polish iterate.
                x_standard = x.copy()
                checkpoint(diag_dir, 'before_fallback', K, f, u_norm, valid, scale, A, b, x_standard, dict(base_done=base_done, continuation_done=continuation_done, slsqp_iterations=slsqp_nit, options=dict(FALLBACK_OPTIONS, maxcor=len(x_standard))))
                t0 = time.perf_counter()
                fb = minimize(objective, x_standard, jac=jac,
                              bounds=[(0, None)] * len(x_standard),
                              method='L-BFGS-B',
                              options=dict(FALLBACK_OPTIONS,
                                           maxcor=len(x_standard)))
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
                checkpoint(diag_dir, 'after_fallback', K, f, u_norm, valid, scale, A, b, candidate, dict(base_done=base_done, continuation_done=continuation_done, slsqp_iterations=slsqp_nit, fallback=fallback, options=dict(FALLBACK_OPTIONS, maxcor=len(x_standard))))
                if fkkt <= KKT_TOL:
                    x = candidate
                    kkt = fkkt
                    nit += int(fb.nit)
                else:
                    if diag_dir is not None:
                        from pathlib import Path
                        out = Path(diag_dir)
                        out.mkdir(parents=True, exist_ok=True)
                        tmp = out / 'fallback_failure_dump.npz.tmp.npz'
                        np.savez(tmp, K=K, f=f, u_norm=np.asarray(u_norm),
                                 valid=valid, scale=scale, A=A, b=b,
                                 x_standard=x_standard, candidate=candidate,
                                 standard_kkt=np.asarray(kkt),
                                 fallback_kkt=np.asarray(fkkt),
                                 slsqp_iterations=np.asarray(fit.nit),
                                 base_done=np.asarray(base_done),
                                 continuation_done=np.asarray(continuation_done),
                                 fallback_nit=np.asarray(fb.nit))
                        import os
                        os.replace(tmp, out / 'fallback_failure_dump.npz')
                    raise AssertionError('dual QP did not meet frozen KKT after '
                                         'L-BFGS-B fallback', fb.message, fkkt)
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
                fallback=fallback)
