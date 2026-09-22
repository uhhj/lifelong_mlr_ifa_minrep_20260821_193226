"""Deterministic feasible working-set refinement of the original nonnegative QP."""
import time
import numpy as np
from .numeric import parts,strict,exact_residual,array_sha

RULES=dict(max_working_set_iterations=512,max_linear_refinements=3,
    initial_support='x > 0 exactly; no magnitude cutoff',
    enter='largest positive b-Ax among ALL inactive coordinates; ties lowest original index',
    leave='minimum feasible line-search ratio for negative stationary variables; ties lowest index',
    solve='numpy.linalg.solve(A_WW,b_W); exact binary64 inputs, no matrix modification',
    linear_target='1e-12/max(1,max(abs(z))) for 80-digit recomputed absolute equation residual',
    singular_policy='LinAlgError or nonfinite condition number => SINGULAR_WORKING_SET_UNRESOLVED',
    refinement='up to 3 corrections using binary64 solve and Decimal80 residual; no pseudoinverse',
    cycle='repeat of identical working-set and binary64 x bytes => CYCLE_DETECTED',
    acceptance='full float64 AND Decimal80 project KKT <=1e-8',
    already_float64_pass='return original x unchanged; report precision mismatch rather than alter it')

def linear(B,c):
    cond=float(np.linalg.cond(B));logs=[];correction_attempts=0
    if not np.isfinite(cond):raise np.linalg.LinAlgError('nonfinite condition number')
    z=np.linalg.solve(B,c)
    for it in range(RULES['max_linear_refinements']+1):
        if not np.isfinite(z).all():raise np.linalg.LinAlgError('nonfinite direct solution')
        r=exact_residual(B,c,z);norm=float(np.max(np.abs(r),initial=0))
        target=1e-12/max(1.,float(np.max(np.abs(z),initial=0)))
        logs.append(dict(refinement=it,decimal80_residual_max=norm,target=target))
        if norm<=target or it==RULES['max_linear_refinements']:break
        delta=np.linalg.solve(B,r);correction_attempts+=1;next_z=z+delta
        if np.array_equal(next_z,z):break
        z=next_z
    return z,dict(condition_number=cond,residual_history=logs,
                  refinement_corrections=len(logs)-1,correction_solve_attempts=correction_attempts)

def run(A,b,start):
    t=time.perf_counter();A=np.asarray(A,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    x=np.array(start,dtype=np.float64,copy=True);initial=x.copy();trace=[];points=[x.copy()]
    assert A.shape==(len(x),len(x)) and b.shape==x.shape
    assert np.isfinite(A).all() and np.isfinite(b).all() and np.isfinite(x).all()
    W=set(np.flatnonzero(x>0).tolist());seen=set();linear_calls=0;state='ITERATION_LIMIT'
    def finish(status,iteration):
        passed,p,hp=strict(A,b,x)
        return x,dict(status=status,strict_pass=passed,iterations=iteration,linear_solves=linear_calls,
            seconds=time.perf_counter()-t,parts=p,decimal80=hp,
            objective=.5*float(x@A@x)-float(b@x),x_delta_norm=float(np.linalg.norm(x-initial)),
            input_returned_exact=np.array_equal(x,initial),working_set=sorted(W)),trace,np.asarray(points)
    if parts(A,b,x)['kkt']<=1e-8:
        ok,_,_=strict(A,b,x)
        return finish('UNCHANGED_STANDARD_PASS' if ok else 'PRECISION_DECISION_MISMATCH',0)
    if np.any(x<0):return finish('INFEASIBLE_START_UNRESOLVED',0)
    for iteration in range(1,RULES['max_working_set_iterations']+1):
        signature=(tuple(sorted(W)),array_sha(x))
        if signature in seen:return finish('CYCLE_DETECTED',iteration-1)
        seen.add(signature);indices=np.array(sorted(W),dtype=int)
        entry=dict(iteration=iteration,working_set=indices.tolist(),input_parts=parts(A,b,x))
        z=np.zeros_like(x)
        try:
            if len(indices):
                linear_calls+=1;z[indices],detail=linear(A[np.ix_(indices,indices)],b[indices])
                entry['linear']=detail
        except np.linalg.LinAlgError as e:
            entry.update(action='UNRESOLVED',error=str(e));trace.append(entry)
            return finish('SINGULAR_WORKING_SET_UNRESOLVED',iteration)
        negative=indices[z[indices]<0]
        if len(negative):
            ratios=x[negative]/(x[negative]-z[negative])
            selected=int(np.argmin(ratios));alpha=float(ratios[selected]);leave=int(negative[selected])
            assert 0<=alpha<=1
            updated=x+alpha*(z-x)
            rounding_budget=64*np.finfo(float).eps*max(1.,float(np.max(np.abs(x))),float(np.max(np.abs(z))))
            if float(np.min(updated)) < -rounding_budget:
                entry.update(action='NEGATIVE_ROUNDOFF_EXCEEDED',min_x=float(np.min(updated)))
                trace.append(entry);return finish('FEASIBILITY_ARITHMETIC_UNRESOLVED',iteration)
            entry['roundoff_clipped_coordinates']=np.flatnonzero(updated<0).tolist()
            x=np.maximum(updated,0);x[leave]=0.;W.remove(leave)
            entry.update(action='LEAVE',coordinate=leave,alpha=alpha)
        else:
            x=z
            ok,p,hp=strict(A,b,x)
            entry.update(action='STATIONARY',parts=p,decimal80=hp)
            if ok:
                trace.append(entry);points.append(x.copy());return finish('STRICT_PASS',iteration)
            inactive=np.array(sorted(set(range(len(x)))-W),dtype=int)
            primal=b-A@x
            if len(inactive) and np.max(primal[inactive])>0:
                enter=int(inactive[int(np.argmax(primal[inactive]))]);W.add(enter)
                entry.update(action='ENTER',coordinate=enter,full_inactive_scan=True,
                    inactive_max_positive=float(np.max(primal[inactive])))
            else:
                trace.append(entry);points.append(x.copy())
                return finish('STATIONARITY_PRECISION_UNRESOLVED',iteration)
        entry['output_parts']=parts(A,b,x);trace.append(entry);points.append(x.copy())
    return finish('ITERATION_LIMIT',RULES['max_working_set_iterations'])
