import time
from decimal import Decimal,localcontext
from pathlib import Path
import numpy as np
from qp_workingset_refinement_diag_v1.refine import run as kernel,linear
from qp_workingset_refinement_diag_v1.numeric import parts,mp_parts

def make(A,b,cached=None):
    start=time.perf_counter()
    if cached is None:
        x,meta,_,_=kernel(A,b,np.zeros(len(b)))
        if not meta['strict_pass']:return x,dict(status='REFERENCE_UNRESOLVED',reason=meta['status'],seconds=time.perf_counter()-start)
    else:x=np.array(cached,copy=True)
    W=np.flatnonzero(x>0);z=np.zeros_like(x)
    try:
        if len(W):z[W],detail=linear(A[np.ix_(W,W)],b[W])
        else:detail=None
    except np.linalg.LinAlgError as e:
        return x,dict(status='REFERENCE_UNRESOLVED',reason=str(e),seconds=time.perf_counter()-start)
    p=parts(A,b,z);hp=mp_parts(A,b,z)
    ok=p['kkt']<=1e-12 and Decimal(hp['kkt'])<=Decimal('1e-12')
    return z,dict(status='PASS' if ok else 'REFERENCE_UNRESOLVED',float64=p,decimal80=hp,
        support=W.tolist(),linear_detail=detail,seconds=time.perf_counter()-start,
        rule='One full stationary refinement of certified full-QP support; if not <=1e-12 no alternate solver')

def compare(A,b,x,ref,N=None):
    began=time.perf_counter()
    with localcontext() as ctx:
        ctx.prec=80;cv=lambda v:Decimal.from_float(float(v));zero=Decimal(0)
        dx=[cv(a)-cv(r) for a,r in zip(x,ref)];n=len(x)
        terms=[dx[i]*cv(A[i,j])*dx[j] for i in range(n) for j in range(n)]
        q=sum(terms,zero);rounding=Decimal('1e-60')*max(Decimal(1),sum(map(abs,terms),zero))
        if q < -rounding:
            return dict(status='NEGATIVE_UPDATE_QUADRATIC',q80=str(q),negative_roundoff_budget=str(rounding),
                E_update=None,update_budget_pass=False,seconds=time.perf_counter()-began)
        eu=max(q,zero).sqrt();budget=eu<=Decimal('1e-5')
        def obj(v):
            vd=[cv(a) for a in v]
            return sum((vd[i]*cv(A[i,j])*vd[j] for i in range(n) for j in range(n)),zero)/2-sum((cv(b[i])*vd[i] for i in range(n)),zero)
        gap=obj(x)-obj(ref)
        result=dict(status='PASS' if budget else 'UPDATE_BUDGET_EXCEEDED',E_update=float(eu),E_update80=str(eu),
            q80=str(q),negative_roundoff_budget=str(rounding),tiny_negative_q_reported=bool(q<0),
            update_budget_pass=bool(budget),objective_difference80=str(gap),objective_difference=float(gap))
    if N is not None:
        actual=float(np.linalg.norm(N@(x-ref)));err=abs(actual-result['E_update'])
        result.update(N_delta_norm=actual,N_crosscheck_abs_error=err,
            N_crosscheck_pass=err<=1e-10+1e-7*max(actual,result['E_update']))
        if not result['N_crosscheck_pass']:result['status']='GRAM_N_CROSSCHECK_MISMATCH';result['update_budget_pass']=False
    result['seconds']=time.perf_counter()-began
    return result
