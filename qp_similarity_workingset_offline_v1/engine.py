import time
from decimal import Decimal,localcontext
import numpy as np
from qp_workingset_refinement_diag_v1.refine import run as kernel
from qp_workingset_refinement_diag_v1.numeric import strict,parts
from .grouping import group

def residual80(A,b,x):
    with localcontext() as ctx:
        ctx.prec=80;cv=lambda v:Decimal.from_float(float(v));z=Decimal(0)
        xd=[cv(v) for v in x]
        return [cv(b[i])-sum((cv(A[i,j])*xd[j] for j in range(len(x))),z) for i in range(len(x))]

def run(A,b,valid,arm,tau=.999,random_seed=None,k_design=None):
    began=time.perf_counter();m=len(b);timing=dict(grouping_seconds=0.,selection_seconds=0.,
        subqp_seconds=0.,construction_seconds=0.,full_check_seconds=0.)
    x=np.zeros(m);trace=[];inner_systems=[];nsolves=0;nit=0;last_working_size=0
    def outcome(status,C,k0,first):
        meta=dict(status=status,k0=k0,k_final=len(C),final_candidate_indices=list(C),
            final_nonzero_support=int(np.sum(x>0)),initial_working_set_size=0,
            final_working_set_size=last_working_size,peak_linear_dimension=max([0]+[len(t['indices']) for t in inner_systems]),
            linear_solves=nsolves,inner_iterations=nit,subqp_rounds=len(trace),added_constraints=len(C)-k0,
            first_missing_violations=first['missing_violation_count'],first_missing_max=first['missing_violation_max'],
            float64=trace[-1]['float64'],decimal80=trace[-1]['decimal80'],
            full_kkt_pass=status=='PASS',total_seconds=time.perf_counter()-began,**timing)
        return dict(x=x.copy(),meta=meta,trace=trace,linear_systems=inner_systems,first=first)
    if np.max(b)<=0:
        t=time.perf_counter();ok,p,hp=strict(A,b,x);timing['full_check_seconds']+=time.perf_counter()-t
        first=dict(x=x.copy(),float64=p,decimal80=hp,full_kkt_pass=ok,
            missing_violation_count=0,missing_violation_max=0.,elapsed_seconds=time.perf_counter()-began,
            C=[],nonzero_support=0,inner_iterations=0,working_set_size=0,linear_solves=0,peak_linear_dimension=0,kernel_pass=True,timing=dict(timing))
        trace.append(dict(round=0,C=[],added=[],float64=p,decimal80=hp,kernel_status='SHARED_EXACT_SAFE_ZERO'))
        return outcome('PASS' if ok else 'NUMERICAL_CERTIFICATION_UNRESOLVED',[],0,first)
    t=time.perf_counter()
    if arm=='SIM_ADD':
        groups,C=group(A,valid,tau);timing['grouping_seconds']=time.perf_counter()-t
    elif arm=='RANDOM_ADD':
        assert k_design is not None
        C=sorted(np.random.RandomState(random_seed).choice(m,k_design,replace=False).tolist(),key=lambda i:int(valid[i]))
        timing['selection_seconds']=time.perf_counter()-t
    elif arm=='FULL_WS':C=sorted(range(m),key=lambda i:int(valid[i]))
    else:raise ValueError(arm)
    k0=len(C);first=None
    for round_index in range(m):
        t=time.perf_counter();ix=np.asarray(C,dtype=int)
        subA=A[np.ix_(ix,ix)];subb=b[ix];start=x[ix].copy()
        timing['construction_seconds']+=time.perf_counter()-t
        t=time.perf_counter()
        try:z,km,kt,kpoints=kernel(subA,subb,start)
        except (np.linalg.LinAlgError,AssertionError,ArithmeticError) as exc:
            km=dict(status=type(exc).__name__+':'+str(exc),strict_pass=False,linear_solves=0,iterations=0)
            z=start;kt=[]
        timing['subqp_seconds']+=time.perf_counter()-t
        correction_solves=sum(r.get('linear',{}).get('correction_solve_attempts',0) for r in kt)
        nsolves+=km['linear_solves']+correction_solves;nit+=km['iterations'];last_working_size=len(km.get('working_set',[]))
        for row in kt:
            if 'linear' in row:
                inner_systems.append(dict(round=round_index,iteration=row['iteration'],
                    indices=[C[j] for j in row['working_set']],
                    condition_number=row['linear']['condition_number'],
                    refinement=row['linear']['residual_history'],correction_solve_attempts=row['linear']['correction_solve_attempts'],action=row['action']))
        x=np.zeros(m);x[ix]=z
        t=time.perf_counter()
        # FULL_WS is not charged a redundant full Decimal check: the identical inner
        # kernel has just checked the full original matrix. Same reuse for any full C.
        if np.array_equal(ix,np.arange(m)) and 'parts' in km:
            p,hp=km['parts'],km['decimal80'];ok=km['strict_pass']
        else:ok,p,hp=strict(A,b,x)
        inactive=sorted(set(range(m))-set(C),key=lambda i:int(valid[i]));rf=b-A@x
        add=[];missing_max=0.
        if not ok:
            r80=residual80(A,b,x)
            add=[i for i in inactive if rf[i]>1e-8 or r80[i]>Decimal('1e-8')]
            missing_max=max([0.]+[max(float(rf[i]),float(r80[i])) for i in inactive])
        timing['full_check_seconds']+=time.perf_counter()-t
        if first is None:
            first=dict(x=x.copy(),float64=p,decimal80=hp,full_kkt_pass=bool(ok),
                missing_violation_count=len(add),missing_violation_max=missing_max,
                elapsed_seconds=time.perf_counter()-began,C=list(C),
                nonzero_support=int(np.sum(x>0)),inner_iterations=nit,
                working_set_size=last_working_size,linear_solves=nsolves,
                kernel_pass=bool(km['strict_pass'] and km['status'] in ('STRICT_PASS','UNCHANGED_STANDARD_PASS')),
                peak_linear_dimension=max([0]+[len(r['indices']) for r in inner_systems]),timing=dict(timing))
        trace.append(dict(round=round_index,C=list(C),added=add,float64=p,decimal80=hp,
            kernel_status=km['status'],kernel_strict_pass=km['strict_pass'],
            kernel_iterations=km['iterations'],kernel_linear_solves=km['linear_solves'],linear_refinement_solves=correction_solves))
        if not km['strict_pass'] or km['status'] not in ('STRICT_PASS','UNCHANGED_STANDARD_PASS'):
            return outcome('KERNEL_'+km['status'],C,k0,first)
        if ok:return outcome('PASS',C,k0,first)
        if not add:return outcome('NUMERICAL_CERTIFICATION_UNRESOLVED',C,k0,first)
        if round_index==m-1:return outcome('SUBQP_ROUND_LIMIT',C,k0,first)
        C=sorted(set(C)|set(add),key=lambda i:int(valid[i]))
    raise AssertionError('unreachable')
