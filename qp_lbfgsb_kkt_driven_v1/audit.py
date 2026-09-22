"""Independent recomputation of the single frozen candidate; no parameter search."""
import csv,inspect,json,random,time
from decimal import Decimal
import numpy as np
from scipy.optimize import minimize,_lbfgsb_py
from .offline import OUT,SOURCE,mp,save_npz
from qp_v3_failure_capture_v1.capture import write_json
from sample_qp_lifelong_v1_continuation_v3 import common as c

def main():
    z=np.load(OUT/'frozen_b258/before_fallback.npz');A=z['A'];b=z['b'];x0=z['x'];points=[]
    protocol=c.read(OUT/'FALLBACK_PROTOCOL.json');options=protocol['options'].copy();options['maxcor']=len(x0)
    state=np.random.get_state();py=random.getstate()
    fun=lambda x:.5*float(x@A@x)-float(b@x)
    t=time.perf_counter()
    fit=minimize(fun,x0.copy(),jac=lambda x:A@x-b,bounds=[(0,None)]*len(x0),method='L-BFGS-B',options=options,callback=lambda x:points.append(x.copy()))
    elapsed=time.perf_counter()-t
    state_after=np.random.get_state();assert random.getstate()==py
    assert all(np.array_equal(a,b) if isinstance(a,np.ndarray) else a==b for a,b in zip(state,state_after))
    logged=list(csv.DictReader((OUT/'B258/iterations.csv').open()))
    assert len(points)==len(logged)==39
    rows=[]
    for i,x in enumerate(points):
        p=b-A@x;vals=[max(0,float(np.max(p))),max(0,float(np.max(-x))),float(np.max(np.abs(x*p)))];kkt=max(vals);hp=mp(A,b,x)
        assert kkt==float(logged[i]['kkt'])
        rows.append(dict(iteration=i+1,kkt_float64=kkt,kkt_decimal80=hp['kkt'],passes_float64=kkt<=1e-8,passes_decimal80=Decimal(hp['kkt'])<=Decimal('1e-8')))
    frozen=np.load(OUT/'B258/candidate.npz');assert np.array_equal(frozen['x_final'],np.maximum(fit.x,0))
    save_npz(OUT/'B258/audit_iterates.npz',iterates=np.stack(points),A=A,b=b)
    with (OUT/'B258/iteration_high_precision.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    vals=np.array([r['kkt_float64'] for r in rows]);objectives=np.array([fun(x) for x in points])
    asset=c.read(OUT/'B258_ASSET_VALIDATION.json');unchanged={p:c.sha(p)==v['sha256'] for p,v in asset['assets'].items()};assert all(unchanged.values())
    formal=c.read(c.OUT/'PROTOCOL_FROZEN.json')['files'];files=['sample_qp_optimized_v1_continuation_v3/solver.py','sample_qp_optimized_v1_continuation_v3/core.py','sample_qp_lifelong_v1_continuation_v3/step.py','sample_qp_lifelong_v1_continuation_v3/train.py'];integrity={p:c.sha(c.ROOT/p)==formal[p] for p in files};assert all(integrity.values())
    audit=dict(status='PASS',candidate_rejected=True,all_39_iteration_kkt_reproduced=True,no_pass_at_either_precision=not any(r['passes_float64'] or r['passes_decimal80'] for r in rows),minimum_callback_kkt=float(vals.min()),minimum_callback_iteration=int(vals.argmin())+1,maximum_callback_kkt=float(vals.max()),objective_monotone_nonincreasing=bool(np.all(np.diff(objectives)<=0)),final_two_objectives=[float(v) for v in objectives[-2:]],rng_unchanged=True,asset_hashes_unchanged=unchanged,formal_sources_unchanged=integrity,seconds=elapsed,formal_training_resumed=False)
    write_json(OUT/'INDEPENDENT_AUDIT.json',audit)
    scipy_source=inspect.getsourcefile(_lbfgsb_py);write_json(OUT/'SCIPY_STOPPING_SOURCE.json',dict(path=scipy_source,sha256=c.sha(scipy_source),relevant_lines=[l for l in inspect.getsource(_lbfgsb_py._minimize_lbfgsb).splitlines() if any(k in l for k in ['ftol','factr','pgtol'])]))
    write_json(OUT/'SOURCE_HASHES.json',{str(p.relative_to(c.ROOT)):c.sha(p) for p in (c.ROOT/'qp_lbfgsb_kkt_driven_v1').glob('*.py')})
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
