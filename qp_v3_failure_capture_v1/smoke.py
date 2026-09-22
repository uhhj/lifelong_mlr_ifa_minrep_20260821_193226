import json,random
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from . import solver,core,step
from .capture import write_json
from sample_qp_optimized_v1_continuation_v3 import solver as original

def main():
    out=Path('reports/qp_v3_failure_capture_v1/smoke');out.mkdir()
    assert core.solve is solver.solve and step.project is core.project
    z=np.load('reports/qp_later_failure_diag_v1/t8_e11_b180/before_extended.npz');args=(z['K'],z['f'],float(z['u_norm']))
    before=random.getstate(); nr=np.random.get_state()
    a=original.solve(*args);b=solver.solve(*args,diag_dir=out/'actual')
    for k in ['lam','valid','rates','active','kkt_error','iterations','correction_norm','base_sweeps','continuation_sweeps']:assert a[k]==b[k],k
    assert random.getstate()==before
    now=np.random.get_state();assert nr[0]==now[0] and np.array_equal(nr[1],now[1]) and nr[2:]==now[2:]
    real=solver.minimize
    def controlled(*args,**kwargs):
        if kwargs.get('method')=='L-BFGS-B':return SimpleNamespace(x=np.zeros_like(args[1]),success=True,message='intentional smoke failure',nit=1,nfev=1,njev=1)
        return real(*args,**kwargs)
    failed=False
    try:
        with patch.object(solver,'minimize',controlled):solver.solve(*args,diag_dir=out/'forced_failure')
    except AssertionError as e:
        assert 'dual QP did not meet frozen KKT after' in str(e);failed=True
    assert failed
    for name in ['before_fallback.npz','after_fallback.npz','fallback_failure_dump.npz']:
        with np.load(out/'forced_failure'/name,allow_pickle=False) as dump:assert dump.files
    assert (out/'forced_failure'/'FAILURE_CONTEXT.json').exists()
    write_json(out.parent/'CAPTURE_SMOKE.json',dict(status='PASS',same_numeric_result=True,explicit_binding=True,failed_dump_readable=True,rng_unchanged=True,formal_training_resumed=False))
    print('CAPTURE_SMOKE PASS',flush=True)
if __name__=='__main__':main()
