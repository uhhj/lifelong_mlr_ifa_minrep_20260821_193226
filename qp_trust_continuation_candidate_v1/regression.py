import json,time
from decimal import Decimal
from pathlib import Path
import numpy as np
from . import solver
from .io import write_json
from qp_lbfgsb_kkt_driven_v1.offline import OUT,mp,save_npz
from sample_qp_lifelong_v1_continuation_v3 import common as c
from sample_qp_optimized_v1_continuation_v3.solver import solve as oldsolve

FIXTURES={'F1':c.OUT/'fixtures/F1_dual_qp.npz','F2':c.OUT/'fixtures/F2_dual_qp.npz','B185':c.OUT/'fixtures/B185_dual_qp_before_extended.npz','B258':OUT/'frozen_b258/before_fallback.npz'}

def main():
    assert c.read(OUT/'B258_TRUST_CONSTR.json')['B258_TRUST_CONSTR_PASS']
    expected={'F1':1.5633119541946794e-8,'F2':1.0038755857838533e-8,'B185':4.864608038612772e-8,'B258':1.0093558544532755e-8}
    for name,p in FIXTURES.items():
        z=np.load(p);key='x_before' if name=='B185' else 'x'
        assert max(solver.dual_kkt(z['A'],z['b'],z[key]))==expected[name],('Wrong fixture',name)
    manifest={k:dict(path=str(p),sha256=c.sha(p)) for k,p in FIXTURES.items()};write_json(OUT/'HARD_FIXTURES.json',manifest)
    results={}
    for name,path in FIXTURES.items():
        z=np.load(path);t=time.perf_counter();sol=solver.solve(z['K'],z['f'],float(z['u_norm']),diag_dir=OUT/'hard'/name,context=dict(fixture=name,source_sha256=manifest[name]['sha256']))
        elapsed=time.perf_counter()-t;x=np.array(sol['normalized_x']);hp=mp(z['A'],z['b'],x);lam=np.array(sol['lam']);valid=z['valid'];mapped=lam[valid]/float(z['u_norm'])*z['scale'];hp_mapped=mp(z['A'],z['b'],mapped)
        checks=dict(float64=sol['kkt_error']<=1e-8,decimal80=Decimal(hp['kkt'])<=Decimal('1e-8'),mapped_decimal80=Decimal(hp_mapped['kkt'])<=Decimal('1e-8'),lambda_mapping=np.array_equal(lam[valid],x*float(z['u_norm'])/z['scale']),nonnegative=bool((lam>=0).all()),lbfgsb_expected=sol['fallback']['triggered']==(name!='F1'),trust_expected=sol['second_fallback']['triggered']==(name=='B258'))
        if name!='B258':
            old=oldsolve(z['K'],z['f'],float(z['u_norm']))
            checks['v3_return_unchanged']=all(sol[k]==old[k] for k in ('lam','valid','rates','active','kkt_error','iterations','slsqp_iterations','base_sweeps','continuation_sweeps','correction_norm'))
        else:checks['same_first_pass_as_frozen']=np.array_equal(x,np.load(OUT/'B258_TRUST_CONSTR.npz')['x'])
        if name in ('F2','B185','B258'):
            key='x_before' if name=='B185' else 'x';checks['standard_x_matches_capture']=np.array_equal(np.array(sol['fallback']['standard_x']),z[key])
        results[name]=dict(standard_kkt=sol['standard_kkt'],lbfgsb_called=sol['fallback']['triggered'],lbfgsb_kkt=sol['fallback'].get('fallback_kkt'),trust_constr_called=sol['second_fallback']['triggered'],first_pass_iteration=sol['second_fallback'].get('first_pass_iteration'),accepted_kkt=sol['kkt_error'],decimal80=hp,mapped_decimal80=hp_mapped,objective=.5*float(x@z['A']@x)-float(z['b']@x),seconds=elapsed,checks=checks,status='PASS' if all(checks.values()) else 'FAIL')
        (OUT/'hard'/name).mkdir(parents=True,exist_ok=True);save_npz(OUT/'hard'/name/'accepted.npz',x=x,lam=lam,A=z['A'],b=z['b'])
        write_json(OUT/'HARD_QP_REGRESSION_V3.json',dict(results=results,status='RUNNING',formal_training_resumed=False));assert all(checks.values()),(name,checks)
        print(name,results[name]['accepted_kkt'],results[name]['status'],flush=True)
    write_json(OUT/'HARD_QP_REGRESSION_V3.json',dict(results=results,status='PASS',formal_training_resumed=False))
if __name__=='__main__':main()
