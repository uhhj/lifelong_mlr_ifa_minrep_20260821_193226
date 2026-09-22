import csv,json
from decimal import Decimal
import numpy as np
from .trust import run,OPTIONS
from .offline import OUT,mp,save_npz
from qp_v3_failure_capture_v1.capture import write_json
from sample_qp_lifelong_v1_continuation_v3 import common as c

def main():
    assert not c.read(OUT/'B258/RESULT.json')['passed']
    protocol=dict(name='TRUST_CONSTR_SECOND_FALLBACK_V1',options=OPTIONS,starting_point='original x_standard after SLSQP+10k+20k; never failed L-BFGS-B x',A_modification=False,kkt_tolerance=1e-8,selection='first callback pass else final x',callback_does_not_stop=True,candidate_formal_path='unchanged standard -> unchanged v3 LBFGSB -> trust-constr from standard if both failed',formal_training_resumed=False)
    if (OUT/'TRUST_PROTOCOL.json').exists():assert c.read(OUT/'TRUST_PROTOCOL.json')==protocol
    else:write_json(OUT/'TRUST_PROTOCOL.json',protocol)
    z=np.load(OUT/'frozen_b258/before_fallback.npz');x,final,trace,m,points=run(z['A'],z['b'],z['x'])
    m['selected_decimal80']=mp(z['A'],z['b'],x);m['final_decimal80']=mp(z['A'],z['b'],final)
    m['B258_TRUST_CONSTR_PASS']=m['accepted'] and Decimal(m['selected_decimal80']['kkt'])<=Decimal('1e-8');m['formal_training_resumed']=False
    with (OUT/'B258_TRUST_CONSTR_TRACE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(trace[0]));w.writeheader();w.writerows(trace)
    save_npz(OUT/'B258_TRUST_CONSTR.npz',A=z['A'],b=z['b'],x_standard=z['x'],x=x,x_final=final,first_pass_x=x if m['first_pass_iteration'] is not None else np.empty(0),iterates=points)
    write_json(OUT/'B258_TRUST_CONSTR.json',m)
    write_json(OUT/'STATUS.json',dict(stage='B258_TRUST_CONSTR_PASS' if m['B258_TRUST_CONSTR_PASS'] else 'B258_EQUIVALENT_SOLVER_UNRESOLVED',formal_training_resumed=False))
    print(json.dumps(m,indent=2,allow_nan=False),flush=True)
if __name__=='__main__':main()
