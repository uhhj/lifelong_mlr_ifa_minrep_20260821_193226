"""Independent exact-rational residual audit of saved binary64 candidates."""
import json,time,sys
from decimal import Decimal,localcontext
from fractions import Fraction
from pathlib import Path
import numpy as np
from .numeric import write,array_sha,sha
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/qp_workingset_refinement_diag_v1'

def rational_parts(A,b,x):
    cv=lambda v:Fraction.from_float(float(v));xd=list(map(cv,x));zero=Fraction(0)
    residual=[cv(b[i])-sum((cv(A[i,j])*xd[j] for j in range(len(x))),zero) for i in range(len(x))]
    d={'primal_positive':max([zero]+residual),'nonnegative_violation':max([zero]+[-v for v in xd]),
       'complementarity':max([zero]+[abs(xd[i]*residual[i]) for i in range(len(x))])}
    d['kkt']=max(d.values())
    with localcontext() as ctx:
        ctx.prec=80
        display={k:str(Decimal(v.numerator)/Decimal(v.denominator)) for k,v in d.items()}
    return d,display

def main():
    results={};start=time.perf_counter()
    for directory in sorted((OUT/'candidates').iterdir()):
        if not directory.is_dir():continue
        t=time.perf_counter();m=json.loads((directory/'RESULT.json').read_text())
        with np.load(directory/'candidate.npz',allow_pickle=False) as z:
            A,b,x=z['A'],z['b'],z['x'];r,display=rational_parts(A,b,x)
            assert array_sha(A)==m['matrix_sha256'] and array_sha(b)==m['b_sha256']
            passed=r['kkt']<=Fraction(1,100000000)
            assert passed==m['strict_pass']
            difference={k:abs(Decimal(display[k])-Decimal(m['decimal80'][k])) for k in display}
            assert max(difference.values())<Decimal('1e-65')
            results[directory.name]=dict(exact_rational_pass=passed,decimal80_from_exact_rational=display,
                decimal80_checker_abs_difference={k:str(v) for k,v in difference.items()},
                x_nonnegative=bool(np.all(x>=0)),all_64_original_coordinates_checked=len(x)==64,
                seconds=time.perf_counter()-t)
    readonly=json.loads((OUT/'FIXTURES.json').read_text())
    unchanged={k:sha(ROOT/k)==v['sha256'] for k,v in readonly.items()}
    assert all(unchanged.values())
    p=json.loads((OUT/'PROTOCOL.json').read_text())
    assert all(sha(ROOT/k)==v for k,v in p['source_files'].items())
    assert 'torch' not in sys.modules
    result=dict(status='PASS',results=results,readonly_unchanged=unchanged,
        frozen_diagnostic_sources_unchanged=True,torch_imported=False,training_active=False,
        seconds=time.perf_counter()-start)
    write(OUT/'INDEPENDENT_AUDIT.json',result);print('EXACT_RATIONAL_AUDIT_PASS',len(results),result['seconds'])
if __name__=='__main__':main()
