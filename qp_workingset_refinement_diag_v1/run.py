"""Frozen offline B58 diagnosis and five-fixture regression. No torch/model imports."""
import csv,hashlib,json,os,subprocess,time
from decimal import Decimal
from pathlib import Path
import numpy as np
import scipy
from .numeric import parts,mp_parts,strict,write,save,sha,array_sha
from .refine import run as refine,RULES
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/qp_workingset_refinement_diag_v1'
SOURCE=ROOT/'reports/sample_qp_lifelong_v1_continuation_v4/qp_failures/attempt001_t8_e012_b0058'
V3=ROOT/'reports/sample_qp_lifelong_v1_continuation_v3/fixtures'
FIXTURES={'F1':V3/'F1_dual_qp.npz','F2':V3/'F2_dual_qp.npz',
 'B185':V3/'B185_dual_qp_before_extended.npz',
 'B258':ROOT/'reports/qp_lbfgsb_kkt_driven_v1/frozen_b258/before_fallback.npz',
 'E12_B58':SOURCE/'QP_RESULT.npz'}

def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def work_geometry(A,x):
    W=np.flatnonzero(x>0);B=A[np.ix_(W,W)]
    return dict(indices=W.tolist(),size=len(W),
        condition_number=float(np.linalg.cond(B)) if len(W) else None,
        eigenvalues_symmetric_part=np.linalg.eigvalsh((B+B.T)*.5).tolist() if len(W) else [])

def violations(A,b,x,valid):
    p=b-A@x;components={'primal_positive':np.maximum(p,0),
        'nonnegative_violation':np.maximum(-x,0),'complementarity':np.abs(x*p)}
    report={}
    for name,v in components.items():
        worst=int(np.argmax(v));exceed=np.flatnonzero(v>1e-8)
        report[name]=dict(maximum=float(v[worst]),max_coordinate=worst,
            max_original_reference_coordinate=int(valid[worst]),
            exceeding=[dict(coordinate=int(i),original_reference_coordinate=int(valid[i]),
                        value=float(v[i]),x=float(x[i]),primal=float(p[i])) for i in exceed])
    return report

def candidate(name,A,b,x,valid):
    t=time.perf_counter();ok,p,hp=strict(A,b,x)
    return dict(name=name,float64=p,decimal80=hp,strict_pass=ok,
        objective=.5*float(x@A@x)-float(b@x),seconds=time.perf_counter()-t,
        working_set=work_geometry(A,x),violations=violations(A,b,x,valid),x_sha256=array_sha(x))

def run_refinement(name,A,b,x,valid):
    directory=OUT/'candidates'/name;directory.mkdir(parents=True,exist_ok=True)
    result,meta,trace,trajectory=refine(A,b,x)
    from sample_qp_optimized_v1_continuation_v4.solver import dual_kkt
    original=dict(zip(['primal_positive','nonnegative_violation','complementarity'],dual_kkt(A,b,result)))
    original['kkt']=max(original.values())
    assert original==meta['parts'],'Independent float64 checker disagrees with formal checker'
    meta.update(formal_checker=original,final_working_set_geometry=work_geometry(A,result),
        violations=violations(A,b,result,valid),matrix_sha256=array_sha(A),b_sha256=array_sha(b),
        training_active=False)
    save(directory/'candidate.npz',A=A,b=b,x_start=x,x=result,iterates=trajectory,valid=valid)
    write(directory/'TRACE.json',trace);write(directory/'RESULT.json',meta)
    print(name,meta['status'],meta['parts']['kkt'],meta['iterations'],round(meta['seconds'],6),flush=True)
    return result,meta

def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'PROTOCOL.json').exists(),'No overwrite/re-tuning of an executed frozen diagnostic'
    readonly=list(FIXTURES.values())+[SOURCE/'QP_RESULT.json']
    readonly+=list((ROOT/'sample_qp_optimized_v1_continuation_v4').glob('*.py'))
    readonly+=list((ROOT/'sample_qp_lifelong_v1_continuation_v4').glob('*.py'))
    readonly+=[ROOT/'runs/sample_qp_lifelong_v1_continuation_v4/seed46/sample_qp/RECOVERY_INDEX.json']
    manifest={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in readonly}
    for name in ('QP_RESULT.npz','QP_RESULT.json'):
        rel=str((SOURCE/name).relative_to(ROOT))
        blob=subprocess.check_output(['git','show','ec18946524f0612bc8d60a9575198224fe39fc41:'+rel],cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest()==manifest[rel]['sha256']
    protocol=dict(name='qp_workingset_refinement_diag_v1',rules=RULES,kkt=1e-8,
        source_commit='ec18946524f0612bc8d60a9575198224fe39fc41',
        B58_starts=['x_standard','lbfgsb_candidate','trust_final','trust_best_kkt_iterate'],
        primary_start='x_standard',best_trust='minimum recomputed KKT; first index on ties; start only, never acceptance',
        regression='same refinement on original standard 30k x; F1 standard result regenerated using untouched v4 solver',
        regression_trigger='B58 x_standard refinement passes both checkers',
        old_standard_pass='return exact original x without a linear solve',
        constraints='All original coordinates checked, no permanent removal',
        matrix='Use original unsymmetrized binary64 A in solves/KKT; symmetric part used ONLY for spectral reporting',
        precision='Decimal 80 decimal digits, Decimal.from_float on original binary64 values',
        no_ridge=True,no_pseudoinverse=True,no_eigenvalue_clipping=True,
        numpy=np.__version__,scipy=scipy.__version__,
        source_files={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')},
        training_active=False,formal_solver_modified=False,model_writeback=False)
    write(OUT/'PROTOCOL.json',protocol);write(OUT/'FIXTURES.json',manifest)
    q=load(SOURCE/'QP_RESULT.npz');A=q['A'];b=q['b'];valid=q['valid'];source_meta=json.loads((SOURCE/'QP_RESULT.json').read_text())
    assert A.shape==(64,64) and len(valid)==64
    assert all(np.isfinite(v).all() for v in q.values())
    rates=[parts(A,b,x)['kkt'] for x in q['trust_iterates']]
    best=int(np.argmin(rates));starts={k:q[k] for k in ('x_standard','lbfgsb_candidate','trust_final')}
    starts['trust_best_kkt_iterate']=q['trust_iterates'][best].copy()
    observed={k:candidate(k,A,b,x,valid) for k,x in starts.items()}
    assert observed['x_standard']['float64']==source_meta['standard']
    assert observed['lbfgsb_candidate']['float64']['kkt']==source_meta['lbfgsb']['fallback_kkt']
    assert observed['trust_final']['float64']['kkt']==6.78976598226925e-8
    assert len(source_meta['trust_trace'])==len(rates)
    assert all(r==t['kkt'] for r,t in zip(rates,source_meta['trust_trace']))
    write(OUT/'B58_RECOMPUTATION.json',dict(candidates=observed,trust_best_iteration=best+1,
        minimum_is_not_accepted=True,trust_trace_kkt=rates,training_active=False))
    with (OUT/'B58_COORDINATE_RESIDUALS.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['candidate','coordinate','reference_coordinate','x','primal','primal_positive','nonnegative_violation','complementarity'])
        for name,x in starts.items():
            p=b-A@x
            for i in range(len(x)):w.writerow([name,i,int(valid[i]),x[i],p[i],max(p[i],0),max(-x[i],0),abs(x[i]*p[i])])
    eig=np.linalg.eigvalsh((A+A.T)*.5);sv=np.linalg.svd(A,compute_uv=False)
    corr=np.abs(A[np.triu_indices(len(A),1)])
    geometry=dict(shape=list(A.shape),symmetry_max_abs=float(np.max(np.abs(A-A.T))),
        symmetry_relative_frobenius=float(np.linalg.norm(A-A.T)/np.linalg.norm(A)),
        eig_min=float(eig[0]),eig_max=float(eig[-1]),eigenvalues=eig.tolist(),
        eigen_spectrum_definition='eigvalsh((A+A.T)/2) analysis only; actual solve A never modified',
        cond_A=float(np.linalg.cond(A)),singular_values=sv.tolist(),
        rank_default=int(np.linalg.matrix_rank(A)),
        rank_relative={str(t):int(np.sum(sv>sv[0]*t)) for t in (1e-12,1e-10,1e-8)},
        max_off_diagonal_abs=float(corr.max()),
        high_correlation_unique_pairs={str(t):int(np.sum(corr>t)) for t in (.99,.999,.9999)},
        diag_min=float(np.diag(A).min()),diag_max=float(np.diag(A).max()),
        candidate_working_sets={name:work_geometry(A,x) for name,x in starts.items()})
    write(OUT/'GEOMETRY.json',geometry)
    results={}
    for name,x in starts.items():
        result,meta=run_refinement('B58_'+name,A,b,x,valid);results[name]=meta
        write(OUT/'B58_REFINEMENT.json',results)
    regression={}
    if results['x_standard']['strict_pass']:
        for name,path in FIXTURES.items():
            z=load(path);standard_seconds=0.;standard_regenerated=False
            if name=='F1':
                from sample_qp_optimized_v1_continuation_v4.solver import solve
                t=time.perf_counter();sol=solve(z['K'],z['f'],float(z['u_norm']))
                standard_seconds=time.perf_counter()-t;standard_regenerated=True
                assert not sol['fallback']['triggered'] and not sol['second_fallback']['triggered']
                x=np.asarray(sol['normalized_x'],dtype=np.float64)
            else:x=z['x_before' if name=='B185' else ('x_standard' if name=='E12_B58' else 'x')]
            result,meta=run_refinement('regression_'+name,z['A'],z['b'],x,z['valid'])
            meta['standard_input_kkt']=parts(z['A'],z['b'],x)['kkt']
            meta['standard_generation_seconds']=standard_seconds;meta['standard_regenerated']=standard_regenerated
            if meta['standard_input_kkt']<=1e-8:assert np.array_equal(result,x) and meta['linear_solves']==0
            regression[name]=meta;write(OUT/'REGRESSION.json',regression)
    unchanged={name:sha(ROOT/name)==v['sha256'] for name,v in manifest.items()}
    assert all(unchanged.values()),'Read-only source changed'
    audit=dict(readonly_source_unchanged=unchanged,training_active=False,model_loaded=False,model_writeback=False)
    write(OUT/'READONLY_AUDIT.json',audit)
    summary=dict(B58_standard_refinement_strict_pass=results['x_standard']['strict_pass'],
        all_B58_starts_pass=all(v['strict_pass'] for v in results.values()),
        five_fixture_regression_pass=len(regression)==5 and all(v['strict_pass'] for v in regression.values()),
        unresolved={name:v['status'] for name,v in regression.items() if not v['strict_pass']},
        training_active=False,formal_solver_modified=False,formal_v5_created=False)
    write(OUT/'SUMMARY.json',summary)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
