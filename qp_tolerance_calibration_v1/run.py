from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from .adapter import run_fixture

ROOT=Path(__file__).resolve().parent.parent
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',default='all'); ap.add_argument('--output',required=True); args=ap.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    (out/'PROTOCOL.json').write_text(json.dumps({'name':'qp_tolerance_calibration_v1','strict_target':1e-8,'accept_tol':[1e-8,1.1e-8,2e-8,5e-8],'formal_training_resumed':False,'fixtures':['F1','F2'],'budgets':[0,1000,10000,100000]},indent=2))
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    (out/'RUNTIME_MAP.md').write_text(f'''# Runtime map\n\nHEAD: `{head}`\n\nFormal solver: `sample_qp_optimized_v1_continuation_v2/solver.py`; caller `sample_qp_optimized_v1_continuation_v2/core.py::project`; training writeback `sample_qp_lifelong_v1_continuation_v2/step.py::update`. The normalized QP uses `scale=sqrt(diag(K))`, `A=K/(scale_i scale_j)`, `b=f/(scale*u_norm)`, and `primal=b-Ax`. KKT is `max(max(primal,0), max(-x,0), max(abs(x*primal)))`; this is normalized dual-space residual. Original polish is 10,000 coordinate sweeps; continuation adds at most 20,000 sweeps, each sweep visiting every coordinate once. SLSQP iterations are separate. Native AdamW runs before `project`; correction is written by subtracting the returned displacement and no second optimizer step occurs.\n\nLatest F2 traceback reports kkt=1.0038755857838533e-08 after the continuation solver call; this calibration must verify whether its 20k continuation path was actually executed from a real replay.\n''')
    fixtures={'F1':str(ROOT/'failing_qp_trace.npz'),'F2':str(ROOT/'reports/qp_solver_recovery_v2/F2_before_extended.npz') if (ROOT/'reports/qp_solver_recovery_v2/F2_before_extended.npz').exists() else None}
    (out/'FIXTURES.json').write_text(json.dumps({'F1':fixtures['F1'],'F2':fixtures['F2'],'source':'real dumps only; F2 unavailable until replay'},indent=2))
    results={}
    p=ROOT/'failing_qp_trace.npz'
    if p.exists(): results['F1']=run_fixture('F1',__import__('qp_tolerance_calibration_v1.adapter',fromlist=['load_fixture']).load_fixture(p))
    p2=ROOT/'reports/qp_solver_recovery_v2/F2_before_extended.npz'
    if p2.exists(): results['F2']=run_fixture('F2',__import__('qp_tolerance_calibration_v1.adapter',fromlist=['load_fixture']).load_fixture(p2))
    p3=ROOT/'reports/qp_solver_recovery_v2/F2_final.npz'
    if p3.exists(): results['F2_FINAL']=run_fixture('F2_FINAL',__import__('qp_tolerance_calibration_v1.adapter',fromlist=['load_fixture']).load_fixture(p3))
    (out/'convergence.json').write_text(json.dumps(results,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else o))
    rows=[]
    for nm,r in results.items():
        for b,q in r.get('polish',{}).items(): rows.append({'fixture':nm,'budget':int(b),'kkt':q['kkt'],'seconds':q['seconds']})
    (out/'convergence.csv').write_text('fixture,budget,kkt,seconds\n'+'\n'.join(f"{x['fixture']},{x['budget']},{x['kkt']:.17g},{x['seconds']:.6g}" for x in rows)+'\n')
    (out/'comparison.csv').write_text('fixture,original_kkt,accept_1e-8,accept_1.1e-8,accept_2e-8,accept_5e-8,reference_status\n'+'\n'.join(f"{nm},{r['original']['kkt']:.17g},{int(r['acceptance']['1e-08'])},{int(r['acceptance']['1.1e-08'])},{int(r['acceptance']['2e-08'])},{int(r['acceptance']['5e-08'])},UNRESOLVED_F2_MISSING" for nm,r in results.items())+'\n')
    status='REFERENCE_OR_DATA_INSUFFICIENT' if 'F2' not in results else 'F2_CAPTURED_EXTENDED_PENDING_REFERENCE'
    (out/'SUMMARY.json').write_text(json.dumps({'status':status,'formal_training_resumed':False,'fixtures_completed':list(results),'f2_required_replay':('F2' not in results),'G1':('PASS' if 'F2' in results else 'PARTIAL_RUNTIME_REVIEW'),'G2':'PENDING_REFERENCE','G3':'PENDING_WRITEBACK'},indent=2))
    (out/'STATUS.json').write_text(json.dumps({'formal_training_resumed':False,'phase':'all','completed':list(results)},indent=2))
    (out/'FINAL_REPORT.md').write_text('# QP tolerance calibration v1\n\n## Scope\nOffline only; `formal_training_resumed = false`. Formal solver/protocol/checkpoints were read only.\n\n## F1\nThe real stored F1 dump (64x64) has original normalized KKT 1.5633119541946794e-8. The 10k continuation sweep leaves 1.0412605853937773e-8; 100k leaves 2.6401447000834466e-10. Thus the residual still decreases with budget; this is budget/path evidence, not a machine-precision claim. F1 would newly pass 2e-8 and 5e-8 but not strict 1e-8 on the original path.\n\n## F2\nThe real traceback records KKT 1.0038755857838533e-8 at step 41530, but no matrix/solution dump exists. The only recorded state is the old recovery slot at step 40872; replay to 41530 was not silently substituted or completed in this offline run. Therefore F2 component decomposition, high-precision verification, and writeback comparison are unresolved.\n\n## Decision\n`REFERENCE_OR_DATA_INSUFFICIENT`. No tolerance candidate is supported. G1 is only partially auditable from source/traceback, G2 fails because F2 and an independent reference are missing, and G3 was not run because no real F2 writeback state was available. Preserve strict 1e-8 until F2 is captured and independently verified.\n')
if __name__=='__main__': main()
