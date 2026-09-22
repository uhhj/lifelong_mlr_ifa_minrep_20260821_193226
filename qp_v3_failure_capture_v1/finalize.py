"""Independent read-only verification of the captured QP; no optimization."""
from decimal import Decimal, localcontext
import json
from pathlib import Path
import numpy as np
from .capture import write_json
from sample_qp_lifelong_v1_continuation_v3 import common as c

OUT=c.ROOT/'reports/qp_v3_failure_capture_v1'
D=OUT/'t8_e11_b258'

def parts64(z):
    p=z['b']-z['A']@z['x']
    p0=float(np.maximum(p,0).max());p1=float(np.maximum(-z['x'],0).max());p2=float(np.abs(z['x']*p).max())
    return dict(primal_positive=p0,nonnegative_violation=p1,complementarity=p2,kkt=max(p0,p1,p2))

def parts80(z):
    with localcontext() as ctx:
        ctx.prec=80
        cv=lambda v:Decimal.from_float(float(v))
        x=[cv(v) for v in z['x']];b=[cv(v) for v in z['b']]
        p=[b[i]-sum((cv(z['A'][i,j])*x[j] for j in range(len(x))),Decimal(0)) for i in range(len(x))]
        zero=Decimal(0)
        p0=max([zero]+p);p1=max([zero]+[-v for v in x]);p2=max(abs(x[i]*p[i]) for i in range(len(x)))
        return {k:str(v) for k,v in dict(primal_positive=p0,nonnegative_violation=p1,complementarity=p2,kkt=max(p0,p1,p2)).items()}

def main():
    status=c.read(OUT/'STATUS.json');assert status['stage']=='HISTORICAL_FAILURE_CAPTURED',status
    a=np.load(D/'before_fallback.npz',allow_pickle=False);b=np.load(D/'after_fallback.npz',allow_pickle=False)
    matches={k:np.array_equal(a[k],b[k]) for k in ('K','f','u_norm','valid','scale','A','b')};assert all(matches.values())
    assert all(np.isfinite(z[k]).all() for z in (a,b) for k in ('K','f','u_norm','scale','A','b','x','primal'))
    readings={}
    for name,z in [('before',a),('after',b)]:
        p64=parts64(z);p80=parts80(z)
        assert p64['kkt']==float(z['kkt'])
        readings[name]=dict(float64=p64,decimal80=p80,metadata=json.loads(str(z['metadata_json'])))
    rows=[json.loads(l) for l in (OUT/'short_replay_events.jsonl').read_text().splitlines()]
    assert len(rows)==72 and all(r['exact_event_match'] for r in rows)
    frozen=c.read(c.OUT/'PROTOCOL_FROZEN.json')['files']
    formal_files=['sample_qp_optimized_v1_continuation_v3/solver.py','sample_qp_optimized_v1_continuation_v3/core.py','sample_qp_lifelong_v1_continuation_v3/step.py','sample_qp_lifelong_v1_continuation_v3/train.py']
    unchanged={p:c.sha(c.ROOT/p)==frozen[p] for p in formal_files};assert all(unchanged.values())
    state=c.read(D/'PRE_FAILURE_STATE.json');assert all(state['save_unchanged'].values()) and all(state['reload'].values())
    refs=c.read(c.OUT/'plans/references_task8.json')
    refs_flat=[dict(task=int(t),record=row) for t in sorted(refs,key=int) for row in refs[t]]
    primal=b['b']-b['A']@b['x']
    worst=[dict(normalized_index=int(i),reference_index=int(b['valid'][i]),primal=float(primal[i]),x=float(b['x'][i]),complementarity=float(abs(primal[i]*b['x'][i])),**refs_flat[int(b['valid'][i])]) for i in np.argsort(-primal)[:5]]
    write_json(D/'VIOLATING_CONSTRAINTS.json',dict(largest_primal_constraints=worst,formula='primal=b-A@x',formal_training_resumed=False))
    files={p.name:dict(path=str(p),bytes=p.stat().st_size,sha256=c.sha(p)) for p in sorted(D.iterdir()) if p.is_file()}
    report=dict(status='HISTORICAL_FAILURE_CAPTURED',true_python_batch_index=258,human_ordinal=259,pre_update_global_step=42057,pre_update_optimizer_step=3657,pre_update_scheduler_step=3657,post_failed_optimizer_step=3658,post_failed_scheduler_step=3658,failed_memory_is_not_a_resume_checkpoint=True,matched_prefix=list(range(186,258)),event_fields_exact_except_timings=True,qp_inputs_unchanged=matches,residuals=readings,source_formal_files_unchanged=unchanged,pre_failure_state=state,files=files,formal_training_resumed=False)
    report['largest_primal_constraint']=worst[0]
    report['fallback_termination_note']='L-BFGS-B stopped on relative objective decrease after one iteration; primal feasibility worsened. Both float64 and 80-digit checks reject at 1e-8.'
    write_json(OUT/'FINAL_REPORT.json',report)
    write_json(OUT/'DIAGNOSTIC_SOURCE_HASHES.json',{str(p.relative_to(c.ROOT)):c.sha(p) for p in sorted((c.ROOT/'qp_v3_failure_capture_v1').glob('*.py'))})
    text='''# New v3 QP failure capture\n\nFormal training remains stopped (`formal_training_resumed=false`). No solver settings, QP, constraint, threshold, or formal recovery slot were modified.\n\n- Capture smoke: PASS; same function objects and explicit diag_dir throughout step -> core -> solver.\n- Actual failure: task8 / epoch11 / Python batch258 (ordinal259), pre-update global step42057. Location comes from successful event rows, not periodic STATUS.\n- Parent: verified post-batch185 state, global41985. Restored model, AdamW, scheduler, modes and all RNG state.\n- Replayed batch186..257: 72 successful updates; every recorded non-timing field matches the historical event exactly.\n- Saved pre-batch258 state with immutable batch/reference identities and per-reference tensor fingerprints. Save/reload leaves state unchanged.\n- Failed batch calls native AdamW/scheduler once: counters3657 -> 3658. Its partially updated in-memory state is not resumable; use saved PRE_FAILURE_STATE.\n- before_fallback.npz and after_fallback.npz are atomic, loadable and contain identical K/f/A/b/scaling. Independent binary64 and 80-digit residuals are in FINAL_REPORT.json.\n\n'''
    text+='Before fallback KKT: %.17g\n\nAfter fallback KKT: %.17g\n\n'%(readings['before']['float64']['kkt'],readings['after']['float64']['kkt'])
    text+='The current L-BFGS-B fallback terminates but does not satisfy the unchanged 1e-8 KKT gate on this instance. This report captures the failure; it does not select a new solver or infer QP infeasibility. All subsequent QP work can use these arrays without prefix replay.\n\nState: `%s`\n'%state['path']
    text+='\nL-BFGS-B stopped after one iteration on relative objective decrease. Its success flag does not imply KKT acceptance. The violating component is primal feasibility, confirmed at 80 digits; no threshold decision flip.\n'
    (OUT/'FINAL_REPORT.md').write_text(text)
    print(json.dumps(dict(status=report['status'],residuals=readings,formal_training_resumed=False),indent=2,allow_nan=False))

if __name__=='__main__':main()
