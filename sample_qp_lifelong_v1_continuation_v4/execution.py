"""Execution-only revision: scoped integrity and one epoch recovery observation."""
from .common import ROOT, OUT, RUN, read, write, sha

def verify_execution(protocol):
    execution=read(OUT/'EXECUTION_FROZEN.json')
    assert sha(OUT/'PROTOCOL_FROZEN.json')==execution['scientific_protocol_sha256']
    assert sha(OUT/'PREFLIGHT_COMPLETE.json')==execution['preflight_sha256']
    for name,digest in execution['files'].items():
        assert sha(ROOT/name)==digest, ('Execution source changed',name)
    # Cache windows and references retain their native hash-on-load checks.
    # Do not hash thousands of old, unused experiment assets at every entry.
    return protocol

def recovery_records(records):
    by_batch={r['batch']:r for r in records}
    assert len(records)==309 and set(by_batch)==set(range(309))
    a,b=by_batch[185],by_batch[258]
    assert a['fallback_triggered'] and not a['second_fallback_triggered']
    assert b['fallback_triggered'] and b['second_fallback_triggered']
    for r in (a,b):
        assert r['kkt']<=1e-8 and r['projection_residual']<=5e-6
        assert r['optimizer_step']==3399+r['batch']+1
        assert r['scheduler_step']==r['optimizer_step']
        assert r['global_step']==41799+r['batch']+1
    assert by_batch[308]['global_step']==42108
    return dict(B185=a,B258=b,after_B258_successful_batches=50)

def observe_epoch11(task,epoch,state_path,event_path,algo):
    if (task,epoch)!=(8,11):return
    from functional_geometry_diagnostic_v1 import geometry as g
    import json
    saved=g.torch.load(state_path,map_location='cpu')
    assert saved['task']==8 and saved['next_epoch']==12 and saved['global_step']==42108
    checks=g.compare_states(saved['state'],g.capture(algo))
    assert all(checks.values()),checks
    with open(event_path) as f:records=[json.loads(line) for line in f if line.strip()]
    details=recovery_records(records)
    write(OUT/'V4_NUMERICAL_RECOVERY_PASS.json',dict(
        status='V4_NUMERICAL_RECOVERY_PASS',formal_training_resumed=True,
        canonical_checkpoint=str(state_path),checkpoint_sha256=sha(state_path),
        task=8,completed_epoch=11,next_epoch=12,global_step=42108,
        reload_checks=checks,events_sha256=sha(event_path),observations=details,
        remaining_action='Continue task8/task9 and frozen lifecycle evaluation without tuning'))
    print('V4_NUMERICAL_RECOVERY_PASS',42108,flush=True)
