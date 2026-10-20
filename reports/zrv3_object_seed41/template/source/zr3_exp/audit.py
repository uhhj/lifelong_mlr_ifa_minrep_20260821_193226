"""Read-only scientific acceptance; process exit is checked separately."""
from .common import *
from branch_metrics import compare

def main():
    verify();cal=js(O/'CALIBRATED.json');assert cal['epochs']==40 and cal['optimizer_steps']==STEPS and cal['exposures']==EXPOSURES
    rows=js(O/'epochs.json');assert len(rows)==40
    assert {(r['task'],r['epoch']) for r in rows}=={(t,e) for t in range(6,10) for e in range(10)}
    assert sum(r['batches'] for r in rows)==STEPS and sum(r['samples'] for r in rows)==EXPOSURES
    logs=[json.loads(s) for s in (O/'calibration_batches.jsonl').read_text().splitlines()]
    # Successful fresh execution: any resumed duplicate physical computation must be audited separately.
    assert len(logs)==STEPS and len({r['logical_step'] for r in logs})==STEPS
    counts={}
    for r in logs:
        assert len(r['samples'])==len(r['indices']);key=(r['stage'],r['epoch']);counts.setdefault(key,[]).extend(r['indices'])
    for (t,e),v in counts.items():assert sorted(v)==list(range(LENS[t-6]))
    for t in range(6,10):
        for task in range(6,t+1):
            expected=sum(task in r['active'] for r in logs if r['stage']==t)
            assert rows[(t-6)*10+9]['branch_steps'][str(task)]==expected
        assert js(O/f'buffer_after_task{t}.json')['hashes']==js(P/PARM/f'buffer_after_task{t}.json')['hashes']
    structure=[json.loads(s) for s in (O/'STRUCTURE.jsonl').read_text().splitlines()]
    assert len(structure)==16 and all(r['passed'] for r in structure)
    assert all(r['unselected_means_exact'] and r['zero_residual_parent_exact'] and r['selected_component_exact'] for r in structure)
    if EPSILON is not None:
        assert max(r['actual_kl_max'] for r in logs+structure)<=EPSILON+1e-12
        assert all(js(f)['max_executed_kl']<=EPSILON+1e-12 for f in (O/'evaluation').glob('cell*.json'))
    assert len({r['T0_hash'] for r in structure})==1
    attempts=[]
    for f in (O/'evaluation').glob('attempts*.jsonl'):attempts.extend(json.loads(s) for s in f.read_text().splitlines())
    for event,n in [('vector_attempt_start',20),('vector_attempt_complete',20),('environment_construct_start',10),('environment_construct_complete',10),('environment_closed',10)]:assert sum(r['event']==event for r in attempts)==n
    assert not any(r['event']=='technical_exception' for r in attempts)
    comparison=compare(P/PARM/'evaluation',O/'evaluation');write(O/'METRIC_COMPARISON.json',comparison)
    setup();p,t0=new_public();loaded=[]
    for t in range(6,10):
        f=O/'models'/f'zrv3_stage{t}_calibration_ep9.pth';s=torch.load(f,map_location='cpu');assert sha(f)==cal['checkpoints'][f.name]
        assert s['next_epoch']==10 and s['task']==t and s['contract_sha256']==sha(O/'CONTRACT.json')
        load_stage(p,t);assert tensor_hash(p)==s['public_hash'];assert tensor_hash(t0)==s['T0_hash']
        b,_=make_bank(t);b.load_state_dict(s['bank'],strict=True);assert b.gate_w.dtype==torch.float64
        loaded.append(dict(stage=t,checkpoint_sha256=sha(f),public_hash=s['public_hash'],T0_hash=s['T0_hash'],parameters=sum(v.numel() for v in b.parameters())))
    write(O/'CHECKPOINT_LOAD_ACCEPTANCE.json',dict(status='PASS',rows=loaded))
    write(O/'AUDIT.json',dict(status='PASS',epochs=40,optimizer_steps=STEPS,sample_exposures=EXPOSURES,checkpoints=4,episodes=200,vector_attempts=20,structure_checks=16,metric_gate=comparison['metric_gate'],cost=dict(optimizer_steps=STEPS,sample_exposures=EXPOSURES,epoch_seconds=sum(r['seconds'] for r in rows),extra_T0_bytes=js(O/'INTEGRATION.json')['T0_bytes'],residual_parameters=16608,peak_gpu_bytes=cal['gpu_peak_allocated'],latency=js(O/'COST_ISOLATION.json')['latency']),notes=['Single seed gate: AUC improves; FWT and OldFinal do not decrease; engineering all PASS.','Historical hardware may differ; global action RNG not isolated; see parent binding.','Public policy was not retrained; extra calibration compute is not equal-budget comparison.']))
    status('AWAITING_PROCESS_EXIT_ACCEPTANCE',metric_gate=comparison['metric_gate'])
    print(json.dumps(dict(status='AUDIT_PASS',metric_gate=comparison['metric_gate'])),flush=True)
if __name__=='__main__':main()
