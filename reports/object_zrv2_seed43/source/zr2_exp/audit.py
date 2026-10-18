"""Read-only scientific acceptance; process exit is checked separately."""
from .common import *
from compare_metrics import compare

def main():
    verify();cal=js(O/'CALIBRATED.json');assert cal['epochs']==40 and cal['optimizer_steps']==10550 and cal['exposures']==105240
    rows=js(O/'epochs.json');assert len(rows)==40
    assert {(r['task'],r['epoch']) for r in rows}=={(t,e) for t in range(6,10) for e in range(10)}
    assert sum(r['batches'] for r in rows)==10550 and sum(r['samples'] for r in rows)==105240
    logs=[json.loads(s) for s in (O/'calibration_batches.jsonl').read_text().splitlines()]
    # Successful fresh execution: any resumed duplicate physical computation must be audited separately.
    assert len(logs)==10550 and len({r['logical_step'] for r in logs})==10550
    counts={}
    for r in logs:
        assert len(r['samples'])==len(r['indices']);key=(r['stage'],r['epoch']);counts.setdefault(key,[]).extend(r['indices'])
    for (t,e),v in counts.items():assert sorted(v)==list(range([1562,2261,3081,3620][t-6]))
    for t in range(6,10):
        for task in range(6,t+1):
            expected=sum(task in r['active'] for r in logs if r['stage']==t)
            assert rows[(t-6)*10+9]['branch_steps'][str(task)]==expected
        assert js(O/f'buffer_after_task{t}.json')['hashes']==js(P/'coreset'/f'buffer_after_task{t}.json')['hashes']
    structure=[json.loads(s) for s in (O/'STRUCTURE.jsonl').read_text().splitlines()]
    assert len(structure)==16 and all(r['passed'] for r in structure)
    assert len({r['T0_hash'] for r in structure})==1
    attempts=[]
    for f in (O/'evaluation').glob('attempts*.jsonl'):attempts.extend(json.loads(s) for s in f.read_text().splitlines())
    for event,n in [('vector_attempt_start',20),('vector_attempt_complete',20),('environment_construct_start',10),('environment_construct_complete',10),('environment_closed',10)]:assert sum(r['event']==event for r in attempts)==n
    assert not any(r['event']=='technical_exception' for r in attempts)
    comparison=compare(P/'coreset/evaluation',O/'evaluation');write(O/'METRIC_COMPARISON.json',comparison)
    setup();p,t0=new_public();loaded=[]
    for t in range(6,10):
        f=O/'models'/f'zrv2_stage{t}_calibration_ep9.pth';s=torch.load(f,map_location='cpu');assert sha(f)==cal['checkpoints'][f.name]
        assert s['next_epoch']==10 and s['task']==t and s['contract_sha256']==sha(O/'CONTRACT.json')
        load_stage(p,t);assert tensor_hash(p)==s['public_hash'];assert tensor_hash(t0)==s['T0_hash']
        b,_=make_bank(t);b.load_state_dict(s['bank'],strict=True);assert b.gate_w.dtype==torch.float64
        loaded.append(dict(stage=t,checkpoint_sha256=sha(f),public_hash=s['public_hash'],T0_hash=s['T0_hash'],parameters=sum(v.numel() for v in b.parameters())))
    write(O/'CHECKPOINT_LOAD_ACCEPTANCE.json',dict(status='PASS',rows=loaded))
    write(O/'AUDIT.json',dict(status='PASS',epochs=40,optimizer_steps=10550,sample_exposures=105240,checkpoints=4,episodes=200,vector_attempts=20,structure_checks=16,metric_gate=comparison['metric_gate'],cost=dict(optimizer_steps=10550,sample_exposures=105240,epoch_seconds=sum(r['seconds'] for r in rows),extra_T0_bytes=js(O/'INTEGRATION.json')['T0_bytes'],residual_parameters=16608,peak_gpu_bytes=cal['gpu_peak_allocated'],latency=js(O/'COST_ISOLATION.json')['latency']),notes=['Three-seed/five-seed expansion is conditional on all metrics and process acceptance.','Historical RTX4090 parent vs current RTX3090; global action RNG not isolated.','Public policy was not retrained; extra calibration compute is not equal-budget comparison.']))
    status('AWAITING_PROCESS_EXIT_ACCEPTANCE',metric_gate=comparison['metric_gate'])
    print(json.dumps(dict(status='AUDIT_PASS',metric_gate=comparison['metric_gate'])),flush=True)
if __name__=='__main__':main()
