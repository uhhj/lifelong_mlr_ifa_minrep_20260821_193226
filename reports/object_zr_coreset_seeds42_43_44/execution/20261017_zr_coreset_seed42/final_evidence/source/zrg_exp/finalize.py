import json,time
import torch
from .common import *
from .model import cfg,data_setup,complete,load_stage
from .metrics import comparison
def main():
    verify();epochs=json.loads((ROOT/'epochs.json').read_text());assert len(epochs)==400 and epochs[-1]['steps']==105500
    assert {(r['task'],r['epoch']) for r in epochs}=={(t,e) for t in range(6,10) for e in range(100)}
    assert all(r['batches']=={6:157,7:227,8:309,9:362}[r['task']] for r in epochs)
    ev=ROOT/'evaluation';attempts=[json.loads(l) for f in ev.glob('attempts*.jsonl') for l in f.read_text().splitlines()]
    assert sum(r['event']=='vector_attempt_start' for r in attempts)==20
    assert sum(r['event']=='vector_attempt_complete' for r in attempts)==20
    assert not any(r['event']=='technical_exception' for r in attempts)
    isolation=[json.loads(l) for l in (O/'DIRECT_CHANNEL_ISOLATION.jsonl').read_text().splitlines()]
    assert all(r['passed'] for r in isolation)
    assert all(any(r['stage']==t and r['epoch']==99 and r['full_legal_old_buffer'] for r in isolation) for t in range(6,10))
    births=[json.loads(l) for l in (O/'BRANCH_BIRTH.jsonl').read_text().splitlines()];assert [b['task'] for b in births]==[6,7,8,9]
    ids=[json.loads(l) for l in (O/'FROZEN_BRANCH_IDENTITY.jsonl').read_text().splitlines()];assert len(ids)==4 and all(r['old_unchanged'] for r in ids)
    for t in range(6,10):
        mine=json.loads((ROOT/'storage'/f'task{t}.json').read_text());old=json.loads((P/'coreset/storage'/f'task{t}.json').read_text());assert mine['selected']==old['selected']
        assert json.loads((ROOT/f'buffer_after_task{t}.json').read_text())['hashes']==json.loads((P/'coreset'/f'buffer_after_task{t}.json').read_text())['hashes']
        assert json.loads((ROOT/f'ifa_task{t}.json').read_text())==json.loads((P/'coreset'/f'ifa_task{t}.json').read_text())
    assert hashes(ROOT/'buffer')==json.loads((ROOT/'buffer_after_task9.json').read_text())['hashes']
    cpchecks=[];c=cfg('eval');_,_,shape=data_setup(c,False);c.shape_meta=shape
    for t in range(6,10):
        path=ROOT/'models'/f'mlrifa_{t}_model_ep99.pth';p=complete(c,shape);n=load_stage(p,path,t)
        cpchecks.append(dict(stage=t,sha256=sha(path),expected_keys_loaded=n,branches=list(p.residual_bank),gate_dtypes={k:str(ch.gate_w.dtype) for k,ch in p.residual_bank.items()}));del p;torch.cuda.empty_cache()
    write(O/'CHECKPOINT_ACCEPTANCE.json',dict(status='PASS',checkpoints=cpchecks))
    metrics=comparison(P/'coreset/evaluation',ev);write(O/'METRIC_COMPARISON.json',metrics)
    cost=dict(training=json.loads((ROOT/'TRAINING_COMPLETE.json').read_text()),evaluation=json.loads((ev/'EVALUATION_COMPLETE.json').read_text()),integration=json.loads((O/'INTEGRATION.json').read_text())['seconds'],structural_diagnostic_seconds=sum(r['seconds'] for r in isolation),residual_fp32_parameters=18784,residual_parameter_bytes=75136,gate_fp64_bytes=16384,buffer_bytes=sum(p.stat().st_size for p in (ROOT/'buffer').glob('*.pt')),model_bytes=sum(p.stat().st_size for p in (ROOT/'models').glob('*.pth')))
    write(O/'COST.json',cost);verify()
    write(O/'FINAL_AUDIT.json',dict(status='PASS_AWAITING_PROCESS_EXIT',epochs=400,steps=105500,checkpoints=4,formal_episodes=200,vector_attempts=20,inputs_unchanged=True,buffer_members_order_payloads_equal=True,IFA_identity_equal=True,structural_checks_pass=True,optimizer_integration_calls=4,automatic_extension=False))
    report=['# Seed42 Coreset + IFA + zero-response residual','', 'Status: scientific/engineering audits passed; process exit must be verified separately.','', '| Stage | Task6 | Task7 | Task8 | Task9 |','|---|---:|---:|---:|---:|']
    for s in range(6,10):report.append('| '+str(s)+' | '+' | '.join(str(metrics['new_counts'][f'{s},{t}']*5)+'%' if t<=s else '—' for t in range(6,10))+' |')
    report+=['','| Metric | Coreset baseline % | New % | Difference pp |','|---|---:|---:|---:|']
    for k in metrics['new']:report.append(f"| {k} | {metrics['baseline'][k]['percent']:.8f} | {metrics['new'][k]['percent']:.8f} | {metrics['delta_pp'][k]:+.8f} |")
    report+=['','Prespecified progression gate: '+str(metrics['progression_gate']), '', 'Limitations: one exploratory seed; unmatched total parameter count; original global action RNG is not isolated per trajectory; pretrained state metadata unknown; parent schedule engineering completion retained; shared network can forget; zero direct residual response does not imply whole-policy invariance. No automatic extension.','', 'Counts: 400 epochs, 105500 optimizer steps, 4 final checkpoints, 200 formal episodes, 20 vector attempts. Integration: 4 optimizer calls, zero environment steps. Baseline was not retrained or re-evaluated.']
    (O/'FINAL_REPORT.md').write_text('\n'.join(report)+'\n');write(O/'RUN_STATUS.json',dict(status='AWAITING_PROCESS_EXIT',updated_unix=time.time()))
if __name__=='__main__':main()
