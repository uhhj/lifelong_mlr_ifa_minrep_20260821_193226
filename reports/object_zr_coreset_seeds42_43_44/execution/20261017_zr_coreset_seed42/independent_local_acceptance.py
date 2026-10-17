"""Read-only experiment verification; no torch, environment, or training imports."""
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent
E = ROOT / 'final_evidence'
A = ROOT / 'final_acceptance'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def lines(p):
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def metrics(c):
    task_averages = [sum((F(c[s, t], 20) for s in range(t, 10)), F()) / (10-t) for t in range(6, 10)]
    return {
        'AUC': sum(task_averages, F()) / 4,
        'FWT': F(sum(c[t, t] for t in range(6, 10)), 80),
        'NBT': sum((sum((F(c[t, t]-c[s, t], 20) for s in range(t+1, 10)), F()) / (9-t) for t in range(6, 9)), F()) / 3,
        'OldFinal': F(sum(c[9, t] for t in range(6, 9)), 60),
        'FinalAll': F(sum(c[9, t] for t in range(6, 10)), 80),
    }

def trials(folder):
    result = {}
    counts = {}
    for s in range(6, 10):
        for t in range(6, s+1):
            rows = lines(folder / f'trials_stage{s}_task{t}.jsonl')
            assert len(rows) == 20
            assert {r['initial_state_index'] for r in rows} == set(range(20))
            for r in rows:
                assert (r['seed'], r['stage'], r['task'], r['epoch']) == (42,s,t,99)
                assert type(r['success']) is bool
                result[s,t,r['initial_state_index']] = r
            counts[s,t] = sum(r['success'] for r in rows)
    assert len(result) == 200
    return result, counts

def main():
    manifest = read(ROOT / 'light_evidence_1745_manifest.json')
    assert all(sha(E/p) == v['sha256'] for p,v in manifest.items())
    assert sha(ROOT/'final_acceptance_1746.tar.gz') == 'ae2517ab8ff1932be391a50112a4483cc9239a569d7c6279ef3f4ba2ae4603b3'
    run = E/'run/coreset_ifa_zr'
    old, oc = trials(E/'baseline')
    new, nc = trials(run/'evaluation')
    assert oc == {(6,6):6,(7,6):14,(7,7):17,(8,6):4,(8,7):19,(8,8):20,(9,6):6,(9,7):18,(9,8):20,(9,9):9}
    for k in new:
        for field in ('initial_state_sha256','modal_sha256'):
            assert new[k][field] == old[k][field]
    epochs = read(run/'epochs.json')
    assert len(epochs) == 400
    assert {(r['task'],r['epoch']) for r in epochs} == {(t,e) for t in range(6,10) for e in range(100)}
    cumulative = 0
    for r in epochs:
        assert r['batches'] == {6:157,7:227,8:309,9:362}[r['task']]
        cumulative += r['batches']
        assert r['steps'] == cumulative
    assert cumulative == 105500
    for s,t in nc:
        aa = lines(run/'evaluation'/f'attempts_stage{s}_task{t}.jsonl')
        assert not any(r['event']=='technical_exception' for r in aa)
        starts = [r for r in aa if r['event']=='vector_attempt_start']
        ends = [r for r in aa if r['event']=='vector_attempt_complete']
        assert len(starts)==len(ends)==2
        for group in range(2):
            start = next(r for r in starts if r['group']==group)
            end = next(r for r in ends if r['group']==group)
            assert start['env_seed']==end['env_seed']==42+group
            assert start['initial_indices']==list(range(10*group,10*group+10))
            assert start['unix'] <= end['unix'] and end['steps']<=600
            for lane,value in enumerate(end['successes']):
                assert value == new[s,t,10*group+lane]['success']
    checks = lines(E/'run/DIRECT_CHANNEL_ISOLATION.jsonl')
    assert len(checks)==44 and all(c['passed'] for c in checks)
    sweeps = {}
    for t in range(6,10):
        cc = [c for c in checks if c['stage']==t]
        assert {c['epoch'] for c in cc}=={-1,9,19,29,39,49,59,69,79,89,99}
        final = next(c for c in cc if c['epoch']==99)
        assert final['full_legal_old_buffer']
        sweeps[t] = sum(x['n'] for x in final['rows'] if x['kind']=='legal_buffer')
    assert sweeps == {6:0,7:781,8:1521,9:2301}
    allrows = [r for c in checks for r in c['rows']]
    assert all(r['passed'] and r['selected_exact'] for r in allrows)
    assert max(r['old_gate'] for r in allrows)<=1e-6
    assert max(r['contribution'] for r in allrows)<=1e-6
    assert all(r['old_unchanged'] for r in lines(E/'run/FROZEN_BRANCH_IDENTITY.jsonl'))
    cps = read(A/'run/CHECKPOINT_ACCEPTANCE.json')
    assert cps['status']=='PASS' and len(cps['checkpoints'])==4
    for cp in cps['checkpoints']:
        s=cp['stage']; assert cp['branches']==[str(t) for t in range(6,s+1)]
        assert all(v=='torch.float64' for v in cp['gate_dtypes'].values())
        assert all(r['checkpoint_sha256']==cp['sha256'] for k,r in new.items() if k[0]==s)
    assert read(A/'run/FINAL_AUDIT.json')['inputs_unchanged']
    proc=read(A/'run/PROCESS_EXIT_ACCEPTANCE.json')
    assert proc['status']=='PASS' and proc['remaining']==[] and proc['original_pipeline_exit']==1 and proc['recovery_exit']==0
    assert (A/'engineering_recovery/resume.exit').read_text().strip()=='0'
    assert (A/'engineering_recovery/supervisor.exit').read_text().strip()=='0'
    assert read(A/'run/RUN_STATUS.json')['status']=='COMPLETED'
    cm=read(A/'run/METRIC_COMPARISON.json'); bm=metrics(oc); nm=metrics(nc)
    for key in nm:
        assert F(cm['baseline'][key]['fraction'])==bm[key]
        assert F(cm['new'][key]['fraction'])==nm[key]
    result={'status':'PASS_LOCAL_INDEPENDENT_ACCEPTANCE','epochs':400,'optimizer_steps':cumulative,'new_episodes':200,'baseline_records_readonly':200,'vector_attempts':20,'structural_checks':44,'final_old_buffer_sweeps':sweeps,'baseline':{k:str(v) for k,v in bm.items()},'new':{k:str(v) for k,v in nm.items()},'delta_pp':{k:float(100*(nm[k]-bm[k])) for k in nm},'new_counts':{f'{s},{t}':v for (s,t),v in nc.items()},'baseline_counts':{f'{s},{t}':v for (s,t),v in oc.items()},'remote_checkpoint_load_evidence':'4 strict load checks PASS; no weights downloaded','original_pipeline_exit':1,'recovery_exit':0,'auto_extend':False}
    (ROOT/'LOCAL_FINAL_ACCEPTANCE.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
