"""Local scalar/code analysis only: no model load, forward, training or environment."""
import json, hashlib, datetime
from pathlib import Path
from fractions import Fraction as F

R = Path(__file__).resolve().parent
S = R / 'snapshot_20261024_132735'
B = S / 'metadata/B0'
A = R.parent / '20261022_caet_v3_tac_eftr_object41/final_ALIGN_seed41_20261022_220252/LOCAL_FINAL_AUDIT.json'
sources = {}
def read(p):
    raw = p.read_bytes()
    sources[str(p.relative_to(R)) if p.is_relative_to(R) else str(p)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)

result = read(R / 'B0_COMPLETE_RESULT.json')
assert result['episodes'] == 200 and result['cells'] == 10
download = read(S / 'DOWNLOAD.json')
old = read(A)
matrix = {(r['stage'], r['task']): F(r['successes'],20) for r in result['matrix']}
refs = {name:{(r['stage'],r['task']):F(r[key]) for r in old['matrix']} for name,key in [('RA','ra'),('ALIGN','current')]}
contributions = []
for t in range(6,10):
    avg = sum(matrix[s,t] for s in range(t,10))/(10-t)
    row = dict(task=t, B0_lifetime_mean_percent=float(avg*100))
    for name,m in refs.items():
        ref = sum(m[s,t] for s in range(t,10))/(10-t)
        row[name+'_lifetime_mean_percent'] = float(ref*100)
        row['AUC_contribution_vs_'+name+'_pp'] = float((avg-ref)*25)
    contributions.append(row)
for name in refs:
    assert abs(sum(r['AUC_contribution_vs_'+name+'_pp'] for r in contributions)-result['differences_pp'][name]['AUC']) < 1e-10

diagnostics=[];training=[];masks=[];old6=[]
for stage in range(6,10):
    values={}
    for kind in ('diagnostics','epochs','dataset'):
        p=B/kind/f'task{stage}.json';values[kind]=read(p)
        rel=p.relative_to(S/'metadata').as_posix()
        assert hashlib.sha256(p.read_bytes()).hexdigest()==download[rel]['sha256']==download[rel]['at_snapshot']['sha256']
    ds=values['dataset']['rows'];di=values['diagnostics'];ep=values['epochs']
    assert len(ep)==100 and di['optimizer_steps']==di['environment_steps']==0
    assert len(ds)==len(di['rows'])
    for a,b in zip(ds,di['rows']):
        assert (a['task'],a['sample'],a['demo'],a['u'],a['sha256'])==(b['task'],b['sample'],b['demo'],b['u'],b['sample_sha256'])
        assert sum(a['action_mask'])==b['metrics']['predicted_action']['count']
    counts=[sum(r['action_mask'][i] for r in ds) for i in range(4)]
    assert all(e['action_valid']==sum(counts) and e['action_invalid']==len(ds)*4-sum(counts) for e in ep)
    masks.append(dict(stage=stage,valid_by_horizon=counts,first_share_percent=100*counts[0]/sum(counts)))
    training.append(dict(stage=stage,first_epoch_mean_bc=ep[0]['bc']/ep[0]['batches'],last_epoch_mean_bc=ep[-1]['bc']/ep[-1]['batches'],first_epoch_mean_ifa=ep[0]['ifa']/ep[0]['batches'],last_epoch_mean_ifa=ep[-1]['ifa']/ep[-1]['batches'],updates=sum(e['batches'] for e in ep)))
    for task in range(6,stage+1):
        rows=[r for r in di['rows'] if r['task']==task]
        count=sum(r['metrics']['predicted_action']['count'] for r in rows)
        mean=sum(r['metrics']['predicted_action']['sum'] for r in rows)/count
        diagnostics.append(dict(stage=stage,task=task,windows=len(rows),valid_horizon_positions=count,grouped_huber=mean,closed_loop_percent=float(matrix[stage,task]*100)))
    if stage>=7:
        old6.append({(r['sample'],r['demo'],r['u'],r['sha256']):r['action_mask'] for r in ds if r['task']==6})
assert old6[0]==old6[1]==old6[2]
assert sum(r['updates'] for r in training)==105500
for name in ('model.py','readout.py','train.py','diagnose.py','evaluate.py'):
    p=R/'caet_em_exp'/name;sources[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest()
out=dict(status='PASS_LOCAL_B0_DESCRIPTIVE_ANALYSIS_NOT_CAUSAL_PROOF',time=datetime.datetime.now().astimezone().isoformat(),snapshot=S.name,task_AUC_decomposition=contributions,offline_diagnostics=diagnostics,training=training,mask_counts=masks,task6_old_RA_identical_stages7_8_9=True,source_sha256=sources,extra_model_forwards=0,extra_optimizer_steps=0,extra_environment_steps=0,limitations=['Single seed, 20 episodes per cell; exploratory and repeatedly inspected benchmarks.', 'Offline errors use training demos and old RA, all four horizons, selected-component means; not held-out first-action error or sampled action risk.', 'Readout, random adapter and four-step objective changed together relative to ALIGN; no single-factor causal attribution.', 'Complete B0 records and CPU preflight available; whole-queue final acceptance and independent initial-state file-byte audit pending.'])
(R/'B0_FAILURE_ANALYSIS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:out[k] for k in ('task_AUC_decomposition','offline_diagnostics','training','mask_counts')},ensure_ascii=False,indent=2))
