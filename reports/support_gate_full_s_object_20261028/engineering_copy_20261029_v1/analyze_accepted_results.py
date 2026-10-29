"""Recompute interpretation tables from accepted records only; no model calls."""
from pathlib import Path
import hashlib, json, statistics, sys
D=Path(sys.argv[1]); N=Path(__file__).resolve().parent.parent
js=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=js(D/'LOCAL_FINAL_AUDIT.json'); v=js(D/'EVIDENCE.json'); summary=js(D/'RESULT_SUMMARY.json')
assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE'
L=v['light'];get=lambda n:json.loads(L[n]['text'])
oldpath=N.parent/'20261026_ccbp_v4_s_object41_42_43/final_readonly_20261027_120754/LOCAL_FINAL_AUDIT.json'
old=js(oldpath)
pairs=[]; training={}; historical={}
for seed,r in a['seeds'].items():
    outcomes=get('seed'+seed+'/OUTCOMES.json')
    for task in range(6,9):
        before={x['initial_index']:x for x in outcomes if x['stage']==8 and x['task']==task}
        after={x['initial_index']:x for x in outcomes if x['stage']==9 and x['task']==task}
        assert set(before)==set(after)==set(range(20))
        assert all(all(before[i][k]==after[i][k] for k in ['initial_sha256','noise_sha256','env_seed','initial_file_sha256']) for i in before)
        pairs.append(dict(seed=seed,task=task,stage8_successes=sum(x['success'] for x in before.values()),stage9_successes=sum(x['success'] for x in after.values()),SF=sum(before[i]['success'] and not after[i]['success'] for i in before),FS=sum(not before[i]['success'] and after[i]['success'] for i in before)))
    epochs=get('seed'+seed+'/EPOCHS_S.json'); gradients=r['gradient_records']
    training[seed]={}
    for stage in range(7,10):
        rows=[x for x in epochs if x['stage']==stage];gg=[x for x in gradients if x['stage']==stage]
        assert len(rows)==100 and len(gg)==4
        training[seed][str(stage)]=dict(first_epoch_probe=rows[0]['training_probe'],last_epoch_probe=rows[-1]['training_probe'],gradient_samples=4,retain_dot_bc_negative=sum(x['retain_vs_bc']['dot']<0 for x in gg),retain_dot_exec_negative=sum(x['retain_vs_exec']['dot']<0 for x in gg),retain_vs_bc_cosines=[x['retain_vs_bc']['cosine'] for x in gg],retain_vs_exec_cosines=[x['retain_vs_exec']['cosine'] for x in gg],gate=r['gate_audit']['by_stage'][str(stage)])
    baseline=old['seeds'][seed]
    historical[seed]=dict(delta_metrics_pp={k:r['metrics'][k]['percent']-baseline['metrics'][k]['percent'] for k in ['AUC','FWT','NBT','OldFinal','FinalAll']},delta_cells_pp={k:x['percent']-baseline['matrix'][k]['percent'] for k,x in r['matrix'].items()})
means={k:statistics.mean(historical[s]['delta_metrics_pp'][k] for s in historical) for k in ['AUC','FWT','NBT','OldFinal','FinalAll']}
events={s:[json.loads(line) for line in L['seed'+s+'/EVAL_EVENTS.jsonl']['text'].splitlines()] for s in a['seeds']}
out=dict(status='ACCEPTED_RECORDS_INTERPRETATION_RECOMPUTED',final_audit_sha256=sha(D/'LOCAL_FINAL_AUDIT.json'),historical_audit_path=str(oldpath),historical_audit_sha256=sha(oldpath),stage8_to_stage9_pairs=pairs,training_diagnostics=training,historical_original_S_comparison=historical,historical_original_S_mean_delta_pp=means,evaluation_seconds={s:sum(x['seconds'] for x in ev if x['kind']=='complete') for s,ev in events.items()},limits=['Historical original S differs in training method, actual caches and compute; descriptive differences are not controlled causal effects or statistical significance.', 'Negative gradient dot products are sampled local conflicts, not causal explanations of episode failures.', 'Training probe BC, execution costs and gate acceptance do not establish closed-loop retention.', 'No extra training, model forward, environment, retry or selection was performed.'])
(D/'INTERPRETATION_AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(pairs=pairs,historical_mean_delta_pp=means,negative_cells={s:{k:x for k,x in r['delta_cells_pp'].items() if x<0} for s,r in historical.items()},gradients={s:{t:{k:z[k] for k in ['retain_dot_bc_negative','retain_dot_exec_negative']} for t,z in x.items()} for s,x in training.items()}),indent=2))
