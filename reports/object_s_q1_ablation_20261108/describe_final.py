"""Accepted-results interpretation inputs; no training or environment execution."""
from pathlib import Path
import csv, hashlib, json, statistics, sys
R=Path(__file__).resolve().parent;D=Path(sys.argv[1])
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=read(D/'LOCAL_FINAL_AUDIT.json');comp=read(D/'PRIMARY_S_COMPARISON.json');e=read(D/'EVIDENCE.json')
assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE'
assert comp['source_Variant_final_sha256']==sha(D/'LOCAL_FINAL_AUDIT.json')
binding=comp['source_S']['Q1'];base=read(Path(binding['final_path'])/'EVIDENCE.json')
assert sha(Path(binding['final_path'])/'EVIDENCE.json')==binding['evidence_sha256']
keys=['AUC','FWT','NBT','OldFinal','NewFinal','FinalAll'];rows=[];cases={};allpairs=[]
for seed in (41,42,43):
    case=f'Q1_{seed}';c=comp['cases'][case];pair=c['paired_outcomes'];allpairs+=pair
    methods={}
    for label,metric_key,data,prefix in [('Fixed q=1','Variant_metrics',e,case+'/'),('Full S','S_metrics',base,f'seed{seed}/')]:
        get=lambda name:json.loads(data['light'][prefix+name]['text'])
        rec=get('OUTCOMES.json');m=dict(c[metric_key]);m['NewFinal']=5*sum(x['success'] for x in rec if x['stage']==x['task']==9)
        rows.append(dict(seed=seed,method=label,**m))
        gate=[]
        for name,item in data['light'].items():
            if name.startswith(prefix) and (name==prefix+'GATES.jsonl' or '/GATES_stage' in name):
                gate.extend(json.loads(line) for line in item['text'].splitlines())
        assert len(gate)==9100 and len({(g['stage'],g['epoch'],g['batch']) for g in gate})==9100
        gg=get('GRADIENTS_S.json');stage={}
        for t in (7,8,9):
            gs=[x for x in gate if x['stage']==t];grad=[x for x in gg if x['stage']==t]
            stage[str(t)]=dict(blocks=len(gs),Q_accepts=sum(x['accepted']=='Q' for x in gs),
                Q_accept_percent=100*sum(x['accepted']=='Q' for x in gs)/len(gs),
                gradient_proxies={k:statistics.mean(x['norms'][k] for x in grad) for k in ['bc_new','exec_new','retain']},
                retention_loss_proxy=statistics.mean(x['retain_weighted'] for x in grad),
                retain_bc_cosine_mean=statistics.mean(x['retain_vs_bc']['cosine'] for x in grad if x['retain_vs_bc']['cosine'] is not None),
                sampled_epochs=[x['epoch'] for x in grad])
        methods[label]=dict(metrics=m,collection_successes=[get(f'MEMORY_S_{t}.json')['collection_successes'] for t in range(6,10)],
            stage_proxies=stage,total_Q_accepts=sum(x['accepted']=='Q' for x in gate))
    def discord(p):
        return dict(SF=sum(x['S'] and not x['Variant'] for x in p),FS=sum(x['Variant'] and not x['S'] for x in p),
            both_success=sum(x['S'] and x['Variant'] for x in p),both_fail=sum(not x['S'] and not x['Variant'] for x in p),episodes=len(p))
    diffs=[x['S_minus_Variant_pp'] for x in c['cells'].values()]
    cases[case]=dict(methods=methods,paired_all=discord(pair),paired_final=discord([x for x in pair if x['stage']==9]),
        S_cell_wins=sum(x>0 for x in diffs),cell_ties=sum(x==0 for x in diffs),S_cell_losses=sum(x<0 for x in diffs),
        cells=c['cells'],hardware=c['hardware'])
means={label:{k:statistics.mean(row[k] for row in rows if row['method']==label) for k in keys} for label in ['Fixed q=1','Full S']}
for label in means:rows.append(dict(seed='mean',method=label,**means[label]))
with (D/'COMPLETE_METRICS.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['seed','method']+keys);w.writeheader();w.writerows(rows)
task_final={}
for task in (6,7,8,9):
    task_final[str(task)]={label:statistics.mean(cases[f'Q1_{seed}']['cells'][f'9_{task}'][col] for seed in (41,42,43)) for label,col in [('Fixed q=1','Variant_percent'),('Full S','S_percent')]}
out=dict(status='PASS_ACCEPTED_RESULT_INTERPRETATION_RECOMPUTATION',means=means,
    S_minus_fixed_q1_pp={k:means['Full S'][k]-means['Fixed q=1'][k] for k in keys},cases=cases,
    paired_all=discord(allpairs),paired_final=discord([x for x in allpairs if x['stage']==9]),
    cell_counts={k:sum(c[k] for c in cases.values()) for k in ['S_cell_wins','cell_ties','S_cell_losses']},
    final_task_means=task_final,all_S_negative_cells=comp['all_negative_cells'],
    sources={'final_audit_sha256':sha(D/'LOCAL_FINAL_AUDIT.json'),'comparison_sha256':sha(D/'PRIMARY_S_COMPARISON.json')},
    definitions={'SF':'Full S succeeds and fixed-q1 fails','FS':'Full S fails and fixed-q1 succeeds','delta':'Full S minus fixed-q1; NBT lower is better'},
    limits=['Descriptive paired development-seed evidence, not an independent confirmatory test.',
        '600 episodes contain repeated tasks/stages and only three training seeds; do not treat all as independent replications.',
        'Hardware differs and Task6 evaluation can differ despite equal model tensors; caches are generated separately.',
        'Gate acceptance and gradient/cost diagnostics are training proxies, not closed-loop causal effects.',
        'Lower NBT can reflect lower just-learned starting success; inspect FWT and final old-task success jointly.'])
(D/'INTERPRETATION_AUDIT.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print(json.dumps({k:out[k] for k in ['means','S_minus_fixed_q1_pp','paired_all','paired_final','cell_counts','final_task_means']},indent=2))
