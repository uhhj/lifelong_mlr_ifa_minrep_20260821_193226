"""New-seed primary confirmation, historical context, then descriptive pooling."""
import csv
import json
import math
import statistics
from array import array
from collections import defaultdict
from confirmation_experiment.common import ROOT,RUNS,REPORTS,read,write,sha,revision
from confirmation_experiment.protocol import SEEDS,epoch_phase,decide,DECISION_RULE
from fixed_kappa_experiment.metrics import read_fixed_matrix,score_matrix

def csvwrite(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (REPORTS/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def agg(xs):return {'n':len(xs),'mean':statistics.mean(xs),'sample_std':statistics.stdev(xs) if len(xs)>1 else None}
def quantiles(xs):
    if not xs:return {'n':0}
    ys=sorted(xs);n=len(ys)
    def q(p):
        k=(n-1)*p;i=int(k);return ys[i]+(ys[min(i+1,n-1)]-ys[i])*(k-i)
    return {'n':n,'mean':statistics.mean(ys),'min':ys[0],'p05':q(.05),'p25':q(.25),'median':q(.5),
            'p75':q(.75),'p95':q(.95),'max':ys[-1]}
SIGNALS=('z','cosine','cos_new_reference','mixed_grad_norm','reference_grad_norm','actual_correction_ratio')
def event_summary():
    groups={};epochs=defaultdict(lambda:[0,0])
    def get(k):
        if k not in groups:groups[k]={'batches':0,'valid_reference':0,'triggers':0,'valid_z':0,
                                   'ordinary_cancelled':0,'strong_z':0,
                                   'values':{s:array('d') for s in SIGNALS},'trigger_z':array('d')}
        return groups[k]
    for seed in SEEDS:
        run=RUNS/'formal'/f'seed{seed}'/'soft';expected_step=0;epoch_batch=defaultdict(int)
        with (run/'events.jsonl').open() as f:
            for line in f:
                r=json.loads(line);expected_step+=1
                assert r['seed']==seed and r['global_optimizer_step']==expected_step
                t,e=r['current_task'],r['epoch'];k=seed,t,e
                assert r['batch_index_in_epoch']==epoch_batch[k];epoch_batch[k]+=1
                assert r['kappa']==.5 and r['n_new']+r['n_old']==sum(r['task_source_counts'].values())
                assert sum(v for tid,v in r['task_source_counts'].items() if int(tid)<t)==r['n_old']
                assert r['projected']==bool(r['has_reference'] and not r['tiny_reference'] and r['dot_before']<0)
                if r['projected']:assert r['n_new']>0 and r['n_old']>0
                epochs[k][0]+=1;epochs[k][1]+=int(r['projected'])
                keys=[('global',),('seed',seed),('task',t),('phase',epoch_phase(e)),('seed_task_phase',seed,t,epoch_phase(e))]
                for key in keys:
                    g=get(key);g['batches']+=1;g['valid_reference']+=int(r['has_reference'] and not r['tiny_reference'])
                    g['triggers']+=int(r['projected'])
                    if r['z'] is not None:
                        g['valid_z']+=1;g['ordinary_cancelled']+=int(0<r['z']<=1);g['strong_z']+=int(r['z']>1)
                        if r['projected']:g['trigger_z'].append(r['z'])
                    for s in SIGNALS:
                        if r[s] is not None:g['values'][s].append(r[s])
        with (run/'run_001/fixed_kappa_epochs.csv').open() as f:
            for r in csv.DictReader(f):
                assert epochs[seed,int(r['stage']),int(r['epoch'])]==[int(r['batches']),int(r['projected_batches'])]
    result=[]
    for key,g in sorted(groups.items(),key=lambda z:repr(z[0])):
        result.append({'scope':list(key),'batches':g['batches'],'valid_reference':g['valid_reference'],
            'triggers':g['triggers'],'trigger_rate_valid_reference':g['triggers']/g['valid_reference'] if g['valid_reference'] else None,
            'valid_z':g['valid_z'],'ordinary_cancelled':g['ordinary_cancelled'],'strong_z':g['strong_z'],
            'ordinary_fraction_of_valid_z':g['ordinary_cancelled']/g['valid_z'] if g['valid_z'] else None,
            'strong_fraction_of_valid_z':g['strong_z']/g['valid_z'] if g['valid_z'] else None,
            'signals':{s:quantiles(v) for s,v in g['values'].items()},'trigger_z':quantiles(g['trigger_z'])})
    write(REPORTS/'event_summary.json',{'groups':result,'z_source':'algebraic same-forward scalars; no extra backward',
          'interpretation':'Raw-gradient event statistics do not establish actual AdamW or closed-loop causal harm.'})
    flat=[]
    for r in result:
        z={k:v for k,v in r.items() if k not in ('scope','signals','trigger_z')};z['scope']='/'.join(map(str,r['scope']))
        for k,v in r['signals'].items():z.update({k+'_'+q:x for q,x in v.items()})
        z.update({'trigger_z_'+q:x for q,x in r['trigger_z'].items()});flat.append(z)
    csvwrite('event_summary.csv',flat)
    return result

def run():
    metrics={};primary=[];per_task=[];paired=[]
    for name,h in read(REPORTS/'PREFLIGHT_COMPLETE.json')['historical_asset_sha256'].items():
        assert sha(ROOT/name)==h,name
    keys=('fwt','nbt','auc','final_mean_success')
    for seed in SEEDS:
        assert read(REPORTS/f'TASK6_PARITY_SEED{seed}.json')['status']=='PASS'
        assert read(REPORTS/f'SEED{seed}_COMPLETE.json')['status']=='PASS'
        for arm in ('mlr','soft'):
            run=RUNS/'formal'/f'seed{seed}'/arm
            train=read(run/'TRAINING_COMPLETE.json');assert train['status']=='PASS'
            assert read(run/'EVALUATION_COMPLETE.json')['native_evaluation_calls']==10
            for t,h in train['checkpoint_sha256'].items():assert sha(run/f'run_001/mlrifa_{t}_model_ep99.pth')==h
            matrix=read_fixed_matrix(run/'primary_ep99.log');m=score_matrix(matrix);metrics[seed,arm]=m
            primary.append(dict(seed=seed,method=arm,**{k:m[k] for k in keys},
                                worst_old_task_final_success=m['worst_old_final_success']))
            for j,t in enumerate((6,7,8,9)):
                values=[matrix[i][j] for i in range(j,4)]
                per_task.append({'seed':seed,'method':arm,'task':t,'first_success':matrix[j][j],
                    'final_success':matrix[-1][j],'column_auc':statistics.mean(values),
                    'peak_to_final_forgetting':max(values)-matrix[-1][j]})
        paired.append(dict(seed=seed,**{k:metrics[seed,'soft'][k]-metrics[seed,'mlr'][k] for k in keys}))
    old={t:[next(r['final_success'] for r in per_task if r['seed']==seed and r['task']==t and r['method']=='soft')-
             next(r['final_success'] for r in per_task if r['seed']==seed and r['task']==t and r['method']=='mlr') for seed in SEEDS] for t in (6,7,8)}
    decision=decide(paired,old)
    history=read(ROOT/'reports/fixed_kappa/fixed_v1/kappa_0p5/results.json')['runs']
    new_agg={arm:{k:agg([metrics[seed,arm][k] for seed in SEEDS]) for k in keys} for arm in ('mlr','soft')}
    hist_agg={arm:{k:agg([history[arm][str(seed)][k] for seed in (42,43,44)]) for k in keys} for arm in ('mlr','soft')}
    pooled={arm:{k:agg([metrics[seed,arm][k] for seed in SEEDS]+[history[arm][str(seed)][k] for seed in (42,43,44)]) for k in keys} for arm in ('mlr','soft')}
    hist_paired=[dict(seed=seed,**{k:history['soft'][str(seed)][k]-history['mlr'][str(seed)][k] for k in keys}) for seed in (42,43,44)]
    events=event_summary()
    result={'decision':decision,'decision_rule':DECISION_RULE,'primary_new_seeds':list(SEEDS),
            'primary_metrics':primary,'primary_paired':paired,'primary_aggregate':new_agg,
            'primary_matrices':{str(seed):{arm:metrics[seed,arm]['r'] for arm in ('mlr','soft')} for seed in SEEDS},
            'per_task':per_task,'old_task_final_deltas':old,'historical_context':hist_agg,
            'historical_paired':hist_paired,'exploratory_pooled':pooled,
            'implementation':read(REPORTS/'IMPLEMENTATION.json'),'preflight':read(REPORTS/'PREFLIGHT_COMPLETE.json'),
            'limitations':['Only three new seeds','Kappa .5 chosen using prior experiments','Historical use_state=False front end',
                           'Event statistics and algebraic z are not closed-loop causal explanations']}
    write(REPORTS/'RESULTS.json',result)
    csvwrite('primary_metrics.csv',primary);csvwrite('paired_deltas.csv',paired);csvwrite('per_task.csv',per_task)
    def fmt(x):return f"{x['mean']:.2f} +/- {x['sample_std']:.2f}"
    def md(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
    lines=['# Fixed-kappa confirmation v2',decision,
        '## Execution integrity',
        'Six fixed runs and six fixed-ep99 20-trial-per-cell success matrices completed. No best-epoch selection or performance stopping. The paired task6 gates passed before task7 for every seed.',
        'Implementation commit: '+result['implementation']['implementation_sha']+'. Full source hashes and precheck provenance are in IMPLEMENTATION.json and PREFLIGHT_COMPLETE.json.',
        'Input/config/cache hashes: FROZEN_INPUTS.json. Logging and snapshot parity: GPU_PARITY.json. Per-seed task6 checks: TASK6_PARITY_SEED*.json. Raw matrices/logs/events/snapshots remain in runs/fixed_kappa_confirmation_v2/.',
        '## Primary confirmation: new seeds 45/46/47',
        md(['method','FWT','NBT','AUC','final success'],[[arm]+[fmt(new_agg[arm][k]) for k in keys] for arm in ('mlr','soft')]),
        md(['seed','delta FWT','delta NBT','delta AUC','delta final'],[[r['seed']]+[f'{r[k]:+.4f}' for k in keys] for r in paired]),
        'All differences are soft minus MLR in percentage points. Lower NBT alone is not evidence of a better stability/plasticity tradeoff. The decision uses only new seeds and the frozen descriptive rule.',
        '## Historical replication context: 42/43/44',
        md(['method','FWT','NBT','AUC','final success'],[[arm]+[fmt(hist_agg[arm][k]) for k in keys] for arm in ('mlr','soft')]),
        md(['seed','delta FWT','delta NBT','delta AUC','delta final'],[[r['seed']]+[f'{r[k]:+.4f}' for k in keys] for r in hist_paired]),
        '## Exploratory pooled summary: six seeds',
        md(['method','FWT','NBT','AUC','final success'],[[arm]+[fmt(pooled[arm][k]) for k in keys] for arm in ('mlr','soft')]),
        'Descriptive context only; pooling cannot overturn an unsuccessful new-seed confirmation. No exaggerated significance claims are made.',
        '## Per-task localization',
        md(['seed','method','task','first','final','column AUC','peak-to-final'],[[r['seed'],r['method'],r['task']]+[f'{r[k]:.4f}' for k in ('first_success','final_success','column_auc','peak_to_final_forgetting')] for r in per_task]),
        'Worst-old-task final success is included in primary_metrics.csv. Full success matrices are in RESULTS.json.',
        '## Event distribution',
        md(['scope','batches','valid reference','triggers','trigger rate','ordinary cancelled','strong z'],
           [['/'.join(map(str,r['scope'])),r['batches'],r['valid_reference'],r['triggers'],
             f"{100*r['trigger_rate_valid_reference']:.4f}%" if r['trigger_rate_valid_reference'] is not None else '-',r['ordinary_cancelled'],r['strong_z']]
            for r in events if r['scope'][0] in ('global','seed','task','phase')]),
        'Continuous distributions, conditional denominators, z quantiles within triggers, cosine, gradient norms and correction ratios are in event_summary.json/CSV. Trigger rate is not the total new/old negative-gradient rate.',
        '## Limitations',
        'Only three new seeds. Kappa .5 was selected from earlier experiments, and adjacent kappas did not establish a stable positive region. The historical front end remains use_state=False. Algebraic event measurements, checkpoint snapshots and local BC diagnostics do not establish closed-loop causal protection. No counterfactual mechanism trials were run as part of this confirmation.',
        'The preregistered thresholds are descriptive research decisions rather than significance tests. All seed/task results are reported regardless of outcome.']
    (REPORTS/'FINAL_REPORT.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    write(REPORTS/'COMPLETION.json',{'status':'PASS','six_training_runs_complete':True,'six_evaluations_complete':True,
          'primary_seeds':list(SEEDS),'decision':decision,'git_head_at_summary':revision(),
          'results_sha256':sha(REPORTS/'RESULTS.json'),'event_summary_sha256':sha(REPORTS/'event_summary.json')})
if __name__=='__main__':run()
