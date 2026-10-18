"""Portable stdlib-only recomputation of frozen outcomes and derived diagnostics."""
import hashlib,json,statistics
from fractions import Fraction as F
from pathlib import Path
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=read(R/'EVIDENCE_MANIFEST.json')
inputs={str(R/'evidence'/n):v['sha256'] for n,v in manifest.items()}
for name in ('DIAGNOSTICS.json','JOINT_RNG.json'):inputs[str(R/name)]=sha(R/name)
assert all(sha(p)==v for p,v in inputs.items())
d=read(R/'DIAGNOSTICS.json');j=read(R/'JOINT_RNG.json')
assert d['status']==j['status']=='PASS_READONLY' and d['hashes_before_after_equal'] and j['hashes_before_after_equal']
assert all(v==0 for v in d['calls'].values())
result={'status':'PASS_READONLY','runs':{},'suite_summary':{},'diagnostic_summary':{},'matched_state_pairing_caution':'Outcome pairing only; not independently randomized trajectories or extra training seeds.'}
def pair(old,new):
    assert set(old)==set(new)
    table={k:0 for k in ('SS','SF','FS','FF')}
    for key in old:
        assert old[key]['initial_state_sha256']==new[key]['initial_state_sha256']
        table[('S' if old[key]['success'] else 'F')+('S' if new[key]['success'] else 'F')]+=1
    table['net_success_change']=table['FS']-table['SF']
    table['retention_fraction']=table['SS']/(table['SS']+table['SF']) if table['SS']+table['SF'] else None
    return table
for folder in sorted((R/'evidence').iterdir()):
    if folder.name.startswith('SOURCE_'):continue
    suite,seed,run=folder.name.split('_');seed=int(seed);arms={};records={}
    for arm in ('random','coreset'):
        rates={};rr={}
        for s in range(6,10):
            for t in range(6,s+1):
                p=folder/arm/'evaluation'/f'trials_stage{s}_task{t}.jsonl'
                xs=[json.loads(x) for x in p.read_text().splitlines()]
                assert len(xs)==20 and {x['trial'] for x in xs}==set(range(20))
                assert all((x['seed'],x['stage'],x['task'],x['epoch'],x['arm'])==(seed,s,t,99,arm) and type(x['success']) is bool for x in xs)
                rates[s,t]=F(sum(x['success'] for x in xs),20)
                rr[s,t]={x['initial_state_index']:x for x in xs}
        m={'AUC':sum(sum(rates[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,'FWT':sum(rates[t,t] for t in range(6,10))/4,'NBT':sum(sum(rates[t,t]-rates[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3}
        if (folder/'SUMMARY.json').exists():
            saved=read(folder/'SUMMARY.json')
            if 'arms' in saved:
                assert all(F(saved['arms'][arm]['metrics'][k]['fraction'])==v for k,v in m.items())
        arms[arm]={'metrics':{k:dict(fraction=str(v),percent=float(v*100)) for k,v in m.items()},
            'matrix':[[float(rates[s,t]*100) if t<=s else None for t in range(6,10)] for s in range(6,10)],
            'task6_relative_to_first':{str(s):pair(rr[6,6],rr[s,6]) for s in range(7,10)},
            'task6_adjacent_stage':{str(s):pair(rr[s-1,6],rr[s,6]) for s in range(7,10)},
            'task6_group_steps':{str(s):[rr[s,6][g*10]['vector_group_steps'] for g in (0,1)] for s in range(6,10)}}
        records[arm]=rr
    delta={k:float(100*(F(arms['coreset']['metrics'][k]['fraction'])-F(arms['random']['metrics'][k]['fraction']))) for k in ('AUC','FWT','NBT')}
    contrib={str(t):sum(arms['coreset']['matrix'][s-6][t-6]-arms['random']['matrix'][s-6][t-6] for s in range(t,10))/(10-t)/4 for t in range(6,10)}
    assert abs(sum(contrib.values())-delta['AUC'])<1e-10
    result['runs'][folder.name]=dict(suite=suite,seed=seed,run=run,arms=arms,delta_C_minus_R=delta,task_auc_contribution_pp=contrib,
        task6_between_methods={str(s):pair(records['random'][s,6],records['coreset'][s,6]) for s in range(6,10)})
for label,keys in [('GOAL_common',[f'GOAL_{s}_first' for s in (42,43,44)]),('OBJECT_common_first',[f'OBJECT_{s}_first' for s in (42,43,44)]),('OBJECT_five_first',[f'OBJECT_{s}_first' for s in range(40,45)]),('OBJECT_common_repeat43',['OBJECT_42_first','OBJECT_43_repeat1','OBJECT_44_first'])]:
    xs=[result['runs'][k] for k in keys]
    result['suite_summary'][label]={'run_keys':keys,'auc_delta_mean':statistics.mean(x['delta_C_minus_R']['AUC'] for x in xs),'auc_delta_sample_sd':statistics.stdev(x['delta_C_minus_R']['AUC'] for x in xs),
        'task_contribution_mean':{str(t):statistics.mean(x['task_auc_contribution_pp'][str(t)] for x in xs) for t in range(6,10)},
        'task6_mean_curve':{a:[statistics.mean(x['arms'][a]['matrix'][s][0] for x in xs) for s in range(4)] for a in ('random','coreset')}}
for suite in ('GOAL','OBJECT'):
    cells=[c for c in d['cells'] if c['suite']==suite];agg={}
    for method in ('random','coreset'):
        def mean(fn):return statistics.mean(fn(c) for c in cells)
        agg[method]={k:mean(lambda c:c['methods'][method]['profile'][k]) for k in ('demo_total_variation','quartile_total_variation','positive_gripper_fraction','descriptor_mean_bias_rms','padding_n','near_switch_n','events_both_sides')}
        agg[method]['combined_max']=mean(lambda c:c['methods'][method]['coverage']['combined']['max'])
        agg[method]['combined_mean']=mean(lambda c:c['methods'][method]['coverage']['combined']['mean'])
    agg['population_positive_fraction']=cells[0]['population']['positive_gripper_fraction']
    result['diagnostic_summary'][suite]=agg
for c in d['cells']:
    n=c['N'];K=c['K'];windows=c['window_bindings'];assert len(windows)==n and len({w['id'] for w in windows})==n
    for arm in ('random','coreset'):
        assert len(c['methods'][arm]['selected_ids'])==K
        assert set(w['id'] for w in windows if w['selected'][arm])==set(c['methods'][arm]['selected_ids'])
        prof=c['methods'][arm]['profile'];assert sum(prof['demo_counts'].values())==sum(prof['quartile_counts'].values())==K
        assert prof['unique_position_n']==K and c['unique_position_count']==n
        assert prof['events_both_sides']==sum(e['pre'] and e['post'] for e in prof['events'])
    assert set(c['methods']['random']['profile']['demo_counts'])==set(c['population']['demo_counts'])==set(c['methods']['coreset']['profile']['demo_counts'])
    assert len(c['population']['demo_counts'])==10
result['rng_comparison']=[]
for c in d['cells']:
    for stage in range(6,10):
        xs=[v for v in j['rng'] if v['suite']==c['suite'] and v['seed']==c['seed'] and v['stage']==stage]
        assert len(xs)==2
        result['rng_comparison'].append(dict(suite=c['suite'],seed=c['seed'],stage=stage,entry_all_rng_equal=xs[0]['tensor_or_state_hashes']==xs[1]['tensor_or_state_hashes']))
assert all(x['entry_all_rng_equal'] for x in result['rng_comparison'])
result['formal_records_read']=len(result['runs'])*400
result['input_hashes_unchanged']=all(sha(p)==v for p,v in inputs.items());assert result['input_hashes_unchanged']
(R/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'records':result['formal_records_read'],'suite_summary':result['suite_summary'],'diagnostics':result['diagnostic_summary']},indent=2))
