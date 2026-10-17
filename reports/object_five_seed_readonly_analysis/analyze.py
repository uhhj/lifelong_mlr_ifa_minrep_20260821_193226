import json,statistics,hashlib
from pathlib import Path

OUT=Path(__file__).resolve().parent
source=OUT.parent/'object_five_seed_summary/RESULTS.json'
diagnostic=OUT/'EVENT_DIAGNOSTICS.json'
hashes={p.relative_to(OUT.parent).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,diagnostic)}
expected=json.loads((OUT/'INPUT_SHA256.json').read_text())
assert hashes == expected, 'Input hashes differ from published analysis'
d=json.loads(source.read_text()); ev=json.loads(diagnostic.read_text())
rows={(r['seed'],r['method']):r for r in d['first_runs']}
comparisons={}
for a,b in [('coreset','random'),('event_coreset','coreset')]:
    seeds=[]
    for s in range(40,45):
        ra,rb=rows[s,a],rows[s,b]
        delta={k:ra['metrics'][k]['percent']-rb['metrics'][k]['percent'] for k in ('AUC','FWT','NBT')}
        task_delta=[statistics.mean(ra['matrix'][i][j]-rb['matrix'][i][j] for i in range(j,4)) for j in range(4)]
        contribution=[x/4 for x in task_delta]
        assert abs(sum(contribution)-delta['AUC'])<1e-10
        cells=[{'stage':i+6,'task':j+6,'delta':ra['matrix'][i][j]-rb['matrix'][i][j]} for i in range(4) for j in range(i+1)]
        seeds.append({'seed':s,'metric_delta_pp':delta,'task_lifetime_delta_pp':task_delta,'auc_contribution_pp':contribution,'cell_deltas':cells})
    aucs=[r['metric_delta_pp']['AUC'] for r in seeds]
    comparisons[a+' minus '+b]={
        'per_seed':seeds,'mean_auc_delta':statistics.mean(aucs),'sd_auc_delta':statistics.stdev(aucs),
        'median_auc_delta':statistics.median(aucs),'excluding43_mean':statistics.mean(x for s,x in zip(range(40,45),aucs) if s!=43),
        'leave_one_out_means':{s:statistics.mean(x for t,x in zip(range(40,45),aucs) if t!=s) for s in range(40,45)},
        'task_lifetime_delta_mean_pp':[statistics.mean(r['task_lifetime_delta_pp'][j] for r in seeds) for j in range(4)],
        'auc_contribution_mean_pp':[statistics.mean(r['auc_contribution_pp'][j] for r in seeds) for j in range(4)],
        'cell_delta_mean':[{**{k:v for k,v in seeds[0]['cell_deltas'][i].items() if k!='delta'},'delta':statistics.mean(r['cell_deltas'][i]['delta'] for r in seeds)} for i in range(10)]}
admissions=[]
for s,rs in ev['seeds'].items():
    for r in rs:
        admissions.append(dict(seed=int(s),task=r['task'],replaced=r['q']-r['overlap'],replaced_percent=100*(r['q']-r['overlap'])/r['q'],
            parent_event_fraction=r['parent_event_count']/r['q'],event_fraction=r['actual_event_count']/r['q'],
            parent_global_radius=r['parent_global_radius'],event_global_radius=r['global_radius'],
            radius_relative_change_percent=100*(r['global_radius']/r['parent_global_radius']-1),
            event_radius=r['event_radius'],regular_radius=r['regular_radius'],
            ifa_pairs_equal=r['c_ifa']==r['ec_ifa'],c_ifa=r['c_ifa'],ec_ifa=r['ec_ifa'],confirmed_types=r['confirmed_event_types']))
assert all(hashlib.sha256((OUT.parent/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
out={'status':'PASS_READONLY','input_sha256':hashes,'comparisons':comparisons,'admissions':admissions}
(OUT/'ANALYSIS.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
for k,v in comparisons.items():
    print(k,json.dumps({x:y for x,y in v.items() if x!='per_seed'},indent=2))
print('ADMISSIONS',json.dumps(admissions,indent=2))
