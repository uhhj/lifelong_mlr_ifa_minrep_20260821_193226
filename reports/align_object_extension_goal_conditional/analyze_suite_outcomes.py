"""Local-only decomposition of already accepted ALIGN matrices; no new evaluation."""
import datetime
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGES = {
    'OBJECT': [ROOT.parent/'20261022_caet_v3_tac_eftr_object41/final_ALIGN_seed41_20261022_220252',
               ROOT/'final_OBJECT_42_20261023_012252', ROOT/'final_OBJECT_43_20261023_032427'],
    'GOAL': [ROOT/'final_GOAL_41_20261023_052225', ROOT/'final_GOAL_42_20261023_072244',
             ROOT/'final_GOAL_43_20261023_092244'],
}

def metrics(r):
    return dict(AUC=sum(sum(r[s,t] for s in range(t,10))/F(10-t) for t in range(6,10))/4,
                FWT=sum(r[t,t] for t in range(6,10))/4,
                NBT=sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/F(9-t) for t in range(6,9))/3,
                OldFinal=sum(r[9,t] for t in range(6,9))/3,
                FinalAll=sum(r[9,t] for t in range(6,10))/4)

def number(x):
    return dict(fraction=str(x), percent=float(x*100))

def main():
    capture_path=ROOT.parent/'20261023_tacr_tacst_object_conditional/token_source_readonly_20261023_120646/EVIDENCE.json'
    capture=json.loads(capture_path.read_text(encoding='utf8'))
    shared={k:v['sha256'] for k,v in capture['files'].items() if k in (
        'libero/lifelong/models/brief.py','libero/lifelong/models/complete_policy.py',
        'libero/lifelong/models/modules/transformer_modules.py')}
    result=dict(status='LOCAL_ACCEPTED_MATRIX_DECOMPOSITION_PASS',
                observed_at=datetime.datetime.now().astimezone().isoformat(),
                scope='Existing accepted matrices only; no training, environment, remote writes or renewed acceptance',
                shared_token_source=shared,source_capture_sha256=hashlib.sha256(capture_path.read_bytes()).hexdigest(),
                suites={})
    for suite, paths in PACKAGES.items():
        runs=[]; provenance=[]
        for seed,p in zip([41,42,43],paths):
            f=p/'LOCAL_FINAL_AUDIT.json'; v=json.loads(f.read_text(encoding='utf8'))
            assert v['status']=='COMPLETED_LOCAL_INDEPENDENT_PASS' and v['seed']==seed and v['suite']==suite
            assert v['formal']['records']==200 and v['formal']['completed_cells']==10
            contract=json.loads((p/'current/CONTRACT.json').read_text())
            assert all(contract['source_sha256'][k]==sha for k,sha in shared.items())
            r={(z['stage'],z['task']):F(z['current']) for z in v['matrix']}
            b={(z['stage'],z['task']):F(z.get('parent',z.get('ra'))) for z in v['matrix']}
            for label,data,key in [('ALIGN',r,'metrics'),('RA',b,'baseline' if 'baseline' in v else 'ra')]:
                assert all(value==F(v[key][m]['fraction']) for m,value in metrics(data).items()),label
            runs.append((r,b))
            provenance.append(dict(seed=seed,path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                                   shared_token_source_exact=True,
                                   bc_valid=v['counts']['bc_exposures'],bc_invalid=v['counts']['invalid_bc_exposures'],
                                   delta_AUC=number(metrics(r)['AUC']-metrics(b)['AUC'])))
        mean=[{key:sum(run[i][key] for run in runs)/3 for key in runs[0][i]} for i in (0,1)]
        r,b=mean
        aggregate=json.loads((ROOT/('aggregate_OBJECT_20261023_0326' if suite=='OBJECT' else 'aggregate_GOAL_20261023_0925')/'LOCAL_AGGREGATE.json').read_text())
        for m,v in metrics(r).items():
            assert v==F(aggregate['metrics'][m]['ALIGN']['mean_fraction'])
            assert metrics(b)[m]==F(aggregate['metrics'][m]['RA']['mean_fraction'])
        cells=[dict(stage=s,task=t,ALIGN=number(r[s,t]),RA=number(b[s,t]),delta=number(r[s,t]-b[s,t]),
                    AUC_weight=str(F(1,4*(10-t))),AUC_delta_contribution=number((r[s,t]-b[s,t])/F(4*(10-t))))
               for s,t in r]
        tasks=[]
        names=json.loads((paths[-1]/'current/TASKS.json').read_text())
        for t in range(6,10):
            delta=sum(r[s,t]-b[s,t] for s in range(t,10))/F(10-t)
            tasks.append(dict(task=t,name=names[str(t)]['name'],trajectory_delta=number(delta),
                              AUC_delta_contribution=number(delta/4),
                              initial_ALIGN=number(r[t,t]),initial_RA=number(b[t,t]),
                              final_ALIGN=number(r[9,t]),final_RA=number(b[9,t]),
                              mean_initial_to_final_drop_ALIGN=number(r[t,t]-r[9,t]),
                              mean_initial_to_final_drop_RA=number(b[t,t]-b[9,t]),
                              final_per_seed=[dict(seed=s,ALIGN=number(x[0][9,t]),RA=number(x[1][9,t]))
                                              for s,x in zip([41,42,43],runs)]))
        delta=metrics(r)['AUC']-metrics(b)['AUC']
        assert sum(F(z['AUC_delta_contribution']['fraction']) for z in cells)==delta
        assert sum(F(z['AUC_delta_contribution']['fraction']) for z in tasks)==delta
        diag=sum((r[t,t]-b[t,t])/F(4*(10-t)) for t in range(6,10))
        successes=[sum(sum(v.values())*20 for v in (x[i] for x in runs)) for i in (0,1)]
        result['suites'][suite]=dict(provenance=provenance,metrics=aggregate['metrics'],cells=cells,tasks=tasks,
            AUC_diagonal_contribution=number(diag),AUC_post_learning_contribution=number(delta-diag),
            pooled_600_episode_successes=dict(ALIGN=str(successes[0]),RA=str(successes[1])),
            unweighted_10_cell_mean_delta=number(sum(r[k]-b[k] for k in r)/10),
            paired_cell_counts=dict(positive=sum(x[0][k]>x[1][k] for x in runs for k in r),
                                    equal=sum(x[0][k]==x[1][k] for x in runs for k in r),
                                    negative=sum(x[0][k]<x[1][k] for x in runs for k in r)))
        result['suites'][suite]['bc_masked_exposure_fraction']=number(F(sum(z['bc_invalid'] for z in provenance),sum(z['bc_invalid']+z['bc_valid'] for z in provenance)))
        pop,old=(3620,781+740+780) if suite=='OBJECT' else (3365,499+445+452)
        result['suites'][suite]['task9_population']=dict(total=pop,old=old,new=pop-old,old_fraction=number(F(old,pop)))
    out=ROOT/'analysis_OBJECT_GOAL_20261023'
    out.mkdir(exist_ok=True)
    (out/'EVIDENCE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    for suite,v in result['suites'].items():
        print(suite, 'successes',v['pooled_600_episode_successes'],'cells',v['paired_cell_counts'])
        print('AUC diagonal/postlearning',v['AUC_diagonal_contribution']['percent'],v['AUC_post_learning_contribution']['percent'])
        for x in v['tasks']:
            print(x['task'],x['name'],'AUC contribution',round(x['AUC_delta_contribution']['percent'],6),
                  'initial',x['initial_ALIGN']['percent'],x['initial_RA']['percent'],
                  'final',x['final_ALIGN']['percent'],x['final_RA']['percent'],
                  'seed final',[(z['ALIGN']['percent'],z['RA']['percent']) for z in x['final_per_seed']])
        print('cells',[(x['stage'],x['task'],round(x['ALIGN']['percent'],3),round(x['RA']['percent'],3)) for x in v['cells']])

if __name__=='__main__':
    main()
