"""Recompute the lambda sensitivity table from accepted paired outcomes."""
from pathlib import Path
from fractions import Fraction
import csv
import hashlib
import json
import statistics

R=Path(__file__).resolve().parent
E=R.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,s):p.write_bytes(s.rstrip().encode('utf8')+b'\n')
cross=E/'20261103_lambda_r_goal_sensitivity/cross_suite_comparison_20261104/COMPARISON.json'
summary=read(cross)
assert summary['status']=='PASS_RECOMPUTED_FROM_BOTH_ACCEPTED_SUITES'
roots={'OBJECT':E/'20261101_lambda_r_object_sensitivity',
       'GOAL':E/'20261103_lambda_r_goal_sensitivity'}
style=E/'20261107_main_comparison_table/main_comparison.tex'
sources=[cross,style]
seeds=[];rows=[]
for suite,root in roots.items():
    acceptance=root/'ACCEPTANCE.json';acc=read(acceptance)
    assert acc['status'].startswith('PASS_COMPLETE')
    final=Path(summary['sources'][suite]['directory'])
    cp=final/'PRIMARY_S_COMPARISON.json'
    audit=final/'LOCAL_FINAL_AUDIT.json'
    assert sha(cp)==acc['comparison_sha256']==summary['sources'][suite]['comparison_sha256']
    assert sha(audit)==acc['local_final_audit_sha256']
    comp=read(cp);assert comp['status'].startswith('PASS_')
    sources += [acceptance,cp,audit]
    for lam,group,which in [(0.05,'L005','Variant'),(0.1,'L005','S'),(0.2,'L020','Variant')]:
        local=[]
        for seed in [41,42,43]:
            case=comp['cases'][f'{group}_{seed}']
            episodes=case['paired_outcomes'];assert len(episodes)==200
            matrix={}
            for stage in range(6,10):
                for task in range(6,stage+1):
                    es=[x for x in episodes if x['stage']==stage and x['task']==task]
                    assert sorted(x['initial_index'] for x in es)==list(range(20))
                    matrix[stage,task]=Fraction(sum(bool(x[which]) for x in es)*100,20)
                    assert float(matrix[stage,task])==case['cells'][f'{stage}_{task}'][which+'_percent']
            m=matrix
            vals=dict(FWT=sum(m[t,t] for t in range(6,10))/4,
                NBT=sum(sum(m[t,t]-m[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,
                AUC=sum(sum(m[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4)
            ref=next(x for x in summary['per_seed'] if x['suite']==suite and x['lambda_R']==lam and x['seed']==seed)
            for k,v in vals.items():
                assert abs(float(v)-case[which+'_metrics'][k])<1e-10
                assert abs(float(v)-ref[k])<1e-10
            if lam==0.1:
                other=comp['cases'][f'L020_{seed}']
                assert case['S_metrics']==other['S_metrics']
                assert [(x['stage'],x['task'],x['initial_index'],x['S']) for x in episodes]==[(x['stage'],x['task'],x['initial_index'],x['S']) for x in other['paired_outcomes']]
            rec=dict(suite=suite,lambda_R=lam,seed=seed,**{k:float(v) for k,v in vals.items()})
            seeds.append(rec);local.append(rec)
        cells=[]
        for metric in ['FWT','NBT','AUC']:
            values=[x[metric] for x in local]
            mean=statistics.mean(values);sd=statistics.stdev(values)
            ref=next(x for x in summary['means'] if x['suite']==suite and x['lambda_R']==lam)
            assert abs(mean-ref[metric])<1e-10
            cells.append(dict(metric=metric,mean=mean,sd=sd))
        rows.append(dict(suite=suite,lambda_R=lam,values=cells))
for suite in roots:
    subset=[r for r in rows if r['suite']==suite]
    for j in range(3):
        best=(min if j==1 else max)(r['values'][j]['mean'] for r in subset)
        for r in subset:r['values'][j]['best_mean']=abs(r['values'][j]['mean']-best)<1e-10
body=[r'\begin{table}[t]',r'  \centering',
      r'  \caption{Sensitivity to the retention-loss coefficient $\lambda_R$ on LIBERO-OBJECT and LIBERO-GOAL. Results are mean $\pm$ sample standard deviation over three seeds. The best mean for each metric within each suite is shown in bold.}',
      r'  \label{tab:lambda-sensitivity}',r'  \begingroup',r'  \setlength{\tabcolsep}{10pt}',
      r'  \renewcommand{\arraystretch}{1.18}',r'  \begin{tabular}{@{}lcccc@{}}',r'    \toprule',
      r'    & $\lambda_R$ & FWT$\uparrow$ & NBT$\downarrow$ & AUC$\uparrow$ \\',r'    \midrule']
for i,row in enumerate(rows):
    if i==3:body.append(r'    \midrule')
    label=r'\multirow{3}{*}{LIBERO-'+row['suite']+'}' if i%3==0 else ''
    vals=[(r'\bestresult' if v['best_mean'] else r'\result')+f"{{{v['mean']:.1f}}}{{{v['sd']:.1f}}}" for v in row['values']]
    body.append('    '+label+' & '+str(row['lambda_R'])+' & '+' & '.join(vals)+r' \\')
body += [r'    \bottomrule',r'  \end{tabular}',r'  \endgroup',r'\end{table}']
table='\n'.join(body)+'\n'
preamble=style.read_text(encoding='utf8').split(r'\begin{document}')[0]
write(R/'lambda_sensitivity.tex',preamble+r'\begin{document}'+'\n\n'+table+'\n'+r'\end{document}')
write(R/'table_lambda_sensitivity.tex','% Requires booktabs, multirow and result/bestresult macros; see standalone source.\n'+table)
with (R/'per_seed_metrics.csv').open('w',encoding='utf8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['suite','lambda_R','seed','FWT','NBT','AUC'],lineterminator='\n');w.writeheader();w.writerows(seeds)
with (R/'table_values.csv').open('w',encoding='utf8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['suite','lambda_R','metric','mean','sd','best_mean'],lineterminator='\n');w.writeheader()
    for r in rows:
        for v in r['values']:w.writerow(dict(suite=r['suite'],lambda_R=r['lambda_R'],**v))
audit=dict(status='PASS_LAMBDA_OUTCOME_MATRIX_RECOMPUTATION',seed_runs=18,unique_outcomes=3600,matrix_cells=180,
           table_cells=18,seeds=[41,42,43],std='sample, ddof=1',rows=rows,
           source_bindings={str(p.relative_to(E.parent)):sha(p) for p in sources},new_training=0,new_evaluation=0,
           limits=summary['limits'],baseline_counted_once_per_suite=True)
write(R/'TABLE_AUDIT.json',json.dumps(audit,ensure_ascii=False,indent=2))
print(json.dumps(dict(status=audit['status'],rows=rows),indent=2))
