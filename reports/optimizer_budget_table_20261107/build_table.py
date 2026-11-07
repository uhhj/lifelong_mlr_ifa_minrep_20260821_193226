"""Build the optimizer-budget table from accepted task-stage matrices."""
from pathlib import Path
import csv
import hashlib
import json
import statistics
from fractions import Fraction

R=Path(__file__).resolve().parent
E=R.parent
source=E/'20261101_ablation_summary/DATA.json'
lineage=E/'20261107_align_cqs_audit/AUDIT.json'
style=E/'20261107_main_comparison_table/main_comparison.tex'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d=json.loads(source.read_text(encoding='utf8'))
a=json.loads(lineage.read_text(encoding='utf8'))
assert d['status']=='PASS_LOCAL_METRIC_RECOMPUTATION'
assert a['status']=='PASS_LOCAL_LINEAGE_AND_PROTOCOL_AUDIT'
assert a['source_bindings']['execution/20261101_ablation_summary/DATA.json']['sha256']==sha(source)
spec=[('current','C_normal','Execution Supervision'),
      ('current','C_equal_budget','+ Matched Optimizer Budget'),('current','S_latest','Ours')]
budget_root=E/'20261030_equal_optimizer_budget_cs_two_suites'
acceptance=budget_root/'ACCEPTANCE.json'
comparison=budget_root/'final_readonly_20261031_065009/PRIMARY_S_COMPARISON.json'
final_audit=budget_root/'final_readonly_20261031_065009/LOCAL_FINAL_AUDIT.json'
acc=json.loads(acceptance.read_text(encoding='utf8'))
comp=json.loads(comparison.read_text(encoding='utf8'))
assert acc['status']=='ACCEPTED_SIX_C_EQUAL_OPTIMIZER_BUDGET'
assert acc['comparison_sha256']==sha(comparison) and acc['audit_sha256']==sha(final_audit)
budgets={}
for suite in ['OBJECT','GOAL']:
    for seed in [41,42,43]:
        stages=comp['cases'][f'{suite}_{seed}']['stage_budgets']
        assert all(v['C_production']==v['S_production'] for v in stages.values())
        total=sum(v['C_production'] for v in stages.values())
        normal=stages['6']['C_production']+sum(stages[str(t)]['C_production']//2 for t in [7,8,9])
        assert total==dict(OBJECT=195300,GOAL=142200)[suite]
        assert normal==dict(OBJECT=105500,GOAL=76100)[suite]
        budgets[suite]=dict(C_normal=normal,C_equal_budget=total,S_latest=total)
metrics=['FWT','NBT','AUC']
rows=[];per_seed=[]
for study,method,label in spec:
    cells=[]
    for suite in ['OBJECT','GOAL']:
        found=[r for r in d['rows'] if r['study']==study and r['method']==method and r['suite']==suite]
        assert sorted(r['seed'] for r in found)==[41,42,43]
        local=[]
        for r in found:
            m={tuple(map(int,k.split('_'))):Fraction(str(v)) for k,v in r['matrix_percent'].items()}
            assert set(m)=={(s,t) for s in range(6,10) for t in range(6,s+1)}
            vals=dict(FWT=sum(m[t,t] for t in range(6,10))/4,
                NBT=sum(sum(m[t,t]-m[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,
                AUC=sum(sum(m[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4)
            assert all(abs(float(v)-r['metrics_percent'][k])<1e-10 for k,v in vals.items())
            rec=dict(configuration=label,suite=suite,seed=r['seed'],**{k:float(v) for k,v in vals.items()})
            per_seed.append(rec);local.append(rec)
        for metric in metrics:
            values=[v[metric] for v in local]
            mean=statistics.mean(values);sd=statistics.stdev(values)
            old=d['summary'][f'{study}/{suite}/{method}'][metric]
            assert abs(mean-old['mean'])<1e-10 and abs(sd-old['sample_sd'])<1e-10
            cells.append(dict(suite=suite,metric=metric,mean=mean,sd=sd))
    rows.append(dict(configuration=label,source_study=study,source_method=method,values=cells))
for j in range(6):
    best=(min if j%3==1 else max)(r['values'][j]['mean'] for r in rows)
    for r in rows:r['values'][j]['best_mean']=abs(r['values'][j]['mean']-best)<1e-10
caption=(r'Optimizer-budget comparison on LIBERO-OBJECT and LIBERO-GOAL (mean $\pm$ sample standard deviation over three seeds). '
         r'Execution Supervision denotes RA-Coreset + ALIGN with execution supervision; the second row extends its training to match the optimizer budget of Ours. '
         r'Opt. steps counts physical production optimizer calls per run, including discarded candidates. Best means are bold.')
body=[r'\begin{table*}[t]',r'  \centering',r'  \caption{'+caption+'}',r'  \label{tab:optimizer-budget}',
      r'  \begingroup',r'  \setlength{\tabcolsep}{7pt}',r'  \renewcommand{\arraystretch}{1.16}',
      r'  \small',r'  \resizebox{\textwidth}{!}{%',r'  \begin{tabular}{@{}l*{8}{c}@{}}',r'    \toprule',
      r'    \multirow{2}{*}{Configuration} & \multicolumn{4}{c}{LIBERO-OBJECT} & \multicolumn{4}{c}{LIBERO-GOAL} \\',
      r'    \cmidrule(lr){2-5}\cmidrule(l){6-9}',
      r'    & Opt. steps & FWT$\uparrow$ & NBT$\downarrow$ & AUC$\uparrow$ & Opt. steps & FWT$\uparrow$ & NBT$\downarrow$ & AUC$\uparrow$ \\',r'    \midrule']
for row in rows:
    if row['configuration']=='Ours':body.append(r'    \rowcolor{oursblue}')
    label=r'\textbf{Ours}' if row['configuration']=='Ours' else row['configuration']
    vals=[(r'\bestresult' if v['best_mean'] else r'\result')+f"{{{v['mean']:.1f}}}{{{v['sd']:.1f}}}" for v in row['values']]
    vals=[f"{budgets['OBJECT'][row['source_method']]:,}"]+vals[:3]+[f"{budgets['GOAL'][row['source_method']]:,}"]+vals[3:]
    body.append('    '+label+' & '+' & '.join(vals)+r' \\')
body += [r'    \bottomrule',r'  \end{tabular}%',r'  }',r'  \endgroup',r'\end{table*}']
table='\n'.join(body)+'\n'
preamble=style.read_text(encoding='utf8').split(r'\begin{document}')[0]
def write(p,s):p.write_bytes(s.rstrip().encode('utf8')+b'\n')
write(R/'optimizer_budget.tex',preamble+r'\begin{document}'+'\n\n'+table+'\n'+r'\end{document}')
write(R/'table_optimizer_budget.tex','% Optimizer-budget comparison; packages/macros documented in README.\n'+table)
with (R/'per_seed_metrics.csv').open('w',encoding='utf8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['configuration','suite','seed',*metrics],lineterminator='\n');w.writeheader();w.writerows(per_seed)
with (R/'table_values.csv').open('w',encoding='utf8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['configuration','suite','metric','mean','sd','best_mean'],lineterminator='\n');w.writeheader()
    for row in rows:
        for v in row['values']:w.writerow(dict(configuration=row['configuration'],**v))
audit=dict(status='PASS_OPTIMIZER_BUDGET_MATRIX_RECOMPUTATION',seed_runs=18,matrix_cells=180,table_cells=18,budgets=budgets,
           seeds=[41,42,43],std='sample standard deviation, ddof=1',precision='one decimal, matching main table',
           rows=rows,source_bindings={str(p.relative_to(E.parent)):sha(p) for p in [source,lineage,style,acceptance,comparison,final_audit,budget_root/'PROTOCOL.md']},
           new_training=0,new_evaluation=0,interpretation='Matched physical production optimizer calls, not matched FLOPs, wall-clock time or environment interactions. Standard and matched baselines share the execution-supervision mechanism; only training budget and schedule horizon change.')
write(R/'TABLE_AUDIT.json',json.dumps(audit,ensure_ascii=False,indent=2))
print(json.dumps(dict(status=audit['status'],rows=[dict(name=r['configuration'],values=[f"{v['mean']:.1f}+/-{v['sd']:.1f}" for v in r['values']]) for r in rows]),ensure_ascii=False,indent=2))
