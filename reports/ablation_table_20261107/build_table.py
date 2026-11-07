"""Build the five-configuration table from accepted task-stage matrices."""
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
spec=[('baseline','RA','RA-Coreset'),('baseline','RA_ALIGN','RA-Coreset + ALIGN'),
      ('current','C_normal','+ Execution Supervision'),
      ('current','Q','+ Support-Weighted Retention'),('current','S_latest','Ours')]
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
caption=(r'Component ablation on LIBERO-OBJECT and LIBERO-GOAL (mean $\pm$ sample standard deviation over three seeds). '
         r'Rows cumulatively add ALIGN, execution supervision, support-weighted retention, and candidate gating to RA-Coreset; Ours includes all components. '
         r'Higher FWT and AUC and lower NBT are better. The best mean in each column is shown in bold.')
body=[r'\begin{table*}[t]',r'  \centering',r'  \caption{'+caption+'}',r'  \label{tab:component-ablation}',
      r'  \begingroup',r'  \setlength{\tabcolsep}{7pt}',r'  \renewcommand{\arraystretch}{1.16}',
      r'  \small',r'  \resizebox{\textwidth}{!}{%',r'  \begin{tabular}{@{}l*{6}{c}@{}}',r'    \toprule',
      r'    \multirow{2}{*}{Configuration} & \multicolumn{3}{c}{LIBERO-OBJECT} & \multicolumn{3}{c}{LIBERO-GOAL} \\',
      r'    \cmidrule(lr){2-4}\cmidrule(l){5-7}',
      r'    & FWT$\uparrow$ & NBT$\downarrow$ & AUC$\uparrow$ & FWT$\uparrow$ & NBT$\downarrow$ & AUC$\uparrow$ \\',r'    \midrule']
for row in rows:
    if row['configuration']=='Ours':body.append(r'    \rowcolor{oursblue}')
    label=r'\textbf{Ours}' if row['configuration']=='Ours' else row['configuration']
    vals=[(r'\bestresult' if v['best_mean'] else r'\result')+f"{{{v['mean']:.1f}}}{{{v['sd']:.1f}}}" for v in row['values']]
    body.append('    '+label+' & '+' & '.join(vals)+r' \\')
body += [r'    \bottomrule',r'  \end{tabular}%',r'  }',r'  \endgroup',r'\end{table*}']
table='\n'.join(body)+'\n'
preamble=style.read_text(encoding='utf8').split(r'\begin{document}')[0]
def write(p,s):p.write_bytes(s.rstrip().encode('utf8')+b'\n')
write(R/'ablation.tex',preamble+r'\begin{document}'+'\n\n'+table+'\n'+r'\end{document}')
write(R/'table_ablation.tex','% Cumulative five-configuration ablation; packages/macros documented in README.\n'+table)
with (R/'per_seed_metrics.csv').open('w',encoding='utf8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['configuration','suite','seed',*metrics],lineterminator='\n');w.writeheader();w.writerows(per_seed)
with (R/'table_values.csv').open('w',encoding='utf8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['configuration','suite','metric','mean','sd','best_mean'],lineterminator='\n');w.writeheader()
    for row in rows:
        for v in row['values']:w.writerow(dict(configuration=row['configuration'],**v))
audit=dict(status='PASS_FIVE_CONFIGURATION_MATRIX_RECOMPUTATION',seed_runs=30,matrix_cells=300,table_cells=30,
           seeds=[41,42,43],std='sample standard deviation, ddof=1',precision='one decimal, matching main table',
           rows=rows,source_bindings={str(p.relative_to(E.parent)):sha(p) for p in [source,lineage,style]},
           new_training=0,new_evaluation=0,interpretation='Cumulative configurations; source implementation and protocol differences retained in lineage audit; no evaluation random-stream footnote in table.')
write(R/'TABLE_AUDIT.json',json.dumps(audit,ensure_ascii=False,indent=2))
print(json.dumps(dict(status=audit['status'],rows=[dict(name=r['configuration'],values=[f"{v['mean']:.1f}+/-{v['sd']:.1f}" for v in r['values']]) for r in rows]),ensure_ascii=False,indent=2))
