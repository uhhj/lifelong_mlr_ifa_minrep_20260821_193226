"""Generate manuscript tables from accepted lightweight scalar evidence."""
from pathlib import Path
import hashlib, json, statistics

R = Path(__file__).resolve().parent
W = R.parents[1]
P = W / 'execution/20261101_ablation_summary/DATA.json'
Q = W / 'execution/20261103_lambda_r_goal_sensitivity/final_readonly_20261104_122613/PRIMARY_S_COMPARISON.json'
d = json.loads(P.read_text(encoding='utf8'))
q = json.loads(Q.read_text(encoding='utf8'))
summary = {k:v for k,v in d['summary'].items() if k.startswith(('baseline/', 'current/'))}
rows = [r for r in d['rows'] if r['study'] in ('baseline','current')]
snapshot = dict(source_bindings={str(p.relative_to(W)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P,Q)}, summary=summary, rows=rows, sensitivity=q['suites'])
(R/'results_snapshot.json').write_text(json.dumps(snapshot,indent=2,ensure_ascii=False)+'\n',encoding='utf8')

def table(caption, label, header, body, columns):
    return '\\begin{table}[tb]\n\\centering\\small\n\\caption{'+caption+'}\\label{'+label+'}\n\\begin{tabular}{'+columns+'}\n\\toprule\n'+header+' \\\\\n\\midrule\n'+'\n'.join(body)+'\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'

def row(label, key, metrics):
    s=summary[key]
    vals=[f"${s[m]['mean']:.2f}\\pm{s[m]['sample_sd']:.2f}$" if m=='AUC' else f"{s[m]['mean']:.2f}" for m in metrics]
    return label+' & '+' & '.join(vals)+' \\\\'

out=[]
for suite in ('GOAL','OBJECT'):
    body=[]
    for label,key in [('MLR+IFA','baseline/'+suite+'/MLR_IFA'),('+ RA-Coreset','baseline/'+suite+'/RA'),('+ RA + ALIGN','baseline/'+suite+'/RA_ALIGN'),('Complete method','current/'+suite+'/S_latest')]:
        body.append(row(label,key,['AUC','FWT','NBT','FinalAll']))
    out.append(table(suite+': reproduced baseline and complete method. Percent; AUC is mean $\\pm$ sample standard deviation over seeds 41--43; other columns are means.','tab:main-'+suite,'Method & AUC $\\uparrow$ & FWT $\\uparrow$ & NBT $\\downarrow$ & FinalAll $\\uparrow$',body,'lrrrr'))
(R/'table_main.tex').write_text('\n'.join(out),encoding='utf8')
out=[]
for suite in ('GOAL','OBJECT'):
    body=[row(label,'current/'+suite+'/'+key,['AUC','OldFinal','FinalAll']) for label,key in [('Expert control $C$','C_normal'),('Optimizer-matched $C$','C_equal_budget'),('Ungated $Q$','Q'),('Complete method $S$','S_latest')]]
    out.append(table(suite+': update and retention controls. AUC shows mean $\\pm$ sample standard deviation; final scores are three-seed means.','tab:controls-'+suite,'Method & AUC $\\uparrow$ & OldFinal $\\uparrow$ & FinalAll $\\uparrow$',body,'lrrr'))
(R/'table_controls.tex').write_text('\n'.join(out),encoding='utf8')
body=[]
for label,variant in [('0.05','L005'),('0.10',None),('0.20','L020')]:
    ss=q['suites'][variant or 'L005']
    key='Variant_mean' if variant else 'S_mean'
    body.append(label+' & '+' & '.join(f"{ss[m][key]:.2f}" for m in ['AUC','FWT','NBT','OldFinal','FinalAll'])+' \\\\')
text=table('GOAL retention-weight sensitivity: three-seed means in percent. The primary 0.10 setting reuses accepted historical runs; additional settings use the same scientific budgets, with hardware differences recorded.','tab:lambda','$\\lambda_R$ & AUC & FWT & NBT $\\downarrow$ & OldFinal & FinalAll',body,'rrrrrr')
text+='\nThe primary weight 0.1 achieves the highest mean AUC and FinalAll. A weight of 0.2 increases OldFinal but reduces FWT, revealing a retention--acquisition trade-off. These results do not establish invariance to the retention weight.\n'
(R/'table_sensitivity.tex').write_text(text,encoding='utf8')
print('Generated three table files and source-bound scalar snapshot.')
