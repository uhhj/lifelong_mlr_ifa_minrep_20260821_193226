"""Build a self-contained LaTeX comparison table from the supplied Table 1 and accepted S results."""
from pathlib import Path
import csv
import hashlib
import json
import re
import statistics
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parent
EXEC=ROOT.parent
PDF=next((Path('C:/Users/Lenovo/Zotero/storage')/'8WVD3WFZ').glob('*.pdf'))
COMP=EXEC/'20261031_full_q_vs_latest_s_two_suites/final_readonly_20261101_032229/PRIMARY_S_COMPARISON.json'
ACC=COMP.parent.parent/'ACCEPTANCE.json'
PREVIOUS=EXEC/'20261103_lambda_r_goal_sensitivity/publication_figures_20261106/data/method_metrics_per_seed.csv'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
acceptance=json.loads(ACC.read_text(encoding='utf8'))
assert acceptance['comparison_sha256']==sha(COMP)
comparison=json.loads(COMP.read_text(encoding='utf8'))
page=PdfReader(PDF).pages[5].extract_text()
lines=page.splitlines()
spec=[
    ('Sequential','Sequential',r'\cite{liu2023libero}'),
    ('ER','ER',r'\cite{chaudhry2019tiny,liu2023libero}'),
    ('BUDS','BUDS',r'\cite{wan2024lotus,zhu2022buds}'),
    ('LOTUS','LOTUS',r'\cite{wan2024lotus}'),
    ('ISCIL','ISCIL',r'\cite{lee2024iscil}'),
    ('M2Distill','M2Distill',r'\cite{roy2025m2distill}'),
    ('TAIL','TAIL',r'\cite{liu2023tail}'),
]
rows=[]
for source_name,name,cite in spec:
    line=next(x for x in lines if x.startswith(source_name+' '))
    numbers=re.findall(r'(\d+(?:\.\d)?)\s*±\s*(\d+\.\d)',line)
    assert len(numbers)>=6,(name,line,numbers)
    rows.append(dict(method=name,citation=cite,values=[dict(mean=float(a),sd=float(b)) for a,b in numbers[:6]]))
# Recompute the project's reproduced MLR+IFA from its six fixed matrices.
REPRO=EXEC/'20261101_ablation_summary/DATA.json'
REPRO_REVIEW=EXEC/'20261105_final_vs_mlr_nbt/EVIDENCE_REVIEW.json'
binding=json.loads(REPRO_REVIEW.read_text(encoding='utf8'))
assert binding['source_bindings']['execution/20261101_ablation_summary/DATA.json']['sha256']==sha(REPRO)
reproduction=json.loads(REPRO.read_text(encoding='utf8'))
repro_seed=[];repro_values=[]
for suite in ['OBJECT','GOAL']:
    subset=[r for r in reproduction['rows'] if r['study']=='baseline' and r['method']=='MLR_IFA' and r['suite']==suite]
    assert sorted(r['seed'] for r in subset)==[41,42,43]
    for r in subset:
        mat={tuple(map(int,k.split('_'))):v for k,v in r['matrix_percent'].items()}
        vals=dict(FWT=statistics.mean(mat[t,t] for t in range(6,10)),
            NBT=statistics.mean(statistics.mean(mat[t,t]-mat[s,t] for s in range(t+1,10)) for t in range(6,9)),
            AUC=statistics.mean(statistics.mean(mat[s,t] for s in range(t,10)) for t in range(6,10)))
        assert all(abs(vals[k]-r['metrics_percent'][k])<1e-10 for k in vals)
        repro_seed.append(dict(suite=suite,seed=r['seed'],**vals))
    for metric in ['FWT','NBT','AUC']:
        v=[r[metric] for r in repro_seed if r['suite']==suite]
        repro_values.append(dict(mean=statistics.mean(v),sd=statistics.stdev(v)))
rows.append(dict(method='MLR + IFA',citation='',values=repro_values))
per_seed=[];ours=[]
previous=list(csv.DictReader(PREVIOUS.open(encoding='utf8')))
for suite in ['OBJECT','GOAL']:
    seed_metrics=[]
    for seed in [41,42,43]:
        case=comparison['cases'][f'{suite}_{seed}'];episodes=case['paired_outcomes']
        key={(r['stage'],r['task'],r['initial_index']):r for r in episodes}
        assert len(key)==200 and set(key)=={(s,t,i) for s in range(6,10) for t in range(6,s+1) for i in range(20)}
        mat={(s,t):5*sum(key[s,t,i]['S'] for i in range(20)) for s in range(6,10) for t in range(6,s+1)}
        vals=dict(FWT=statistics.mean(mat[t,t] for t in range(6,10)),
            NBT=statistics.mean(statistics.mean(mat[t,t]-mat[s,t] for s in range(t+1,10)) for t in range(6,9)),
            AUC=statistics.mean(statistics.mean(mat[s,t] for s in range(t,10)) for t in range(6,10)))
        old=next(r for r in previous if r['suite']==suite and int(r['seed'])==seed and r['method']=='S')
        assert all(abs(vals[k]-float(old[k]))<1e-10 and abs(vals[k]-case['S_metrics'][k])<1e-10 for k in vals)
        per_seed.append(dict(suite=suite,seed=seed,**vals));seed_metrics.append(vals)
    for metric in ['FWT','NBT','AUC']:
        v=[r[metric] for r in seed_metrics]
        ours.append(dict(mean=statistics.mean(v),sd=statistics.stdev(v)))
rows.append(dict(method='Ours',citation='',values=ours))
best=[(min if j%3==1 else max)(r['values'][j]['mean'] for r in rows) for j in range(6)]
for row in rows:
    for j,val in enumerate(row['values']):val['best_mean']=abs(val['mean']-best[j])<1e-9

caption=(r'Comparison on LIBERO-OBJECT and LIBERO-GOAL (mean $\pm$ standard deviation). '
         r'Higher FWT and AUC and lower NBT are better. The best mean in each column is shown in bold.')
body=[r'\begin{table*}[t]',r'  \centering',r'  \caption{'+caption+'}',r'  \label{tab:main-comparison}',
      r'  \begingroup',r'  \setlength{\tabcolsep}{7pt}',r'  \renewcommand{\arraystretch}{1.16}',
      r'  \small',r'  \resizebox{\textwidth}{!}{%',r'  \begin{tabular}{@{}l*{6}{c}@{}}',r'    \toprule',
      r'    \multirow{2}{*}{Method} & \multicolumn{3}{c}{LIBERO-OBJECT} & \multicolumn{3}{c}{LIBERO-GOAL} \\',
      r'    \cmidrule(lr){2-4}\cmidrule(l){5-7}',
      r'    & FWT$\uparrow$ & NBT$\downarrow$ & AUC$\uparrow$ & FWT$\uparrow$ & NBT$\downarrow$ & AUC$\uparrow$ \\',r'    \midrule']
for row in rows:
    if row['method']=='Ours':body.append(r'    \rowcolor{oursblue}')
    label=(r'\textbf{Ours}' if row['method']=='Ours' else row['method'])
    cells=[]
    for val in row['values']:
        macro=r'\bestresult' if val['best_mean'] else r'\result'
        cells.append(macro+f"{{{val['mean']:.1f}}}{{{val['sd']:.1f}}}")
    body.append('    '+label+' & '+' & '.join(cells)+r' \\')
body += [r'    \bottomrule',r'  \end{tabular}%',r'  }',r'  \endgroup',r'\end{table*}']
table='\n'.join(body)+'\n'
preamble=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=17mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{mathptmx}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{graphicx}
\usepackage[table]{xcolor}
\usepackage{caption}
\usepackage[colorlinks=true,citecolor=blue,linkcolor=black,urlcolor=blue]{hyperref}
\captionsetup{font=small,labelfont=bf,singlelinecheck=false}
\definecolor{oursblue}{RGB}{211,235,241}
\newcommand{\result}[2]{\ensuremath{#1\!\pm\!#2}}
\newcommand{\bestresult}[2]{\ensuremath{\mathbf{#1\!\pm\!#2}}}
\pagestyle{empty}
\begin{document}
'''
# Keep the requested main .tex self-contained; no external files or BibTeX needed.
(ROOT/'main_comparison.tex').write_text(preamble+'\n'+table+'\n'+r'\end{document}'+'\n',encoding='utf8')
(ROOT/'table_comparison.tex').write_text('% Table fragment; add packages/macros from README to the manuscript preamble.\n'+table,encoding='utf8')
with (ROOT/'table_values.csv').open('w',encoding='utf8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['method','suite','metric','mean','sd','best_mean']);w.writeheader()
    for row in rows:
        for j,val in enumerate(row['values']):w.writerow(dict(method=row['method'],suite=['OBJECT','GOAL'][j//3],metric=['FWT','NBT','AUC'][j%3],**val))
with (ROOT/'ours_per_seed.csv').open('w',encoding='utf8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['suite','seed','FWT','NBT','AUC']);w.writeheader();w.writerows(per_seed)
with (ROOT/'mlr_ifa_per_seed.csv').open('w',encoding='utf8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['suite','seed','FWT','NBT','AUC']);w.writeheader();w.writerows(repro_seed)
audit=dict(status='PASS_BASELINE_TRANSCRIPTION_AND_OURS_RECOMPUTATION',suites=['OBJECT','GOAL'],methods=len(rows),numeric_cells=6*len(rows),
    baseline_source=dict(filename=PDF.name,pdf_page=6,table=1,sha256=sha(PDF),provenance='Baseline values reproduced from supplied paper table, not new reruns'),
    ours_source=dict(path=str(COMP),sha256=sha(COMP),acceptance_sha256=sha(ACC),previous_plotted_values_sha256=sha(PREVIOUS),method='Latest complete S; lambda_R=0.1',seeds=[41,42,43],std='Sample standard deviation across seeds, ddof=1',new_training=0,new_evaluation=0),
    mlr_ifa_source=dict(path=str(REPRO),sha256=sha(REPRO),review_sha256=sha(REPRO_REVIEW),study='baseline',method='MLR_IFA',seeds=[41,42,43],recomputed_from='six task-stage matrices',std='sample, ddof=1'),
    rows=rows,best_means=best,
    formatting='Nine methods in one unified table; MLR removed; MLR+IFA uses project reproduction; no citation labels or bibliography. No LIBERO-50 columns.',
    citation_metadata_url='https://github.com/yfqi/lifelong_mlr_ifa#citation',
    limits='Table is not a claim that all baselines were rerun under an identical hardware/evaluation pipeline; source provenance is retained in this audit.')
(ROOT/'TABLE_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=audit['status'],ours=ours),indent=2))
