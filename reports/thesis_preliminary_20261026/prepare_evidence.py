"""Recompute thesis tables and plot recorded scalar logs; no experiment execution."""
from pathlib import Path
import json,hashlib,statistics,csv,datetime,sys
from fractions import Fraction as F
R=Path(__file__).resolve().parent;W=R.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metrics(m):
 v=lambda s,t:F(str(m[f'{s}_{t}']))/100
 return {'AUC':sum(sum(v(s,t) for s in range(t,10))/(10-t) for t in range(6,10))/4*100,
 'FWT':sum(v(t,t) for t in range(6,10))/4*100,
 'NBT':sum(sum(v(t,t)-v(s,t) for s in range(t+1,10))/(9-t) for t in range(6,9))/3*100,
 'OldFinal':sum(v(9,t) for t in range(6,9))/3*100,
 'FinalAll':sum(v(9,t) for t in range(6,10))/4*100}
sources={};runs=[];summary={}
for suite in ['GOAL','OBJECT']:
 p=W/f'execution/20261025_ccbp_v4_full/{suite}_THREE_SEED_BASELINES.json';sources[str(p.relative_to(W))]=sha(p);a=read(p)
 assert a['status']=='PASS_LOCAL_READONLY_RECOMPUTATION'
 for run in a['runs']:
  got=metrics(run['matrix'])
  for k,v in got.items():assert abs(float(v)-run['metrics'][k]['percent'])<1e-9
  runs.append(dict(suite=suite,seed=run['seed'],method=run['method'],**{k:float(v) for k,v in got.items()}))
 summary[suite]={}
 for method in ['MLR_IFA','RA','RA_ALIGN']:
  rows=[x for x in runs if x['suite']==suite and x['method']==method];assert len(rows)==3
  summary[suite][method]={k:{'mean':statistics.mean(x[k] for x in rows),'sd':statistics.stdev(x[k] for x in rows)} for k in ['AUC','FWT','NBT','OldFinal','FinalAll']}
ref=W/'execution/20261026_ccbp_v4_s_goal42_43/REFERENCE_RESULTS.json';sources[str(ref.relative_to(W))]=sha(ref);s=read(ref)['S41'];sm=metrics({k:v['percent'] for k,v in s['matrix'].items()})
assert all(abs(float(sm[k])-s['metrics'][k]['percent'])<1e-9 for k in sm)
runs.append(dict(suite='GOAL',seed=41,method='S',**{k:float(v) for k,v in sm.items()}))
with (R/'all_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(runs[0]));w.writeheader();w.writerows(runs)
rp=W/'execution/20261025_ccbp_v4_full/final_readonly_20261026_014654/EVIDENCE.json';sources[str(rp.relative_to(W))]=sha(rp);ev=read(rp)
epoch=ev['light']['EPOCHS_S.json'];raw=epoch['text'].encode('utf8');assert hashlib.sha256(raw).hexdigest()==epoch['sha256']
(R/'EPOCHS_S41.json').write_bytes(raw);epochs=json.loads(raw);assert len(epochs)==400 and epochs[-1]['steps']==76100
pr=W/'execution/20261025_ccbp_v4_full/check_readonly_20261026_005436/EVIDENCE.json';sources[str(pr.relative_to(W))]=sha(pr);log=read(pr)['log_tails']['train_S.log']
assert isinstance(log,str)
validlines=[]
for line in log.splitlines():
 try:
  row=json.loads(line)
  if row.get('arm')=='S' and row.get('stage')==9 and row.get('epoch') in [97,98,99]:
   match=next(x for x in epochs if x['stage']==9 and x['epoch']==row['epoch']);assert all(row[k]==v for k,v in match.items());validlines.append(line)
 except json.JSONDecodeError:pass
assert len(validlines)==3;(R/'train_S41_original_excerpt.jsonl').write_text('\n'.join(validlines)+'\n',encoding='utf8')
with (R/'training_scalars_S41.csv').open('w',encoding='utf-8-sig',newline='') as f:
 fields=['stage','epoch','completed_epochs_in_stage','steps','batches','seconds','lr_after','bc_batch_mean','expert_batch_mean','retain_batch_mean'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for x in epochs:w.writerow(dict(stage=x['stage'],epoch=x['epoch'],completed_epochs_in_stage=x['epoch']+1,steps=x['steps'],batches=x['batches'],seconds=x['seconds'],lr_after=x['lr_after'],bc_batch_mean=x['totals']['bc']/x['batches'],expert_batch_mean=x['totals']['expert_loss']/x['batches'],retain_batch_mean=x['totals']['retain_loss']/x['batches']))
sys.path.insert(0,str(R/'.plot_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(10.6,3.6),layout='constrained');colors=['#1599BE','#EB8D23','#2879B2','#67A53F']
for stage,col in zip(range(6,10),colors):
 rows=[x for x in epochs if x['stage']==stage];axes[0].plot([x['epoch']+1 for x in rows],[x['totals']['bc']/x['batches'] for x in rows],label=f'Task {stage}',color=col,lw=1.8)
axes[0].set(xlabel='Completed epochs in stage',ylabel='BC NLL (batch mean)',title='Recorded BC training loss');axes[0].legend(frameon=False,ncol=2);axes[0].grid(alpha=.17)
rows=[x for x in epochs if x['stage']==9]
axes[1].plot([x['epoch']+1 for x in rows],[x['totals']['retain_loss']/x['batches'] for x in rows],color=colors[-1],lw=1.8)
axes[1].set(xlabel='Completed epochs in Task 9',ylabel='Retention loss (batch mean)',title='Recorded S retention loss');axes[1].grid(alpha=.17)
fig.savefig(R/'training_curves_S41.png',dpi=220);fig.savefig(R/'training_curves_S41.pdf');plt.close(fig)
latex=R/'latex_tables';latex.mkdir(exist_ok=True)
def fmt(q):return f"{q['mean']:.2f} $\\pm$ {q['sd']:.2f}"
rows=[]
for suite,method,label in [('GOAL','MLR_IFA','原MLR+IFA'),('GOAL','RA','MLR+IFA+RA'),('OBJECT','MLR_IFA','原MLR+IFA'),('OBJECT','RA','MLR+IFA+RA'),('OBJECT','RA_ALIGN','MLR+IFA+RA+ALIGN')]:
 q=summary[suite][method];rows.append(f"{suite} & {label} & {fmt(q['AUC'])} & {fmt(q['FinalAll'])} \\\\")
tab='''% Source: audited GOAL/OBJECT seeds 41,42,43; units percent; sample SD.
\\begin{table}[htbp]
\\centering
\\caption{MLR+IFA复现与回放方法的已完成闭环结果}
\\label{tab:preliminary-baselines}
\\begin{tabular}{llcc}
\\toprule
套件 & 方法 & AUC（\\%） & 最终成功率（\\%） \\\\
\\midrule
'''+ '\n'.join(rows)+'''
\\bottomrule
\\end{tabular}
\\par\\smallskip
{\\footnotesize seed41、42、43的均值$\\pm$样本标准差，每个阶段--任务单元20回合。
最终成功率为最后阶段四任务平均，数值为项目复现结果。}
\\end{table}
'''
(latex/'table_baselines.tex').write_text(tab,encoding='utf8')
stab=r'''% S41 accepted final evaluation, separate from three-seed means.
\begin{table}[htbp]
\centering
\caption{S方法在GOAL seed41上的最终闭环结果}
\label{tab:preliminary-s41}
\begin{tabular}{lrrrrrr}
\toprule
方法 & Task6 & Task7 & Task8 & Task9 & 旧任务平均 & 全部平均 \\
\midrule
S & 45.00 & 100.00 & 100.00 & 95.00 & 81.67 & 85.00 \\
\bottomrule
\end{tabular}
\par\smallskip
{\footnotesize 单位为\%，每任务20正式回合。另有40次训练采集，单种子初步验证。}
\end{table}
'''
(latex/'table_S41.tex').write_text(stab,encoding='utf8')
defs=r'''\paragraph{指标定义}
令$r_{s,t}$表示学完阶段$s$后在任务$t$上的闭环成功率，$T=4$，任务索引重编号为$1,\ldots,T$。
\begin{align}
\mathrm{AUC}&=\frac{1}{T}\sum_{t=1}^{T}\frac{1}{T-t+1}\sum_{s=t}^{T}r_{s,t},\\
\mathrm{OldFinal}&=\frac{1}{T-1}\sum_{t=1}^{T-1}r_{T,t},\\
\mathrm{FinalAll}&=\frac{1}{T}\sum_{t=1}^{T}r_{T,t}.
\end{align}
表中数值为上述比例乘以100。三种子的离散程度使用样本标准差（分母$n-1$）。
'''
(latex/'metric_definitions.tex').write_text(defs,encoding='utf8')
audit={'status':'PASS_LOCAL_SCALAR_RECOMPUTATION','time':datetime.datetime.now().astimezone().isoformat(),'baseline_runs':18,'metrics_recomputed':['AUC','FWT','NBT','OldFinal','FinalAll'],'summary':summary,'S41':{k:float(v) for k,v in sm.items()},'main_training_epochs':400,'main_training_steps':76100,'training_excerpt_original_lines':3,'curve_definition':'Per-epoch recorded BC or retention totals divided by that epoch batch count; unsmoothed; not success rate.','sources_sha256':sources,'new_training':0,'new_rollouts':0,'S42_43':'PARTIAL per existing 2026-10-26 14:55:25 observation; not evaluated or included by this artifact task.'}
(R/'EVIDENCE_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({'status':audit['status'],'epochs':400,'baseline_runs':18,'excerpt_lines':3,'files_generated':['all_metrics.csv','training_scalars_S41.csv','training_curves_S41.png','training_curves_S41.pdf','latex_tables/table_baselines.tex','latex_tables/table_S41.tex']}))
