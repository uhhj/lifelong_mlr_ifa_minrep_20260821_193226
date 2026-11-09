"""Recompute the 18 historical, protocol-matched runs; no scientific execution."""
from pathlib import Path
from fractions import Fraction as F
import csv
import datetime
import hashlib
import json
import statistics
from decimal import Decimal, ROUND_HALF_UP

def fmt(x, signed=False):
    q=Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return ("+" if signed and q>=0 else "")+str(q)

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sources={}
def raw(p):
    p=Path(p)
    if not p.is_absolute():p=ROOT/p
    b=p.read_bytes();sources[p.relative_to(ROOT).as_posix()]=hashlib.sha256(b).hexdigest()
    return b
def read(p):return json.loads(raw(p).decode('utf-8-sig'))
def metrics(m):
    return dict(AUC=sum(sum(m[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,
        FWT=sum(m[t,t] for t in range(6,10))/4,
        NBT=sum(sum(m[t,t]-m[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,
        OldFinal=sum(m[9,t] for t in range(6,9))/3,NewFinal=m[9,9],FinalAll=sum(m[9,t] for t in range(6,10))/4)
def packed(m):return {k:dict(fraction=str(v),percent=float(100*v)) for k,v in m.items()}

prior=read('execution/20261109_kcenter_ra_matched_comparison/RESULTS.json')
assert prior['status']=='PASS_READONLY_MATCHED_PROTOCOL_RECOMPUTATION'
for p,h in prior['source_sha256'].items():assert hashlib.sha256(raw(p)).hexdigest()==h,p
obj=read('execution/20261019_ra_object_conditional/BASELINE_EPISODE_AUDIT.json')
runs=[];ids={}
for suite in ['OBJECT','GOAL']:
    base=read(f'execution/20261025_ccbp_v4_full/{suite}_THREE_SEED_BASELINES.json')
    hashes={str(Path(p).resolve()):h for p,h in base['source_sha256'].items()}
    for seed in [41,42,43]:
        kr=next(r for r in prior['runs'] if (r['suite'],r['seed'],r['method'])==(suite,seed,'K-Center'))
        for method in ['Random','K-Center','RA']:
            if method=='Random':
                files=[ROOT/str(Path(p)).replace('coreset\\evaluation','random\\evaluation') for p in kr['sources']]
                reference=next(r for r in base['runs'] if r['seed']==seed and r['method']=='MLR_IFA')['metrics']
            else:
                rr=next(r for r in prior['runs'] if (r['suite'],r['seed'],r['method'])==(suite,seed,method))
                files=[ROOT/p for p in rr['sources']];reference=rr['metrics']
            m={};identity={}
            assert len(files)==10
            for p in files:
                b=raw(p);h=hashlib.sha256(b).hexdigest()
                if method=='Random':
                    assert h==hashes[str(p.resolve())],p
                    if suite=='OBJECT':
                        expected=obj['seeds'][str(seed)]['input_episode_sha256']
                        assert h==expected[obj['seeds'][str(seed)]['parent']+'/random/evaluation/'+p.name]
                else:assert h==prior['source_sha256'][p.relative_to(ROOT).as_posix()]
                rows=[json.loads(x) for x in b.decode('utf-8-sig').splitlines() if x.strip()]
                assert len(rows)==20
                s,t=rows[0]['stage'],rows[0]['task']
                assert (s,t) not in m
                assert all(r['seed']==seed and r['stage']==s and r['task']==t and isinstance(r['success'],bool) for r in rows)
                assert sorted(r['initial_state_index'] for r in rows)==list(range(20))
                m[s,t]=F(sum(r['success'] for r in rows),20)
                identity[s,t]={r['initial_state_index']:r['initial_state_sha256'] for r in rows}
            assert set(m)=={(s,t) for s in range(6,10) for t in range(6,s+1)}
            vals=metrics(m)
            for k,v in reference.items():assert vals[k]==F(v['fraction']),(suite,seed,method,k)
            ids[suite,seed,method]=identity
            runs.append(dict(suite=suite,seed=seed,method=method,records=200,metrics=packed(vals),
                matrix={f'{s}_{t}':float(v*100) for (s,t),v in sorted(m.items())},
                sources=[p.relative_to(ROOT).as_posix() for p in files]))
        assert ids[suite,seed,'Random']==ids[suite,seed,'K-Center']==ids[suite,seed,'RA']

names=list(runs[0]['metrics']);methods=['Random','K-Center','RA']
summary={};deltas=[]
for suite in ['OBJECT','GOAL']:
    summary[suite]={}
    for method in methods:
        rr=[r for r in runs if r['suite']==suite and r['method']==method]
        summary[suite][method]={k:dict(mean_fraction=str(sum(F(r['metrics'][k]['fraction']) for r in rr)/3),
            mean_percent=float(sum(F(r['metrics'][k]['fraction']) for r in rr)/3*100),
            sample_sd_pp=statistics.stdev(r['metrics'][k]['percent'] for r in rr)) for k in names}
    for comparator in ['Random','K-Center']:
        for seed in [41,42,43]:
            a=next(r for r in runs if (r['suite'],r['seed'],r['method'])==(suite,seed,'RA'))
            b=next(r for r in runs if (r['suite'],r['seed'],r['method'])==(suite,seed,comparator))
            deltas.append(dict(suite=suite,seed=seed,comparison='RA minus '+comparator,
                metrics=packed({k:F(a['metrics'][k]['fraction'])-F(b['metrics'][k]['fraction']) for k in names})))
        deltas.append(dict(suite=suite,seed='mean',comparison='RA minus '+comparator,
            metrics=packed({k:F(summary[suite]['RA'][k]['mean_fraction'])-F(summary[suite][comparator][k]['mean_fraction']) for k in names})))
out=dict(status='PASS_18_RUN_READONLY_RECOMPUTATION',observed=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    records_recomputed=3600,initial_state_identity_sets_checked=1200,methods=methods,seeds=[41,42,43],runs=runs,summary=summary,deltas=deltas,
    definitions=prior['definitions'],limits=prior['limits'],object43_parent=prior['object43_parent'],
    scope='Random, K-Center and RA selection, all on MLR+IFA, before ALIGN or execution retention/gating.',
    new_training=0,new_forward=0,new_environment=0,source_sha256=sources)
for p,h in sources.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
(HERE/'RESULTS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
with (HERE/'FULL_METRICS.csv').open('w',encoding='utf8',newline='') as f:
    w=csv.writer(f);w.writerow(['suite','seed','method',*names])
    for r in runs:w.writerow([r['suite'],r['seed'],r['method'],*[r['metrics'][k]['percent'] for k in names]])
with (HERE/'SUMMARY.csv').open('w',encoding='utf8',newline='') as f:
    w=csv.writer(f);w.writerow(['suite','method',*[x for k in names for x in (k+'_mean',k+'_sd')]])
    for suite in summary:
        for method in methods:w.writerow([suite,method,*[x for k in names for x in (summary[suite][method][k]['mean_percent'],summary[suite][method][k]['sample_sd_pp'])]])
with (HERE/'DELTAS.csv').open('w',encoding='utf8',newline='') as f:
    w=csv.writer(f);w.writerow(['suite','seed','comparison',*names])
    for d in deltas:w.writerow([d['suite'],d['seed'],d['comparison'],*[d['metrics'][k]['percent'] for k in names]])

lines=['# Random、K-Center 与 RA：seed41、42、43 完整指标','',
 '基于18个既有run的3600条正式回合逐项复算，原指标精确分数与接受记录一致；未新增训练、前向或环境调用。三个方案均为 MLR+IFA 下的记忆选样方法，不含 ALIGN、执行保持或门控。','',
 '单位：%。NBT 越低越好，其余越高越好；OldFinal/NewFinal/FinalAll 分别为最后阶段旧任务（6–8）、新任务（9）、全部任务（6–9）的成功率。均值±样本标准差基于三个seed，ddof=1。差值单位为百分点（pp），不是相对百分比。','']
for suite in ['OBJECT','GOAL']:
    lines += [f'## {suite}', '', '|seed|方法|AUC↑|FWT↑|NBT↓|OldFinal↑|NewFinal↑|FinalAll↑|','|---|---|---:|---:|---:|---:|---:|---:|']
    for seed in [41,42,43]:
        for method in methods:
            r=next(r for r in runs if (r['suite'],r['seed'],r['method'])==(suite,seed,method))
            lines.append(f'|{seed}|{method}|'+'|'.join(fmt(r['metrics'][k]['percent']) for k in names)+'|')
    for method in methods:
        lines.append(f'|均值±SD|{method}|'+'|'.join(f"{fmt(summary[suite][method][k]['mean_percent'])} ± {fmt(summary[suite][method][k]['sample_sd_pp'])}" for k in names)+'|')
    lines.append('')
lines += ['## RA相对两种基线的均值差','', '|套件|比较|ΔAUC|ΔFWT|ΔNBT↓|ΔOldFinal|ΔNewFinal|ΔFinalAll|','|---|---|---:|---:|---:|---:|---:|---:|']
for d in deltas:
    if d['seed']=='mean':lines.append(f"|{d['suite']}|{d['comparison']}|"+'|'.join(fmt(d['metrics'][k]['percent'],signed=True) for k in names)+'|')
lines += ['', '## 表注与数据对应','',
 'OBJECT seed43 的 Random 和 K-Center 均使用 RA 当时冻结绑定的 repeat1；不是本次按分数挑选，不混入首次43或历史42/43/44均值。每run Task6–9各100轮，每单元20回合、总200正式回合，初态身份在三方案间逐项一致。','',
 '该表是历史同协议比较。OBJECT seed41/43 的父基线使用4090，RA使用3090；历史阶段全局动作RNG不保证逐轨迹噪声完全相同。硬件与执行边界沿用原记录，不将所有差值解释为严格单因素因果或统计显著提升。','',
 'AUC为逐任务、逐已学习阶段的层次均值；FWT为对角线均值；NBT为旧任务各后续阶段相对刚学完成绩的下降均值，保留负值。确切公式沿用上一轮核验，见RESULTS.json definitions及recompute.py。','',
 'FULL_METRICS.csv 保存全部18run指标；SUMMARY.csv保存三seed均值和SD；DELTAS.csv保存逐seed与均值差；RESULTS.json保存十单元矩阵、精确分数、源路径和SHA。paper_tables.tex 可直接用于加载booktabs、graphicx的LaTeX稿件。']
(HERE/'TABLES.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

tex=['% Requires \\usepackage{booktabs,graphicx}', '% Metrics in percent; sample SD across seeds41/42/43 (ddof=1).', '% Protocol/hardware provenance is documented in TABLES.md.']
for suite in ['OBJECT','GOAL']:
    tex += [r'\begin{table*}[t]',r'\centering',r'\caption{'+suite+r': replay-selection comparison across seeds 41, 42, and 43. All methods retain MLR+IFA; no ALIGN or execution retention is included. Mean $\pm$ sample SD is reported over three seeds.}',r'\label{tab:'+suite.lower()+r'_selection}',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{llrrrrrr}',r'\toprule',r'Seed & Method & AUC $\uparrow$ & FWT $\uparrow$ & NBT $\downarrow$ & OldFinal $\uparrow$ & NewFinal $\uparrow$ & FinalAll $\uparrow$ \\',r'\midrule']
    for seed in [41,42,43]:
        for method in methods:
            r=next(r for r in runs if (r['suite'],r['seed'],r['method'])==(suite,seed,method))
            tex.append(f'{seed} & {method} & '+' & '.join(fmt(r['metrics'][k]['percent']) for k in names)+r' \\')
        tex.append(r'\midrule')
    for method in methods:
        tex.append('Mean & '+method+' & '+' & '.join(f"${fmt(summary[suite][method][k]['mean_percent'])} \\pm {fmt(summary[suite][method][k]['sample_sd_pp'])}$" for k in names)+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}}',r'\end{table*}','']
(HERE/'paper_tables.tex').write_text('\n'.join(tex).rstrip()+'\n',encoding='utf8')
print(json.dumps(dict(status=out['status'],records=3600,mean_deltas=[x for x in deltas if x['seed']=='mean']),ensure_ascii=False,indent=2))
