"""Recompute reported metrics from accepted lightweight matrices; no remote execution."""
from pathlib import Path
from fractions import Fraction as F
import json
import hashlib
import statistics
import datetime

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
bindings = {}
rows = []
names = ['AUC', 'FWT', 'NBT', 'OldFinal', 'FinalAll']

def read(rel):
    raw = (WORK / rel).read_bytes()
    bindings[rel] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    return json.loads(raw.decode('utf-8-sig'))

def value(x):
    return x['percent'] if isinstance(x, dict) else x

def add(study, suite, method, seed, matrix, expected):
    p = {k: value(v) for k, v in matrix.items()}
    m = {(int(k[0]), int(k[2])): F(str(v))/100 for k, v in p.items()}
    assert len(m) == 10
    a = sum(sum(m[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4
    f = sum(m[t,t] for t in range(6,10))/4
    n = sum(sum(m[t,t]-m[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3
    o = sum(m[9,t] for t in range(6,9))/3
    z = sum(m[9,t] for t in range(6,10))/4
    metrics = {k: float(v*100) for k,v in zip(names,[a,f,n,o,z])}
    for k,v in expected.items():
        assert abs(metrics[k]-value(v)) < 1e-8, (study,method,seed,k)
    rows.append(dict(study=study,suite=suite,method=method,seed=seed,matrix_percent=p,metrics_percent=metrics))

def stats(group):
    return {k: {'mean': statistics.mean(r['metrics_percent'][k] for r in group),
                'sample_sd': statistics.stdev(r['metrics_percent'][k] for r in group) if len(group)>1 else None}
            for k in names}

def table(headers, body):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(str(c) for c in r)+' |' for r in body])+'\n'

def main():
    early = read('execution/20261017_five_seed_summary/RESULTS.json')
    assert early['audit']=='PASS'
    rel='execution/20261017_five_seed_summary/REPORT.md'
    raw=(WORK/rel).read_bytes()
    assert 'LIBERO-OBJECT' in raw.decode('utf-8')
    bindings[rel]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    for r in early['first_runs']:
        mat={f'{s}_{t}':r['matrix'][s-6][t-6] for s in range(6,10) for t in range(6,s+1)}
        add('early_selection','OBJECT',r['method'],r['seed'],mat,r['metrics'])
    for suite in ['GOAL','OBJECT']:
        b=read(f'execution/20261025_ccbp_v4_full/{suite}_THREE_SEED_BASELINES.json')
        for r in b['runs']:
            add('baseline',suite,r['method'],r['seed'],r['matrix'],r['metrics'])
    v1=read('execution/20261024_ccbp/final_readonly_20261024_224310/FINAL_ANALYSIS.json')
    v1audit=read('execution/20261024_ccbp/final_readonly_20261024_224310/LOCAL_FINAL_AUDIT.json')
    for r in v1['matrix_percent'].values():
        assert r['mean']==(r['task8']+r['task9'])/2
    single = {
      'v2':'execution/20261024_ccbp_v2_full/final_readonly_20261025_054702/LOCAL_FINAL_AUDIT.json',
      'v3':'execution/20261025_ccbp_v3_full/final_readonly_20261025_181226/LOCAL_FINAL_AUDIT.json',
      'DOS':'execution/20261025_ccbp_v4_full/final_readonly_20261026_014654/LOCAL_FINAL_AUDIT.json'}
    costs={}
    for study,path in single.items():
        d=read(path)
        costs[study]={'training':d['training'],'evaluation':d['evaluation']}
        for arm in d['matrix']:
            add(study,'GOAL',arm,41,d['matrix'][arm],d['metrics'][arm])
    current={
      'GOAL':'execution/20261029_support_gate_full_s_goal/final_readonly_20261030_111546/',
      'OBJECT':'execution/20261028_support_gate_full_s_object/final_readonly_20261029_201235/'}
    for suite,base in current.items():
        d=read(base+'PRIMARY_C_COMPARISON.json')
        read(base+'LOCAL_FINAL_AUDIT.json')
        for seed,r in d['seeds'].items():
            for arm,label in [('C','C_normal'),('S','S_latest')]:
                add('current',suite,label,int(seed),{k:v[arm+'_percent'] for k,v in r['cells'].items()},r[arm+'_metrics'])
    comparisons={}
    for arm,base in [('C','execution/20261030_equal_optimizer_budget_cs_two_suites/final_readonly_20261031_065009/'),
                     ('Q','execution/20261031_full_q_vs_latest_s_two_suites/final_readonly_20261101_032229/')]:
        d=read(base+'PRIMARY_S_COMPARISON.json')
        read(base+'LOCAL_FINAL_AUDIT.json')
        comparisons[arm]={}
        for case,r in d['cases'].items():
            suite,seed=case.split('_')
            label='C_equal_budget' if arm=='C' else 'Q'
            add('current',suite,label,int(seed),{k:v[arm+'_percent'] for k,v in r['cells'].items()},r[arm+'_metrics'])
            existing=next(x for x in rows if x['study']=='current' and x['suite']==suite and x['method']=='S_latest' and x['seed']==int(seed))
            assert existing['matrix_percent']=={k:v['S_percent'] for k,v in r['cells'].items()}
            comparisons[arm][case]=r['delta_pp']
    groups={}
    for r in rows:
        key='/'.join([r['study'],r['suite'],r['method']])
        groups.setdefault(key,[]).append(r)
    summaries={key:stats(group) for key,group in groups.items()}
    report=['# 已完成机制与设计对照实验统计（截至2026-11-01）','',
      '本次只读取已接受的本地轻量矩阵并重新计算指标，没有训练、仿真或远程检查。表中单位为%；差值为百分点。多seed表为种子均值，完整逐seed矩阵和样本标准差（ddof=1）见DATA.json。',
      '', 'AUC沿用项目的“各任务学习后各阶段均值，再跨任务平均”；FWT沿用项目的首次学会时对角线均值定义，不是相对零样本基线的迁移增益。NBT为后续相对首次学会时的下降，越低越好，可为负值；OldFinal为最终三个旧任务均值，FinalAll为最终四任务均值。',
      '', '## 1. 早期选样比较：OBJECT，seed40–44首次运行', '']
    body=[]
    for arm in ['random','coreset','event_coreset']:
        st=summaries['early_selection/OBJECT/'+arm]
        body.append([arm]+[f"{st[k]['mean']:.2f} ± {st[k]['sample_sd']:.2f}" for k in ['AUC','FWT','NBT']])
    report += [table(['方法','AUC↑','FWT↑','NBT↓'],body),
      '共15个首次运行、3000正式回合；seed43另一次重复的600回合不纳入此均值。该五seed集合与后续三seed基线的取数口径不同，不能直接当成RA的单变量对照。',
      '', '## 2. 原方法→RA→RA+ALIGN：seed41、42、43', '']
    body=[]
    for suite in ['GOAL','OBJECT']:
        for arm in ['MLR_IFA','RA','RA_ALIGN']:
            st=summaries[f'baseline/{suite}/{arm}']
            body.append([suite,arm]+[f"{st[k]['mean']:.2f}" for k in names])
    report += [table(['任务套件','方法']+names,body),
      '历史3方法×2套件×3seed，共18个200回合结果。OBJECT seed43原基线沿用冻结repeat1；本轮不重新挑选结果。历史基线不保证与后续S逐轨迹动作噪声相同。',
      '', '## 3. CCBP-v1：共同Stage8起点，仅训练Task9，GOAL seed41','']
    report += [table(['方法','最终Task8','Task9','两任务平均'],[[a]+[v1['matrix_percent'][a][k] for k in ['task8','task9','mean']] for a in ['A','B','C','D']]),
      'A=ALIGN；B=固定执行Normal蒸馏；C=持续专家cost+原margin路由；D=教师误差预算+可接受集合路由。共400epoch、134800主更新、160正式回合。这轮没有完整序列AUC，不能与下一表的AUC互换。',
      '', '## 4. 完整GOAL seed41设计对照','']
    body=[]
    for study in ['v2','v3','DOS']:
        for r in rows:
            if r['study']!=study: continue
            body.append([study,r['method']]+[f"{r['matrix_percent'][f'9_{t}']:.0f}" for t in range(6,10)]+[f"{r['metrics_percent'][k]:.2f}" for k in names])
    report += [table(['轮次','方法','最终T6','最终T7','最终T8','T9']+names,body),
      'v2：B固定教师蒸馏；C专家监督+原margin；N专家监督+代价差softplus路由；V混合评分J+新路由。',
      'v3：C为该轮实际强对照；R保留J、回退margin；G专家监督+纠错线段锚+原margin。v2和v3的C是独立运行，不混并。',
      'DOS：D示范历史锚；O实际全轨迹历史锚；S成功轨迹历史锚，均固定任务获得时教师。这里S是早期未门控版本；C复用v3，不增加一次实验。D/O/S各40训练采集，共120；缓存窗口为512/512/499（S的Task8为115），无补采。',
      '', '## 5. 最新完整对照：GOAL与OBJECT各seed41–43','']
    body=[]
    for suite in ['GOAL','OBJECT']:
        for arm in ['C_normal','C_equal_budget','Q','S_latest']:
            st=summaries[f'current/{suite}/{arm}']
            body.append([suite,arm]+[f"{st[k]['mean']:.2f}" for k in names])
    report += [table(['套件','方法']+names,body),
      'C_normal=原主更新预算强C；C_equal_budget=匹配S正式优化器调用数的C；Q=关闭门控的完整对照；S_latest=最新门控S。相同S结果跨三类对照只引用一次，不算三套新S实验。',
      '', '最新S相对各对照（均值差，百分点）：','']
    body=[]
    for suite in ['GOAL','OBJECT']:
        st=summaries[f'current/{suite}/S_latest']
        for arm in ['C_normal','C_equal_budget','Q']:
            ref=summaries[f'current/{suite}/{arm}']
            body.append([suite,'S - '+arm]+[f"{st[k]['mean']-ref[k]['mean']:+.2f}" for k in names])
    report += [table(['套件','比较']+names,body),
      '', '## 6. 可以从对照中读出的结论','',
      '- 原教师预算D不如B/C；单seed结果支持继续采用持续专家监督，但不证明某个单步失败的因果机制。',
      '- v2中N相对C：AUC -4.375pp、Task9 -15pp，新路由没有体现收益。V相对N的增益同时含教师、评分缩放和后续教师差异，不能全归因于一个权重。',
      '- v3中G相对C：AUC -4.375pp，最终Task8 -20pp、Task9 -5pp；方向锚没有通过预先登记的探索门。',
      '- DOS中O相对D：AUC +9.1667pp、FinalAll +7.5pp；S相对O：AUC -0.625pp、FinalAll +5pp。成功筛选体现保持收益，但不能宣称所有指标均提升，也非完全相同历史池上的纯筛选因果实验。',
      '- 最新S相对等优化器预算C：GOAL/OBJECT的AUC分别+3.5417/+5.5903pp，说明仅增加C的更新数未解释全部优势；这不是等墙钟或等交互对照。',
      '- 最新S相对完整Q：GOAL/OBJECT AUC分别+3.75/+1.875pp；六个seed-suite中AUC为5胜1负，FinalAll为3胜2平1负，仍保留13个负向阶段任务单元。S正式优化器调用1012500，Q为544800，不能把均值差解释为等计算下的纯门控增益。',
      '', '## 7. 规模与去重说明','',
      table(['实验轮次','新增训练分支','训练epoch','主更新/正式优化器调用','训练采集','正式回合'],[
       ['v1 Task9分叉',4,400,134800,0,160],['v2完整',4,1600,304400,0,800],
       ['v3完整',3,1200,228300,0,600],['DOS完整',3,1200,228300,120,600],
       ['最新S，两套件三seed',6,2400,'544800主更新 / 1012500物理调用',240,1200],
       ['等优化器预算C，两套件三seed',6,4200,1012500,0,1200],
       ['完整Q，两套件三seed',6,2400,544800,240,1200]]),
      '此处列各轮规模，不把重用基线相加为全项目唯一总数；早期RA/ALIGN、普通C扩展及旧S扩展有独立账目。隔离准入、额外autograd、诊断和失败版本成本不混入正式主更新；完整审计原记录保留。本汇总没有将取消或未完成的分叉列为已完成科学结果。',
      '', '所有seed41及既有初态均经历反复探索；三seed均值不等于独立确认结论。历史版本/硬件和训练交互成本见各原协议。',
      '', '## 8. 数据来源（SHA绑定）','']
    report += ['- `'+p+'`' for p in bindings]
    data={'status':'PASS_LOCAL_METRIC_RECOMPUTATION',
      'observed_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
      'scope':'Accepted lightweight matrices only; no new training, rollout or remote audit',
      'source_bindings':bindings,'rows':rows,'summary':summaries,
      'v1_two_task_results':v1['matrix_percent'],'latest_paired_differences':comparisons,
      'matrix_rows_recomputed':len(rows),'reuse_not_unique_run_count':True,'sample_sd_ddof':1}
    (HERE/'DATA.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (HERE/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps({'status':data['status'],'matrix_rows':len(rows),'sources':len(bindings)}))

if __name__=='__main__': main()
