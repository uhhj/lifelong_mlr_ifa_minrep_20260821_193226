"""Recompute all three-lambda metrics from accepted paired outcomes, no new runs."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,statistics,csv
D=Path(__file__).resolve().parent;G=D.parent;W=G.parent
paths={'OBJECT':W/'20261101_lambda_r_object_sensitivity/final_acceptance_20261103_v1/final_readonly_20261103_124306','GOAL':G/'final_readonly_20261104_122613'}
js=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
keys=['AUC','FWT','NBT','OldFinal','NewFinal','FinalAll'];rows=[];sources={};pair_counts={};matrices={}
def metric(out,field):
    r={}
    for s in range(6,10):
        for t in range(6,s+1):
            rr=[x for x in out if x['stage']==s and x['task']==t];assert len(rr)==20 and {x['initial_index'] for x in rr}==set(range(20))
            r[s,t]=F(sum(x[field] for x in rr),20)
    return dict(AUC=sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(r[t,t] for t in range(6,10))/4,NBT=sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(r[9,t] for t in range(6,9))/3,NewFinal=r[9,9],FinalAll=sum(r[9,t] for t in range(6,10))/4),r
for suite,p in paths.items():
    a=js(p/'LOCAL_FINAL_AUDIT.json');c=js(p/'PRIMARY_S_COMPARISON.json')
    assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE' and a['formal_episodes']==1200
    assert c['source_Variant_final_sha256']==sha(p/'LOCAL_FINAL_AUDIT.json') and a['evidence_sha256']==sha(p/'EVIDENCE.json')
    ap=(W/'20261101_lambda_r_object_sensitivity/ACCEPTANCE.json') if suite=='OBJECT' else G/'ACCEPTANCE.json';accepted=js(ap)
    assert accepted['local_final_audit_sha256']==sha(p/'LOCAL_FINAL_AUDIT.json') and accepted['comparison_sha256']==sha(p/'PRIMARY_S_COMPARISON.json')
    sources[suite]={'directory':str(p),'acceptance_sha256':sha(ap),'audit_sha256':sha(p/'LOCAL_FINAL_AUDIT.json'),'comparison_sha256':sha(p/'PRIMARY_S_COMPARISON.json'),'evidence_sha256':a['evidence_sha256']}
    pair_counts[suite]={}
    for group,lam in [('L005',0.05),('L020',0.2)]:
        paired=[];deltas=[]
        for seed in [41,42,43]:
            case=group+'_'+str(seed);v=c['cases'][case];out=v['paired_outcomes'];assert len(out)==200
            m,r=metric(out,'Variant');base,br=metric(out,'S')
            for k in keys:
                if k!='NewFinal':
                    assert str(m[k])==a['cases'][case]['metrics'][k]['fraction']
                    assert abs(float(m[k]*100)-v['Variant_metrics'][k])<1e-10 and abs(float(base[k]*100)-v['S_metrics'][k])<1e-10
            rows.append(dict(suite=suite,lambda_R=lam,seed=seed,**{k:float(m[k]*100) for k in keys}))
            matrices[(suite,lam,seed)]={f'{s}_{t}':float(x*100) for (s,t),x in r.items()}
            if group=='L005':
                rows.append(dict(suite=suite,lambda_R=0.1,seed=seed,**{k:float(base[k]*100) for k in keys}))
                matrices[(suite,0.1,seed)]={f'{s}_{t}':float(x*100) for (s,t),x in br.items()}
            else:
                b=next(x for x in rows if (x['suite'],x['lambda_R'],x['seed'])==(suite,0.1,seed));assert all(abs(b[k]-float(base[k]*100))<1e-10 for k in keys)
            paired+=out;deltas += [float((r[k]-br[k])*100) for k in r]
        pair_counts[suite][str(lam)]=dict(SF=sum(x['Variant'] and not x['S'] for x in paired),FS=sum(not x['Variant'] and x['S'] for x in paired),wins=sum(x>0 for x in deltas),ties=sum(x==0 for x in deltas),losses=sum(x<0 for x in deltas))
means=[];tasks=[]
for suite in paths:
    for lam in [0.05,0.1,0.2]:
        rr=[x for x in rows if x['suite']==suite and x['lambda_R']==lam];assert len(rr)==3
        means.append(dict(suite=suite,lambda_R=lam,**{k:statistics.mean(x[k] for x in rr) for k in keys},FinalAll_sample_sd=statistics.stdev(x['FinalAll'] for x in rr)))
        tasks.append(dict(suite=suite,lambda_R=lam,**{f'Task{t}':statistics.mean(matrices[suite,lam,seed][f'9_{t}'] for seed in [41,42,43]) for t in range(6,10)}))
out=dict(status='PASS_RECOMPUTED_FROM_BOTH_ACCEPTED_SUITES',sources=sources,metric_order=keys,means=means,per_seed=rows,final_tasks=tasks,paired_discordance=pair_counts,scientific_new_runs=0,limits=['Three seeds per suite; no fresh holdout or significance claim.','Historical 0.1 hardware differs; Task6 tensors equal but episode outcomes differ.','Fixed-grid sensitivity, not lambda=0 removal ablation, and no universal optimum selection.'])
(D/'COMPARISON.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
for name,data in [('MEANS.csv',means),('PER_SEED.csv',rows),('FINAL_TASKS.csv',tasks)]:
    with (D/name).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
lines=['# OBJECT与GOAL：三档λ_R完整指标与结论','',
'依据两套件已通过独立验收的完整配对outcomes重新计算。每档每套件seed41/42/43各200正式回合；新增0.05/0.2，0.1直接复用，未新增训练或环境。来源SHA见COMPARISON.json。','',
'## 三seed均值','',
'|套件|λ_R|AUC ↑|FWT ↑|NBT ↓|OldFinal ↑|NewFinal ↑|FinalAll ↑|','|---|---:|---:|---:|---:|---:|---:|---:|']
for x in means:lines.append('|%s|%g|'%(x['suite'],x['lambda_R'])+'|'.join('%.4f'%x[k] for k in keys)+'|')
lines+=['',
'AUC：每任务在可见阶段的成功率先平均，再对任务平均。FWT：此项目定义为任务首次学完的对角成功率，未减去未训练基线，不应与其他论文的不同FWT定义混用。NBT：相对初学成绩的后续平均下降，越低越好，负值表示平均后续改善。OldFinal：最终Task6–8平均；NewFinal：最终Task9；FinalAll：最终Task6–9平均。数值为百分数尺度，差值为百分点。','',
'## 逐seed完整指标','',
'|套件|λ_R|seed|AUC|FWT|NBT|OldFinal|NewFinal|FinalAll|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for x in sorted(rows,key=lambda x:(x['suite'],x['lambda_R'],x['seed'])):lines.append('|%s|%g|%d|'%(x['suite'],x['lambda_R'],x['seed'])+'|'.join('%.4f'%x[k] for k in keys)+'|')
lines+=['','## 最终任务分解','','|套件|λ_R|Task6|Task7|Task8|Task9|','|---|---:|---:|---:|---:|---:|']
for x in tasks:lines.append('|%s|%g|'%(x['suite'],x['lambda_R'])+'|'.join('%.4f'%x[f'Task{t}'] for t in range(6,10))+'|')
lines+=['','## 可以得出的结论','',
'1. 既定λ_R=0.1在这两套件上都取得最高的AUC、FWT和FinalAll均值，为原默认配置的总体表现提供一致的经验支持。将系数减半，OBJECT/GOAL最终分别低2.9167/3.3333pp；加倍则分别低6.6667/2.0833pp。这里是既定网格内的观察，不作普适最优或事后重新选参声明。','',
'2. 结果不支持“对λ基本不敏感”。OBJECT最终均值跨档范围6.6667pp，GOAL为3.3333pp；局部单元差异更大。两个非默认值仍能完成训练并取得较高均值，但平均水平未崩溃不等于稳健平台，更不代表每个seed稳定。GOAL 0.2的最终seed结果70/90/88.75%，样本标准差11.20pp，明显异质；0.1为82.5/87.5/85%，标准差2.50pp。','',
'3. GOAL呈现较明确的保持与新学习取舍：0.2相对0.1的OldFinal高1.1111pp、NBT低6.2037pp，但FWT低6.6667pp、最终Task9低11.6667pp。负均值主要受seed41 Task9的40%对80%影响，seed42/43最终总分反而提高。该模式与更强保持约束限制部分新学习相容，仍不是单机制因果证明。','',
'4. OBJECT没有出现“系数越大、旧任务保持越好”的单调规律。0.2 OldFinal低7.2222pp，主要来自最早Task6由55%降至26.6667%；Task7持平，Task8却提高6.6667pp。因此不能用统一的“强化保持换取更多遗忘保护”解释两个套件。各自缓存、支持度和门控选择形成不同训练路径，影响有明显任务依赖。','',
'5. NBT必须结合FWT与绝对最终成绩。GOAL 0.05的NBT略低，却同时具有较低FWT及OldFinal，不能称保持更好；0.2虽有负NBT，也没有取得最高FinalAll。OBJECT三档NBT均为负，代表相对各自初学成绩的后续改善，不代表无误差，也不宜仅比较NBT正负给两个套件排序。','',
'6. 门控没有让系数的影响消失。OBJECT 0.2的Q接受率50.59%高于0.1的45.33%，但最终成功率更低；GOAL 0.05/0.2的Q接受率45.89%/45.72%接近，任务结果仍不同。局部探针通过率、保持距离或训练损失不能替代闭环评价。当前没有关闭门控或λ=0的新增对照，不能用这次实验独立证明门控或保持项的必要性。','',
'## 结论边界和建议表述','',
'最合适的定位是“固定网格的保持系数敏感性分析”。两套件均观察到原默认0.1具有较好的整体平衡，偏离该值会改变新旧任务表现，且影响依赖任务与seed；不声称无敏感性、所有指标最优或显著优于其他系数。','',
'各档每seed的任务顺序、接受更新预算、双候选生产optimizer调用数和采集次数匹配；成功缓存由各自策略产生，不能视为固定缓存的单步对照。历史0.1硬件与新增run不同；所有Task6模型tensor都相等，但无保持的Task6闭环仍存在差异。因此不能把全部差值纯粹归因于λ，固定协议的工程验收通过也不消除这条因果边界。开发评估集复用、预训练来源未闭合以及每套件仅三个seed进一步限制独立确认和显著性结论。','',
'可用于结果段的表述：在OBJECT和GOAL的预定三档敏感性分析中，默认λ_R=0.1均获得最高的AUC和最终平均成功率。减小或增大保持权重会引起任务依赖的性能变化；增大权重在GOAL上改善了部分旧任务指标，但伴随新任务性能下降，在OBJECT上则未带来一致的保持收益。这些观察支持默认权重在当前开发设置中的总体平衡，并表明局部保持代理或门控接受率不足以预测闭环性能。以上为跨硬件复用基线的开发集描述结果，不作纯λ因果或普适最优断言。','',
'完整源报告与所有正负单元保留在各套件FINAL_REPORT.md与PRIMARY_S_COMPARISON.json。本汇总不修改既有接受记录，不恢复监控，不启动新实验。']
(D/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps(dict(status=out['status'],means=means,pairs=pair_counts),ensure_ascii=False,indent=2))
