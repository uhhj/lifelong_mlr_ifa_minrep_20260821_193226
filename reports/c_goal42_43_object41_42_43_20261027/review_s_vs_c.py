"""Read-only recomputation of the accepted S-vs-C ablation; no new experiments."""
from pathlib import Path
import json,hashlib,statistics,datetime
R=Path(__file__).resolve().parent;D=R/'final_readonly_20261027_222119';A=R/'s_vs_c_review_20261027';A.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
c=json.loads((D/'COMPARISON.json').read_text(encoding='utf8'));assert c['status']=='PASS_ACCEPTED_COMPARISON'
result={}
for suite in ['GOAL','OBJECT']:
 runs={m:{s:c['C'][suite+'_'+s] if m=='C' else c['S'][suite][s] for s in ['41','42','43']} for m in ['C','S']}
 summaries={m:{k:statistics.mean(r['metrics'][k]['percent'] for r in rr.values()) for k in ['AUC','FWT','NBT','OldFinal','FinalAll']} for m,rr in runs.items()}
 for m in runs:
  for k in summaries[m]:assert abs(summaries[m][k]-c['summary'][suite][m][k]['mean'])<1e-10
 cells={m:{cell:statistics.mean(r['matrix'][cell]['percent'] for r in rr.values()) for cell in ['7_6','8_8','9_6','9_7','9_8','9_9']} for m,rr in runs.items()}
 nbt={m:{str(t):statistics.mean(sum(r['matrix'][f'{t}_{t}']['percent']-r['matrix'][f'{st}_{t}']['percent'] for st in range(t+1,10))/(9-t) for r in rr.values()) for t in [6,7,8]} for m,rr in runs.items()}
 for m in runs:assert abs(statistics.mean(nbt[m].values())-summaries[m]['NBT'])<1e-10
 pairs={}
 for label,stages in [('final',[9]),('all',[6,7,8,9])]:
  vv=[v for case,cc in c['paired_C_minus_S'].items() if case.startswith(suite) for cell,v in cc.items() if int(cell.split('_')[0]) in stages]
  pairs[label]=dict(S_improves=sum(len(x['S_success_C_failure']) for x in vv),S_worsens=sum(len(x['S_failure_C_success']) for x in vv))
 result[suite]=dict(means=summaries,S_minus_C={k:summaries['S'][k]-summaries['C'][k] for k in summaries['C']},cells=cells,NBT_task_components=nbt,paired=pairs,seed_AUC_deltas={s:runs['S'][s]['metrics']['AUC']['percent']-runs['C'][s]['metrics']['AUC']['percent'] for s in runs['C']})
out=dict(status='PASS_ACCEPTED_DATA_RECOMPUTATION',source_sha256=sha(D/'COMPARISON.json'),suites=result,new_models=0,new_optimizer_calls=0,new_environments=0,interpretation='Mixed evidence: localized retention benefits, no stable overall advantage; no independent success-filter causal isolation',limits=['Three existing development seeds','S adds 40 training collection episodes per seed and retention compute','FWT retains project diagonal acquisition definition','No retrospectively invented OBJECT pass/fail gate'])
(A/'AUDIT.json').write_text(json.dumps(out,indent=2),encoding='utf8')
# Preserve the earlier prose arithmetic error and its hashes before correction.
corrections=[]
for p in [D/'MECHANISM_ANALYSIS.md',D/'REPORT.md',R/'REPORT.md']:
 text=p.read_text(encoding='utf8');old='C最终新Task9均值98.3333高于S86.6667';new='C最终新Task9均值98.3333高于S83.3333'
 assert text.count(old)==1
 target=A/('prior_'+('root_' if p.parent==R else 'final_')+p.name);target.write_bytes(p.read_bytes())
 before=sha(p);p.write_text(text.replace(old,new),encoding='utf8');corrections.append(dict(file=str(p.relative_to(R)),before_sha256=before,after_sha256=sha(p),preserved_original=str(target.relative_to(R)),old_value=86.6667,correct_value=83.3333,reason='S OBJECT final Task9: (55+95+100)/3. Existing COMPARISON.json and accepted primary metrics were already correct.'))
(A/'PROSE_CORRECTION.json').write_text(json.dumps(dict(status='CORRECTED_DERIVED_PROSE_ONLY',changes=corrections,accepted_metrics_unchanged=True),indent=2),encoding='utf8')
lines=['# S 对 C：消融结论复核','','结论：局部正向、整体混合，不能作为S在两个suite上稳定优于C的完整正向消融。C/S表示相同C学习主干去除/保留成功轨迹保持路径的整体比较；不是单独成功筛选的因果消融。','', '|Suite|Method|AUC↑|FWT↑|NBT↓|OldFinal↑|FinalAll↑|','|---|---|---:|---:|---:|---:|---:|']
for suite,r in result.items():
 for m in ['C','S']:lines.append('|'+suite+'|'+m+'|'+'|'.join(f'{r["means"][m][k]:.4f}' for k in ['AUC','FWT','NBT','OldFinal','FinalAll'])+'|')
lines+=['','单位为百分数，seed41/42/43均值。FWT沿用项目对角线学习成绩口径，非未训练任务零样本迁移定义。','',
'GOAL：S的AUC仅+0.2778pp，OldFinal持平，FinalAll+2.5pp，FWT-1.25pp。NBT降低4.2593pp且每seed都更低，但Task8刚学完成绩C为90/85/80，S为80/60/70；均值由85降为70，最终两者均75。Task8的NBT分量从C的10变成S的-5，这一项使总NBT降低5pp，其他两任务合计抵消约0.7407pp。故不能把总NBT下降全部解读成相同学习水平下保持更强。新Task9平均S高10pp，但seed42低15pp、seed43高40pp，收益不稳定。',
'OBJECT：S的AUC仅+0.0347pp，FinalAll完全持平，OldFinal+5pp，但最终Task9从C98.3333%降至S83.3333%，下降15pp。三旧任务各自平均的净增益合计15pp，恰被新任务的15pp损失抵消；主要负例是seed41 Task9从100%降到55%。旧Task6最终S相对C为+15/+30/-10pp，并非每seed均提高。NBT反而增加1.0185pp（变差），原因需结合不同对角线参照和全阶段轨迹解释。',
'配对结果：最终阶段GOAL有23回合S成功/C失败、17回合反向；OBJECT为22/22。全阶段GOAL为39/44，OBJECT为62/42。全阶段回合数各格等权，而AUC沿用按任务寿命平均的权重，两者不必同向。回合不能当作数百个独立训练seed进行显著性论证。',
'原GOAL seed41预登记探索门要求OldFinal严格提高且最终Task8、Task9、早期7_6不下降；S早期7_6为70、C为80，原未过门结论不变。后续扩展没有重新登记三seed或OBJECT的C比较成功门，不能事后宣称通过新门。描述性地看，GOAL平均OldFinal未提高，OBJECT新任务下降，也不支持“更好保持且不损学习”的整体目标。',
'离线补充：36个共同专家窗口单元中有8个出现误差与闭环方向相反。OBJECT42最终Task6，S专家误差0.008895高于C0.002858，闭环却65%高于35%；这支持回放误差不是充分代理，不证明成功筛选单独有效。S与C还同时差异于额外40训练交互/seed、访问状态、固定教师和保持损失。成功筛选本身需S与O对照，本轮C/S三seed不能独立归因。',
'建议结论措辞：加入成功轨迹行为保持后，部分旧任务和阶段获得改善，但三seed平均AUC基本持平，收益依赖任务与seed，并出现新任务学习损失；当前结果尚不足以证明该模块带来稳定的整体优势。保留C为强对照，S作为有取舍的候选，不自动追加实验或扫描。',
'本次同时纠正前报告一处文字算术错误：OBJECT S最终Task9均值应为83.3333%，此前正文写86.6667%；底层逐seed矩阵、COMPARISON和主指标一直正确，原文及前后SHA已保留。']
(A/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
closure=D/'CLOSURE_AUDIT.json';x=json.loads(closure.read_text(encoding='utf8'));x.update(mechanism_analysis_sha256=sha(D/'MECHANISM_ANALYSIS.md'),report_sha256=sha(D/'REPORT.md'),later_ablation_review_sha256=sha(A/'REPORT.md'),prose_correction_sha256=sha(A/'PROSE_CORRECTION.json'));closure.write_text(json.dumps(x,indent=2),encoding='utf8')
p=R/'RUN_STATUS.json';x=json.loads(p.read_text(encoding='utf8'));x.update(closure_sha256=sha(closure),latest_analysis=str((A/'REPORT.md').relative_to(R)));p.write_text(json.dumps(x,indent=2),encoding='utf8')
print(json.dumps(out,indent=2))
