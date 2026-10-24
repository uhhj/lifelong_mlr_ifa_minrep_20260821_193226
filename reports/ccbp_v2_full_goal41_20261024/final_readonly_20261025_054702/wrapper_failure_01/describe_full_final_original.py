"""Describe accepted scalar evidence, preserving all adverse results and limits."""
import sys,json,hashlib,statistics
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);js=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=js(D/'LOCAL_FINAL_AUDIT.json');v=js(D/'EVIDENCE.json');assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE'
light={n:json.loads(x['text']) if n.endswith('.json') else x['text'] for n,x in v['light'].items()}
names=dict(B='固定教师蒸馏',C='专家成本监督',N='新版路由、无教师锚',V='完整v2、软教师锚')
details={};identities={};gradients={}
for arm in 'BCNV':
 gradients[arm]={}
 for stage in [7,8,9]:
  p=D/'diagnostics'/f'{arm}_{stage}.json';rows=js(p);identity=[(x['index'],x['task'],x['sample_sha256']) for x in rows]
  assert identities.setdefault(stage,identity)==identity
  per={}
  for task in range(6,stage):
   selected=[x for x in rows if x['task']==task];experts=[];regrets=[];anchors=[];weights=[];teacher=[];score_regrets=[]
   for x in selected:
    c=np.array(x['student_component_group_costs']).sum(-1);d=np.array(x['student_component_anchor_group_distances']).sum(-1);k=x['student_mode'];w=x['quality_weight']
    score=(c+w*d)/(1+w) if arm=='V' else d if arm=='B' else c
    experts.append(c[k]);regrets.append(c[k]-min(c));anchors.append(d[k]);weights.append(w);teacher.append(x['frozen_teacher_cost']);score_regrets.append(score[k]-min(score))
   per[str(task)]=dict(count=len(selected),mean_expert_cost=float(np.mean(experts)),p90_expert_cost=float(np.percentile(experts,90)),max_expert_cost=float(max(experts)),mean_expert_selection_regret=float(np.mean(regrets)),nonminimum_expert_selection_count=sum(x>1e-12 for x in regrets),mean_own_teacher_anchor=float(np.mean(anchors)),mean_own_score_regret=float(np.mean(score_regrets)),own_teacher_mean_cost=float(np.mean(teacher)),quality_weight=dict(min=float(min(weights)),median=float(np.median(weights)),max=float(max(weights))))
  details[f'{arm}_{stage}']=per
  g=[x for x in light[f'GRADIENTS_{arm}.json'] if x['stage']==stage]
  gradients[arm][str(stage)]={k:dict(mean_l2=float(np.mean([x[k]['l2'] for x in g])),max_l2=max(x[k]['l2'] for x in g),mean_ratio_to_BC=float(np.mean([x[k]['l2']/max(x['bc']['l2'],1e-30) for x in g])),samples=len(g)) for k in ['ifa_weighted','exec_weighted','route_weighted']}
metrics=a['metrics'];matrix=a['matrix'];delta={}
for ref in 'BCN':delta['V-'+ref]={k:metrics['V'][k]['percent']-metrics[ref][k]['percent'] for k in metrics['V']}
peaks=[]
for arm in 'BCNV':
 for task in [6,7,8]:
  old=max(matrix[arm][f'{stage}_{task}']['percent'] for stage in range(task,10));final=matrix[arm][f'9_{task}']['percent']
  if final<old:peaks.append(dict(arm=arm,task=task,peak=old,final=final,delta_pp=final-old))
events=[json.loads(x) for x in light['EVAL_EVENTS.jsonl'].splitlines() if x.strip()];times={arm:sum(x['seconds'] for x in events if x['kind']=='complete' and x['run'].startswith(arm+'_'))/60 for arm in 'BCNV'}
cost=dict(training_epoch_minutes={arm:a['training']['epoch_seconds'][arm]/60 for arm in 'BCNV'},evaluation_vector_minutes=times,eval_cache_admission=light['EVAL_ADMISSION.json'],final_cpu=a['final_cpu'],teacher_windows=10820,teacher_batches=a['training']['teacher_forward_batches'],main_updates=304400,own_isolated_updates=70,formal_extra_autograd=192,admission_extra_autograd=48,old_superseded_fork=dict(complete_epochs=9,main_updates=3033,environments=0,not_part_of_full_run=True),earlier_full_task6_inherited=True,completed_epoch_repeats=0)
out=dict(status='PASS_ACCEPTED_DESCRIPTIVE_ANALYSIS',main_exploratory_criterion=a['scientific_main_exploratory_criterion'],metric_differences_pp=delta,common_window_diagnostics=details,sampled_gradient_scales=gradients,own_peak_declines=peaks,all_comparison_negative_cells=a['all_negative_cells'],costs=cost,limitations=a['limits'],final_audit_sha256=sha(D/'LOCAL_FINAL_AUDIT.json'))
(D/'COMPARISON_AND_MECHANISM.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf8')
lines=['# CCBP-v2 GOAL seed41：四组完整序列最终报告','',f"实验执行和独立验收完成。预登记主探索条件（V最终Task8严格高于C且Task9不低于C）：**{'通过' if a['scientific_main_exploratory_criterion'] else '未通过'}**。验收PASS不等于方法有效；不自动扩seed或扫描权重。",'', '四组分别从官方GOAL模型开始独立训练完整Task6→7→8→9，各400epoch76100次更新；不是共同Stage8的单阶段分叉。RA-Coreset成员、ALIGN标签与mask、BC/IFA、模型及部署保持，Task7起在全部合法旧窗口施加各组辅助。','', '|组|方法|AUC %|FWT %|NBT pp（低好）|OldFinal %|FinalAll %|','|---|---|---:|---:|---:|---:|---:|']
for arm in 'BCNV':lines.append('|'+arm+'|'+names[arm]+'|'+'|'.join(f"{metrics[arm][k]['percent']:.6f}" for k in ['AUC','FWT','NBT','OldFinal','FinalAll'])+'|')
lines+=['','AUC为先对每个任务从学会阶段到最终的成功率取均值，再对四任务等权平均；并非800回合总体成功率。NBT按原定义计算，较低更好。精确分数见LOCAL_FINAL_AUDIT。','', '|训练阶段／评估任务|B %|C %|N %|V %|','|---|---:|---:|---:|---:|']
for stage in range(6,10):
 for task in range(6,stage+1):lines.append(f'|{stage} / {task}|'+ '|'.join(f"{matrix[arm][f'{stage}_{task}']['percent']:.0f}" for arm in 'BCNV')+'|')
lines+=['','每格20初态，同任务／初态跨组和跨阶段复用逐轨迹Normal噪声；单格一回合即5pp。沿用已探索seed41与初态，不是独立确认试验；本轮配对噪声不同于历史globalRNG评估，不将历史RA/ALIGN直接作为同协议因果对照。','', '## 配对改变与副作用','', '|比较（候选−参考）|最终任务|差值pp|参考成功→候选失败|参考失败→候选成功|','|---|---:|---:|---|---|']
for comp in ['V-C','V-N','V-B']:
 for task in range(6,10):
  x=a['paired'][comp][f'9_{task}'];lines.append(f"|{comp}|{task}|{x['delta_pp']}|{x['reference_success_to_candidate_failure']}|{x['reference_failure_to_candidate_success']}|")
lines+=['','全部阶段负单元：'+json.dumps(a['all_negative_cells'],ensure_ascii=False),'','各组自身峰值至最终下降：'+json.dumps(peaks,ensure_ascii=False),'','## 离线代理与闭环','', '以下使用Stage9相同旧窗口、相同专家与尺度；专家cost包含方差，属于动作单位归一化代理，不能换算成毫米或成功概率。','', '|组|旧任务|窗口数|执行专家cost均值|P90|专家选择regret均值|非最低专家cost选择数|','|---|---:|---:|---:|---:|---:|---:|']
for arm in 'BCNV':
 for task in [6,7,8]:
  x=details[f'{arm}_9'][str(task)];lines.append(f"|{arm}|{task}|{x['count']}|{x['mean_expert_cost']:.6g}|{x['p90_expert_cost']:.6g}|{x['mean_expert_selection_regret']:.6g}|{x['nonminimum_expert_selection_count']}|")
lines+=['','四组后来教师各自随训练改变，因此教师距离和J并非跨组相同标尺；只能把共同专家cost直接对照。N−C反映新版路由相对旧路由的整体增量；V−N反映加入软锚及其后续教师变化的整体增量。没有固定w=1组，不能独立归因于一致度权重。配对SF/FS仍是结果配对，不是同物理历史的动作替换实验，不能断言某一次分量切换造成失败。','', '梯度诊断仅每旧阶段epoch0/9/49/99首个合法旧batch的样本，不代表整段平均或完整梯度方向；详见COMPARISON_AND_MECHANISM，名义相同lambda不意味着等梯度强度。','', '## 成本、恢复与验收','', json.dumps(cost,ensure_ascii=False,indent=2),'', '实际16个CPU严格100键模型、四个完整最终恢复、全部教师前阶段源与缓存、原成员与标签、800真实初态字节和noiseSHA、1600epoch人口账目及完整40cell独立复算通过。最终CPU仅诊断21640窗口／688批调用，0优化器／0环境。所有工程失败与外置包装失败如有均留证；失败不能改写为科学负结果。早期B Task6完整继承，未重训已完成epoch；旧三份中间恢复经明确授权删除，最终资产与审计证据保留，已退休的历史中间恢复不能再重演。']
(D/'FULL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps(dict(status=out['status'],main_criterion=out['main_exploratory_criterion'],differences=delta,stage9_diagnostics={arm:details[f'{arm}_9'] for arm in 'BCNV'},costs={k:cost[k] for k in ['training_epoch_minutes','evaluation_vector_minutes']}),indent=2))
