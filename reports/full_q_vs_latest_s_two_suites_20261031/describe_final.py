"""Evidence-bound six-Q interpretation and acceptance; no scientific execution."""
from pathlib import Path
import json, hashlib, sys, statistics, datetime
R=Path(__file__).resolve().parent; D=Path(sys.argv[1]).resolve()
js=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
a=js(D/'LOCAL_FINAL_AUDIT.json'); c=js(D/'PRIMARY_S_COMPARISON.json'); v=js(D/'EVIDENCE.json')
assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE' and a['formal_episodes']==1200
assert a['evidence_sha256']==sha(D/'EVIDENCE.json') and c['source_Q_final_sha256']==sha(D/'LOCAL_FINAL_AUDIT.json')
assert c['paired_episodes']==1200 and not v['processes'] and v['light']['controller.exit']['text']=='0'
previous=js(R/'check_readonly_20261101_032200/EVIDENCE.json')
assert previous['light']==v['light'], 'Final CPU audit changed production evidence'
ga=js(R/'GOAL_ACCEPTANCE.json'); gv=js(Path(ga['final_path'])/'EVIDENCE.json')
assert all(v['light'][k]['sha256']==x['sha256'] for k,x in gv['light'].items() if k.startswith(('GOAL_41/','GOAL_42/','GOAL_43/')))
detail={}; summary={}; matrices={}
for suite,ref in c['source_S'].items():
 sd=Path(ref['final_path']); sv=js(sd/'EVIDENCE.json')
 assert sha(sd/'EVIDENCE.json')==ref['evidence_sha256'] and sha(sd/'LOCAL_FINAL_AUDIT.json')==ref['audit_sha256']
 cells=[x for case,z in c['cases'].items() if case.startswith(suite+'_') for x in z['cells'].values()]
 summary[suite]=dict(S_win=sum(x['S_minus_Q_pp']>0 for x in cells),ties=sum(x['S_minus_Q_pp']==0 for x in cells),S_loss=sum(x['S_minus_Q_pp']<0 for x in cells),S_success_Q_fail=sum(len(x['Q_fail_S_success']) for x in cells),Q_success_S_fail=sum(len(x['Q_success_S_fail']) for x in cells))
 matrices[suite]=[]
 for stage in range(6,10):
  for task in range(6,stage+1):
   rows=[c['cases'][f'{suite}_{seed}']['cells'][f'{stage}_{task}'] for seed in (41,42,43)]
   matrices[suite].append(dict(stage=stage,task=task,Q=statistics.mean(x['Q_percent'] for x in rows),S=statistics.mean(x['S_percent'] for x in rows)))
 for seed in (41,42,43):
  case=f'{suite}_{seed}'; qp=case+'/'; sp=f'seed{seed}/'
  qg=json.loads(v['light'][qp+'GRADIENTS_Q.json']['text']); sg=json.loads(sv['light'][sp+'GRADIENTS_S.json']['text'])
  gates=[json.loads(x) for n,y in sv['light'].items() if n.startswith(sp+'GATES_stage') for x in y['text'].splitlines()]
  assert len(gates)==(6700 if suite=='GOAL' else 9100)
  detail[case]=dict(Q_last=qg[-1],S_last=sg[-1],Q_retention_BC_negative=sum(x['retain_vs_bc']['dot']<0 for x in qg),S_retention_BC_negative=sum(x['retain_vs_bc']['dot']<0 for x in sg),Q_support=json.loads(v['light'][qp+'PROBES_9.json']['text'])['support'],S_support=json.loads(sv['light'][sp+'PROBES_9.json']['text'])['support'],S_gate_Q=sum(x['accepted']=='Q' for x in gates),S_gate_blocks=len(gates))
interpret=dict(status='PASS_SIX_Q_INTERPRETATION_SOURCE_RECOMPUTATION',audit_sha256=sha(D/'LOCAL_FINAL_AUDIT.json'),comparison_sha256=sha(D/'PRIMARY_S_COMPARISON.json'),suites=summary,stage_task_means=matrices,diagnostics=detail,all_negative_cells=c['all_negative_cells'],production_light_unchanged=True,accepted_GOAL_preserved=True,limitations=['Three reused development seeds; not independent confirmation or a significance claim','Own caches and trajectories differ; not a matched-state single-step causal contrast','Same accepted updates and interaction count, unequal physical optimizer/FLOPs/wall time','Pretraining provenance incomplete; cannot prove global absence of leakage'])
put(D/'INTERPRETATION_AUDIT.json',interpret)
lines=['# 完整 Q 与最新 S：GOAL / OBJECT 六 run 最终验收与解释','',
'六个 Q 已全部通过实际执行验收和本地独立复算。GOAL 与 OBJECT 各 seed41、42、43，每个 run 400 轮、40 次训练采集、200 回合正式评估；合计 2400 轮、544800 次生产 optimizer.step、240 次采集与 1200 回合正式评估。已有 S 结果只读复用，没有重训或重评。C 没有重新执行，旧机制没有恢复。','',
'完成来源为活动 v2 的真实 controller.exit=0、六个自身 train_Q.exit/evaluate.exit=0、无残留自身进程。135 项冻结文件通过校验，生产记录前缀和最终 CPU 验收前后的全部轻量证据完全相同。最终远端 CPU 验收严格加载 24 个阶段 CP、6 个完整恢复状态、24 个自身成功缓存并重放 teacher，核验 1200 回合实际初态字节与噪声；本地再核算单候选生产计数、标量与指标。验收新增 optimizer=0、环境=0。自己的隔离调用 546、准入 autograd 246、训练方向 autograd 216 均另行记账。','',
'## 结果概览','',
'在这三个开发 seed 上，S 的最终均值在 GOAL 高 4.58 个百分点，在 OBJECT 高 2.08 个百分点。但 OBJECT 的差异方向随 seed 改变：seed41 支持 S，seed42 支持 Q，seed43 最终持平。结果不支持“门控对每个 seed、任务或指标都更好”。GOAL 的 NBT 反而是 Q 更低。','',
'差值均为 S−Q；NBT 越低越好，其余越高越好。AUC 按任务及其后续阶段平均，FWT 是新任务对角均值，NBT 比较各任务初学与后续成绩，OldFinal 是最终阶段 Task6–8 均值，FinalAll 是最终四任务均值。负 NBT 表示按该定义后续平均成绩高于初学成绩，并非负成功率。']
for suite in ('GOAL','OBJECT'):
 lines+=['','## '+suite,'','|指标（%）|Q 无门控|最新 S 有门控|S−Q 百分点|','|---|---:|---:|---:|']
 for k,x in c['suites'][suite].items():lines.append('|%s|%.4f|%.4f|%+.4f|'%(k,x['Q_mean'],x['S_mean'],x['S_minus_Q_pp']))
 lines+=['','|seed|Q 最终|S 最终|S−Q|ΔAUC|ΔFWT|ΔNBT|','|---|---:|---:|---:|---:|---:|---:|']
 for seed in (41,42,43):
  x=c['cases'][f'{suite}_{seed}'];z=x['delta_pp'];lines.append('|%d|%.2f|%.2f|%+.2f|%+.4f|%+.4f|%+.4f|'%(seed,x['Q_metrics']['FinalAll'],x['S_metrics']['FinalAll'],z['FinalAll'],z['AUC'],z['FWT'],z['NBT']))
 z=summary[suite]
 lines+=['',f"30 个阶段×任务×seed 单元中，S 为 {z['S_win']} 胜、{z['ties']} 平、{z['S_loss']} 负。600 个配对回合中，S 成功而 Q 失败 {z['S_success_Q_fail']} 次；Q 成功而 S 失败 {z['Q_success_S_fail']} 次。这里的回合覆盖不同阶段，同一初态重复使用，不能当成 600 个独立新场景。",'', '|阶段/任务|Q 三seed均值|S 三seed均值|S−Q|','|---|---:|---:|---:|']
 for x in matrices[suite]:lines.append('|%d/%d|%.2f|%.2f|%+.2f|'%(x['stage'],x['task'],x['Q'],x['S'],x['S']-x['Q']))
 if suite=='GOAL':
  lines+=['','GOAL 的新任务 Task8 在三个 seed 初学时 S 分别高 10/15/20 点，但最终阶段差值变为 −10/+25/+5。最终 Task6 为 +10/0/+20，Task7 全部持平，Task9 为 0/+5/0。seed41 的最终持平由 Task6 改善与 Task8 退化相抵。S 最终旧任务均值更高，但 NBT 高 0.8333 点，反映其初学成绩也更高，不能将最终优势解释成所有遗忘指标都改善。']
 else:
  lines+=['','OBJECT 的均值优势主要来自 seed41：最终 Task6 从 Q 的 40% 到 S 的 70%，Task7 从 80% 到 90%，Task9 从 95% 到 100%，同时 Task8 从 100% 降到 95%，总均值 +10 点。seed42 则 Q 最终 90%、S 86.25%；S 在最终 Task6/Task9 各低 10 点，Task8 高 5 点，Task7 持平。seed43 最终均值同为 80%，但 S 的 Task6 低 20 点，被 Task7/8/9 的 +5/+10/+5 抵消。','',
  'OBJECT 的新任务收益也不一致：Task7 初学差值为 +10/−20/+10，Task8 为 −5/+5/+20，Task9 为 +5/−10/+5。seed43 虽然 FWT 高 8.75 点、AUC 高 4.8958 点，最终却持平且 OldFinal 更低；其 NBT 也高 8.0556 点。seed42 的 S NBT 更低却最终更差，说明 NBT 的初学基准会影响解释。套件 NBT 均值差 −0.0926 点很小，不能据此声称稳定改善遗忘。']
lines+=['','## 全部 S 低于 Q 的单元','', '|case|阶段/任务|Q|S|S−Q|Q成功S失败|S成功Q失败|','|---|---|---:|---:|---:|---:|---:|']
for x in c['all_negative_cells']:lines.append('|%s|%d/%d|%.1f|%.1f|%+.1f|%d|%d|'%(x['case'],x['stage'],x['task'],x['Q_percent'],x['S_percent'],x['S_minus_Q_pp'],len(x['Q_success_S_fail']),len(x['Q_fail_S_success'])))
lines+=['','以上 13 个负单元全部保留。GOAL 的 4 个负单元以及 OBJECT 的 9 个负单元均未过滤、重试或追加实验；逐初态 SF/FS 索引保存在 PRIMARY_S_COMPARISON.json。OBJECT42 新 Task7 和 OBJECT43 旧 Task6 的 −20 点反例说明当前训练探针门控不保证测试闭环单调改善。','',
'## 训练代理、梯度与缓存分化','', '|case|Q末次BC/执行误差|S末次BC/执行误差|Q/S保持-BC余弦|Q/S保持-执行余弦|S接受Q/总块|','|---|---|---|---|---|---|']
for case,x in detail.items():
 q=x['Q_last'];s=x['S_last'];lines.append('|%s|%.4f / %.6f|%.4f / %.6f|%.4f / %.4f|%.4f / %.4f|%d/%d|'%(case,q['bc_new'],q['exec_new'],s['bc_new'],s['exec_new'],q['retain_vs_bc']['cosine'],s['retain_vs_bc']['cosine'],q['retain_vs_exec']['cosine'],s['retain_vs_exec']['cosine'],x['S_gate_Q'],x['S_gate_blocks']))
lines+=['',
'表中是 Task9 第99轮固定训练 batch 的诊断，不是独立验证，也不是全任务总体误差。GOAL42 的 Q 末次 BC/执行误差更低但闭环更差；GOAL41 的 S 代理更好但最终持平。OBJECT41 的 S 代理更好且最终更高，但保持梯度与 BC/执行梯度都冲突；OBJECT42 的 S 执行误差更低且梯度同向，却最终不如 Q；OBJECT43 的 S 代理略好但最终持平。代理、梯度符号与闭环没有一致的一一对应关系。CPU 对旧专家和缓存保持误差的逐标量复算保存在 RECOMPUTED 文件中，它们同样不是闭环成功率。','',
'S 的门控接受率衡量在自身训练状态和固定训练探针上的候选选择，不是成功概率。Q 没有生产门控。两条完整训练路径会产生不同模型和缓存；即使某个 seed 的 q 数值相同，也不能推断其缓存样本、teacher、随机状态和后续模型相同。Task9 的自身 q（Task6/7/8）如下：']
for case,x in detail.items():lines.append('- %s：Q %s；S %s。'%(case,list(x['Q_support'].values()),list(x['S_support'].values())))
lines+=['','## 成本与可支持的结论','', '|套件三seed|Q生产调用|S生产调用|Q相对S减少|双方接受更新|双方训练采集|','|---|---:|---:|---:|---:|---:|',
'|GOAL|228300|426600|46.48%|228300|120|','|OBJECT|316500|585900|45.98%|316500|120|','|合计|544800|1012500|46.19%|544800|240|','',
'Q/S 每阶段均100轮、同接受更新及原调度，Task6 相同，Task7–9 的 S 为每个更新执行 C/Q 两条候选后保留一条，Q 只执行一条。双方基础准入均546隔离/246 autograd；S 的历史工程等价验证3914 optimizer/18 autograd等保持原账，不计入Q。Q的v1检查错误在任何更新或环境前退出，真实零成本证据保留。本轮不匹配物理optimizer、FLOPs、墙钟或缓存内容；不能把生产调用减少46.19%解释成等比例墙钟提速。','',
'在本开发协议下，完整有门控 S 的套件均值更高，GOAL 的结果方向较一致，OBJECT 则有明显 seed 分歧和抵消。Q 在较少生产优化器调用下得到接近但不完全一致的最终成绩，且部分 seed/任务更好。只有三个开发 seed，不能声称统计显著、普适追上概率、独立确认或把差异唯一归因于门控某一条件。该实验是完整训练路径的门控消融，不是共同中间状态下单步因果检验。','',
'## 数据与验收边界','',
'新增采集使用20–29，正式使用0–19，1200条初态与噪声和已有S实际逐项配对；没有发现新增缓存直接回流正式数据。开发评估集已反复使用，上游预训练来源独立性仍未完全闭合，故不能写成全局无泄露或独立未见测试确认。缓存teacher重放是缓存特征上的一致性验证，不是原始相机或物理环境重放；成功轨迹过滤也不证明每个动作最优。详见 leakage_audit_20261031/REPORT.md。','',
'此前GOAL独立接受文件永久保留，本次全六Q CPU验收核对其轻量SHA和结果完全一致。未增加训练、闭环评估、seed、调参或负结果重试。证据见 final_readonly_20261101_032229 下 LOCAL_FINAL_AUDIT.json、PRIMARY_S_COMPARISON.json、INTERPRETATION_AUDIT.json、原EVIDENCE与标量复算文件。']
report='\n'.join(lines)+'\n'
for p in [R/'FINAL_REPORT.md',D/'FINAL_REPORT.md']:p.write_text(report,encoding='utf-8')
receipt=dict(status='PASS_COMPLETE_LOCAL_ACCEPTANCE_AND_INTERPRETATION',observed=datetime.datetime.now().astimezone().isoformat(),final_path=str(D),evidence_sha256=sha(D/'EVIDENCE.json'),audit_sha256=sha(D/'LOCAL_FINAL_AUDIT.json'),comparison_sha256=sha(D/'PRIMARY_S_COMPARISON.json'),interpretation_sha256=sha(D/'INTERPRETATION_AUDIT.json'),report_sha256=sha(R/'FINAL_REPORT.md'),checkpoints=24,complete_recoveries=6,cache_teacher_replays=24,production_optimizer=544800,isolated_optimizer=546,admission_autograd=246,direction_autograd=216,training_collection=240,formal_episodes=1200,acceptance_optimizer=0,acceptance_environments=0,suite_means=c['suites'],paired_summary=summary,source_S=c['source_S'],prior_GOAL_acceptance_sha256=sha(R/'GOAL_ACCEPTANCE.json'))
put(R/'ACCEPTANCE.json',receipt);put(D/'ACTUAL_ACCEPTANCE_RECEIPT.json',receipt)
print(json.dumps(dict(status=receipt['status'],suites=c['suites'],paired=summary),ensure_ascii=False,indent=2))
