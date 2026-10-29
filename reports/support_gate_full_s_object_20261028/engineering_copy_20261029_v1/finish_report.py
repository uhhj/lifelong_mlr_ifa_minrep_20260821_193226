"""Save reviewed interpretation and accepted completion metadata. No remote work."""
from pathlib import Path
import datetime, hashlib, json
N=Path(__file__).resolve().parent.parent; D=N/'final_readonly_20261029_201235'
js=lambda p:json.loads(p.read_text(encoding='utf8'))
put=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
a=js(D/'LOCAL_FINAL_AUDIT.json');i=js(D/'INTERPRETATION_AUDIT.json');s=js(D/'RESULT_SUMMARY.json');v=js(D/'EVIDENCE.json');p=js(D/'LOCAL_AUDIT.json')
assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE' and p['prefix_checked']=='check_readonly_20261029_201114'
now=datetime.datetime.now().astimezone().isoformat()
text=(N/'FINAL_REPORT.md').read_text(encoding='utf8')
text=text.replace('报告发布前仍需人工核对解释文字：逐项说明新旧任务变化与负结果，不能仅凭自动表格宣布机制有效。','下文已结合完整闭环矩阵、配对变化、当前训练代理、保持梯度及独立CPU诊断完成解释；执行验收不等于方法优势证明。')
text+='''

## 结果解释

三个seed的训练、采集、600正式回合及独立验收全部完成。最终四任务平均成功率为85.00%，AUC为86.4931%，旧三任务最终均值81.1111%。相对已验收的历史原S，FinalAll从82.50%升至85.00%（+2.50个百分点）、AUC从85.0347%升至86.4931%（+1.4583），但OldFinal从82.2222%降至81.1111%（−1.1111），获得阶段均值FWT也下降0.8333个百分点。因此本轮没有证明旧任务保持整体改善，更不能仅凭FinalAll上升就宣称支持度门控全面有效。历史原S的训练方法、实际缓存及算力不同，没有完成独立机制归因对照，差值只作描述，不称因果或显著。

seed41最终Task6–9为70/90/95/100%，Task9较历史原S的55%明显提高；seed42为65/100/90/90%，最终Task8和Task9分别比原S低10和5个百分点。seed43为30/100/90/100%，Task6仍是突出失败：从获得时70%降至30%，11条成功转失败、3条失败转成功；仅Stage8→9就从65%降至30%，9 SF、2 FS。即使相对历史原S的20%提高10个百分点，也不能称已解决遗忘。seed42 Task8从100%降至90%，2 SF、0 FS。seed41 Task7和Task8的最终成功率虽与前阶段相同，各有1 SF和1 FS抵消，成功率不变不代表轨迹行为完全保持。

新任务训练代理均有下降，但它们不提供闭环保证。例如seed43 Task9固定当前训练探针BC从−16.5098降至−32.6183，执行专家成本从0.501539降至0.012393，新任务闭环100%，同时旧Task6下降35个百分点。该阶段支持加权旧保持探针从0.033253升至0.034262；Task6分量从0.008688升至0.011304。门控比较的是同一块的Q与C候选，并不要求被选轨迹相对块开始或整阶段旧误差单调下降；C更新也可能改变旧任务。因此零容差相对门控通过与跨阶段遗忘可以同时发生。

seed43 Task9有1259/3700块接受Q（34.03%），其余接受C；seed41、42同阶段分别1517/3700（41.00%）、1559/3700（42.14%）。接受率是训练决策统计，不是成功率。seed43 Task6支持度q=.6、缓存76窗口，比seed41/42的q=.2、26窗口更多，最终旧任务仍最弱；不能把覆盖数量或q当作闭环成功概率，也不能认定稀疏缓存是唯一原因。每个旧任务最多8个固定训练探针窗口，其覆盖并不代表后续正式访问的全部状态。

每阶段只在4个固定训练位置记录梯度方向。seed43 Task9保持梯度与当前BC、执行成本梯度分别3/4、2/4次点积为负，与局部目标冲突相容；seed42 Task9两者都是0/4负点积，却仍有Task8闭环退化。局部梯度方向不是轨迹失败的充分解释。独立CPU最终诊断中，seed43 Task6在761个旧专家合法窗口的执行成本均值0.002241、regret均值0.000364，而闭环仅30%；成功缓存76窗口的保持距离均值0.081197。两类窗口和尺度含义不同，不能将它们互相比值或跨seed均值直接作因果解释，也不能据低回放动作误差认定旧任务安全。

因此，本轮支持的结论是：在固定的三个开发seed上，新S最终整体均值略高于历史原S，收益与退化分布不均，长期Task6遗忘仍明显，整体旧任务最终均值没有改善。每cell仅20回合，1回合对应5个百分点；不进行事后补采、负结果重试、阈值扫描或新增seed。旧机制实验按用户要求永久停止，保留其已有结果，不以本轮结果反向修改它。

## 历史原S的描述性差值

历史验收文件及SHA见INTERPRETATION_AUDIT.json；没有重新运行历史实验。下表单位为百分点，NBT负差表示按项目指标改善。

|seed|ΔAUC|ΔFWT|ΔNBT|ΔOldFinal|ΔFinalAll|
|---|---:|---:|---:|---:|---:|
'''
keys=['AUC','FWT','NBT','OldFinal','FinalAll']
for seed,z in i['historical_original_S_comparison'].items():
    text+='|'+seed+'|'+'|'.join('%+.4f'%z['delta_metrics_pp'][k] for k in keys)+'|\n'
text+='|三seed均值|'+'|'.join('%+.4f'%i['historical_original_S_mean_delta_pp'][k] for k in keys)+'|\n'
text+='\n全部低于历史原S的cell均保留：\n\n|seed|阶段/任务|差值pp|\n|---|---|---:|\n'
for seed,z in i['historical_original_S_comparison'].items():
    for cell,value in z['delta_cells_pp'].items():
        if value<0:text+='|%s|%s|%.1f|\n'%(seed,cell,value)
text+='\n## Stage8到Stage9配对变化\n\nSF为成功转失败，FS反之；初态字节、环境seed和动作噪声逐条绑定核验。\n\n|seed/任务|Stage8成功数|Stage9成功数|SF|FS|\n|---|---:|---:|---:|---:|\n'
for x in i['stage8_to_stage9_pairs']:text+='|%s/%s|%d|%d|%d|%d|\n'%(x['seed'],x['task'],x['stage8_successes'],x['stage9_successes'],x['SF'],x['FS'])
text+='\n## 实际成本与完成证据\n\n训练及评估于2026-10-29 19:57:28 CST自然结束；远端CPU核验于20:26返回，本地最终复算通过。真实成功来源为engineering_copy_20261029_v1/controller.exit=0，原根controller.exit保持不存在，全部自身进程为空。63科学冻结项和5工程源不变，最后进度与最终证据的全部epoch和门控前缀相同。\n\n|seed|纯epoch小时|正式vector分钟|CPU调用/窗口/秒|\n|---|---:|---:|---|\n'
for seed,r in a['seeds'].items():
    c=r['final_CPU'];text+='|%s|%.4f|%.4f|%d/%d/%.4f|\n'%(seed,r['epoch_seconds']/3600,i['evaluation_seconds'][seed]/60,c['calls'],c['windows'],c['seconds'])
text+='''
纯epoch时间不含保存、准入、准备及采集；seed43为5.8483小时，比seed42的7.2525小时少19.36%，但跨seed的缓存、mask与资源不同，不把全部差异归因于复制优化。复制优化已通过四阶段各一整轮全部batch、两分支恢复及保存加载下一步的逐位等价验证，并在seed42自然结束后才切换，没有重训已完成生产epoch。

成本分列：生产585900次optimizer，273次自身隔离准入，工程完整路径验证额外3914次optimizer/18次autograd/0环境；准入123次额外autograd、训练方向108次autograd。工程诊断不混入生产预算。120次训练采集与600次正式回合均无补采或重试。最终CPU合计3175次策略调用、16182窗口，810.306772秒，0优化器更新、0环境；外围校验、启动和传输另计。旧机制中断未记录部分成本UNKNOWN保留在旧账，不混入本轮或记零。

真实最终核验覆盖12阶段模型、3完整恢复文件、12缓存教师重放及抽样、RA/mask与输入hash、600初态真实字节与配对noise、27300门控块和全部标量预算。恢复文件结构/历史/模型及优化器等字段检查通过，既有实际GPU恢复与准入证据另行保留；不宣称额外执行了整条生产轨迹重放或原始相机/物理重演。工程验收通过不代表方法优越性得到证明。

本轮授权实验及解释工作已完成。完成轻量证据发布和远端提交核验后关闭同一监控，不自动追加训练、采集、评估或科学对照。
'''
(N/'FINAL_REPORT.md').write_text(text,encoding='utf8')
(N/'REPORT.md').write_text(text,encoding='utf8')
status=js(N/'RUN_STATUS.json');status.update(status='COMPLETE_ACCEPTED',observation=p['observation'],latest_evidence=D.name,final_CPU_acceptance='PASS_COMPLETE_LOCAL_ACCEPTANCE',local_acceptance_time=now,final_audit_sha256=hashlib.sha256((D/'LOCAL_FINAL_AUDIT.json').read_bytes()).hexdigest(),scientific_conclusion='MIXED_OUTCOMES_NO_CAUSAL_SUPERIORITY_CLAIM',summary=s['statistics'],next_check_minutes=None,processes=[],failures={},publication='VERIFY_CURRENT_PUBLICATION_RECEIPT',report='FINAL_REPORT.md')
status.pop('timing_audit',None);put(N/'RUN_STATUS.json',status)
active=js(N/'ACTIVE_ROOT.json');active.update(status='COMPLETE_ACCEPTED',latest_evidence=D.name,observed=v['observed'],controller_state='NATURAL_EXIT_0_NO_PROCESSES');put(N/'ACTIVE_ROOT.json',active)
handoff='''# 最新完整S OBJECT三seed：独立验收与报告完成

三个seed均400轮/40采集/200正式，合计1200轮、120采集、600正式；2026-10-29 19:57:28 CST自然结束。final_readonly_20261029_201235真实CPU核验与engineering_copy_20261029_v1/audit_final_local.py本地最终验收实际PASS_COMPLETE_LOCAL_ACCEPTANCE；12模型、3恢复、12缓存teacher、初态noise、27300门控及预算验收通过。CPU0环境/0optimizer，没有重复训练评估。

FINAL_REPORT.md与REPORT.md已完成解释，RESULT_SUMMARY和INTERPRETATION_AUDIT保留矩阵、SF/FS、历史原S描述性差值、训练探针与梯度限制。FinalAll均值85%、AUC86.4931%、OldFinal81.1111%；历史原S分别82.5%、85.0347%、82.2222%。旧任务均值未改善，seed43 Task6最终30%且Stage8到9为9 SF/2 FS，不能宣布全面有效或因果优势。

科学合同8594c0b609b8bf948e7914fbf071567ed991d1698a9d8357b90fcb03dec3573d及工程合同befceb11089903813ba65c839e604a44ab1d0c18d7f2dc8282436c581c5ed3cf保持。完成来源仅engineering_copy_20261029_v1/controller.exit=0且无进程；原root controller.exit不存在。不再运行build、guard、validation、dispatch、controller、训练或评估。旧机制永久停止，旧UNKNOWN成本保留。

待收尾仅本轮publish.py轻量发布核remoteSHA/localreceipt，然后删除同一监控s-object-seed。如果已有本地最新PUSH_VERIFIED收据及监控关闭收据，则全部完成，不再远程检查或追加科学工作。禁止发布CP/模型/cache/state/RNG/凭据。已完成的最终CPU不重复运行。
'''
(N/'MONITOR_HANDOFF.md').write_text(handoff,encoding='utf8')
ip=N/'IMPLEMENTATION_NOTES.md';oldtext=ip.read_text(encoding='utf8');head,body=oldtext.split('\n',1)
ip.write_text(head+'\n\n## 最终独立验收及分析完成\n\n'+now+'：最终CPU与本地验收实际通过，见final_readonly_20261029_201235/LOCAL_FINAL_AUDIT.json及FINAL_REPORT.md。没有重训、重评或新增科学实验；最终接受替代下文所有PENDING/STATIC_ONLY历史状态，原证据保留。发布远端核验后关闭监控。\n'+body,encoding='utf8')
print(json.dumps(dict(status=status['status'],final_audit_sha256=status['final_audit_sha256'],summary=s['statistics']),ensure_ascii=False))
