"""Recompute descriptive tables and bind the completed OBJECT acceptance."""
from pathlib import Path
import json,hashlib,statistics,datetime
R=Path(__file__).resolve().parent;D=R/'final_acceptance_20261103_v1/final_readonly_20261103_124306'
js=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=js(D/'LOCAL_FINAL_AUDIT.json');c=js(D/'PRIMARY_S_COMPARISON.json');e=js(D/'EVIDENCE.json')
assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE' and a['prior_L005_acceptance_preserved']
assert c['paired_episodes']==1200 and a['formal_episodes']==1200
assert a['evidence_sha256']==sha(D/'EVIDENCE.json')
assert e['light']['engineering_cross_lambda_20261102_v1/controller.exit']['text']=='0' and not e['processes']
lines=['# OBJECT固定λ_R敏感性：六run最终接受与三档解释','',
'六个新增run已完整结束并通过实际远程CPU和本地独立复算。每run400轮、40训练采集、200正式；共2400轮、633000接受更新、1171800生产物理optimizer调用、240采集、1200正式。24个CP、6个完整恢复state、24份缓存teacher、1200实际初态/noise和54600门控块均通过。18次纳入并行锁的采集和6次正式全部互斥释放；其余早期采集保留原单进程序列证据。原0.05独立接受结果逐项SHA及指标保持。','',
'三seed的最终成功率均值，λ=0.05/0.1/0.2分别为82.0833%/85.0000%/78.3333%。0.2的下降主要集中在最终旧Task6；不是所有旧任务都更差。三档是预定敏感性结果，保留全部正负结果，不据此重新选择参数或追加实验。','',
'|指标（%或百分点）|0.05|0.1（已有）|0.2|','|---|---:|---:|---:|']
for k in ['AUC','FWT','NBT','OldFinal','FinalAll']:
 lines.append('|%s|%.4f|%.4f|%.4f|'%(k,c['suites']['L005'][k]['Variant_mean'],c['suites']['L005'][k]['S_mean'],c['suites']['L020'][k]['Variant_mean']))
lines+=['','NBT越低越好；负值表示相对各任务首次学完时，后续平均表现提高，不是负的失败概率。AUC按任务在后续阶段的平均再等权平均；FWT是各任务首次学习后对角表现；OldFinal为最终Task6–8，FinalAll为最终Task6–9。','',
'|seed|0.05最终|0.1最终|0.2最终|0.05−0.1|0.2−0.1|','|---|---:|---:|---:|---:|---:|']
for seed in (41,42,43):
 x=c['cases']['L005_'+str(seed)];y=c['cases']['L020_'+str(seed)]
 lines.append('|%d|%.2f|%.2f|%.2f|%+.2f|%+.2f|'%(seed,x['Variant_metrics']['FinalAll'],x['S_metrics']['FinalAll'],y['Variant_metrics']['FinalAll'],-x['delta_pp']['FinalAll'],-y['delta_pp']['FinalAll']))
lines+=['','0.05的三个seed相对0.1为−11.25、+2.50、0个百分点，方向不一致。0.2三个seed均下降（−15.00、−3.75、−1.25个百分点），但这里只描述这三个开发seed，不给出显著性、普适成功概率或单机制因果结论。','',
'|最终任务|0.05均值|0.1均值|0.2均值|','|---|---:|---:|---:|']
for t in range(6,10):
 vals=[statistics.mean(c['cases'][g+'_'+str(s)]['cells']['9_'+str(t)]['Variant_percent'] for s in (41,42,43)) for g in ['L005','L020']]
 baseline=statistics.mean(c['cases']['L005_'+str(s)]['cells']['9_'+str(t)]['S_percent'] for s in (41,42,43))
 lines.append('|Task%d|%.4f|%.4f|%.4f|'%(t,vals[0],baseline,vals[1]))
lines+=['','0.2最终Task6为26.67%，比0.1的55%低28.33个百分点；Task7持平，Task8反而高6.67个百分点，最新Task9低5个百分点。这种分化说明增大保持系数没有在本次观察中均匀改善旧任务保持。0.05主要在Task6、7和最新Task9低于0.1，Task8均值相同。两档FWT均略低，且OldFinal下降幅度在0.2更大；不能只看一个遗忘均值掩盖任务间差异。','',
'SF定义为新增系数成功、0.1失败；FS反之。下面每档600回合使用同一份既有0.1结果，两个比较不构成1200个独立基线回合。','',
'|λ|30单元胜/平/负|全部600 SF/FS|最终240 SF/FS|新任务240 SF/FS|旧任务360 SF/FS|','|---|---|---|---|---|---|']
interpret={}
for g,lam in [('L005',.05),('L020',.2)]:
 cases=[c['cases'][g+'_'+str(s)] for s in (41,42,43)];pairs=[p for x in cases for p in x['paired_outcomes']];cells=[p for x in cases for p in x['cells'].values()]
 counts=[sum(x['Variant_percent']>x['S_percent'] for x in cells),sum(x['Variant_percent']==x['S_percent'] for x in cells),sum(x['Variant_percent']<x['S_percent'] for x in cells)]
 def discord(p):return [sum(x['Variant'] and not x['S'] for x in p),sum(x['S'] and not x['Variant'] for x in p)]
 groups=dict(all=discord(pairs),final=discord([x for x in pairs if x['stage']==9]),new=discord([x for x in pairs if x['stage']==x['task']]),old=discord([x for x in pairs if x['stage']>x['task']]))
 lines.append('|%s|%s|%s|%s|%s|%s|'%(lam,'/'.join(map(str,counts)),*('/'.join(map(str,groups[k])) for k in ['all','final','new','old'])))
 q=sum(sum(v['Q_accepts'] for v in a['cases'][g+'_'+str(s)]['gate_audit']['by_stage'].values()) for s in (41,42,43))
 gradients=[z for s in (41,42,43) for z in a['cases'][g+'_'+str(s)]['gradient_records']]
 interpret[g]=dict(cell_wins_ties_losses=counts,paired_discordance=groups,Q_accepts=q,gate_blocks=27300,Q_accept_percent=q/27300*100,gradient_records=len(gradients),retention_BC_negative_dot=sum(x['retain_vs_bc']['dot']<0 for x in gradients),retention_exec_negative_dot=sum(x['retain_vs_exec']['dot']<0 for x in gradients))
lines+=['','0.05旧任务SF/FS为29/46，0.2为29/52；主要净损失来自旧任务，而不是仅最新任务少数回合。SF和FS同时存在，不能把均值差解释为每条轨迹都单向改善或变差。','',
'门控与梯度只用于解释训练路径。0.05接受Q为12897/27300（47.2418%）；0.2为13812/27300（50.5934%）；既有0.1为12375/27300（45.3297%）。0.2更常接受Q但闭环最终更低，说明固定探针的局部门控通过率不等于环境收益。每档36条方向记录中，保持与BC负点积为13/11条（0.05/0.2），与执行代理负点积为10/14条；这些稀疏诊断不能证明全训练方向或门控的独立因果效应。每个run用自己的成功缓存与q，缓存内容并不固定；成功筛选不保证每个动作最优，缓存特征teacher回放也不是原图像或物理环境重放。','',
'硬件与数值边界必须随结论保留。用户授权在主机重启后从4080SUPER迁移3090；原260完整轮及43的289轮、全部已记录门控前缀保持，恢复核验成本另记。六个新增run的Task6模型tensor与相应0.1均实际相等，但首次Task6闭环每档为45/40/70%，0.1为35/45/70%；因此同模型、相同初态/noise也没有消除硬件推理或环境执行差异。不能把全部差值归于λ，也不能称为硬件完全受控的单因素因果实验。L00541 Task6唯一已定位1ULP scale重算差异按case/cache/input/旧新值/float64参考精确绑定；其余23缓存保持精确检查，未修改生产或泛化放宽容差。','',
'三档每seed接受更新、双候选物理调用及采集次数相同；不保证FLOPs、墙钟、成功缓存内容相同。新增六run自身546隔离optimizer/246准入autograd及216方向autograd分别计账；历史工程35418 optimizer/94 autograd另列。两次中断放弃候选已记录540调用，最后未记账部分保持UNKNOWN，不能声称全部实际计算成本精确。最终CPU验收0 optimizer更新、0环境回合；既有0.1未重训重评。','',
'沿用数据泄露审计边界：初态20–29训练采集与0–19正式分离且逐字节配对核验，不能据此宣称全局无泄露。开发评估集已反复用于方法开发，预训练来源仍未完全闭合；本结果是开发集敏感性分析，不是新的独立确认。不会基于成绩选择最佳λ或追加实验。','',
'以下列出全部60个seed×阶段×任务单元（包括两方向负单元），每单元20回合。差值为新增系数−0.1；具体初态索引及CP/noise哈希见PRIMARY_S_COMPARISON.json。','',
'|λ/seed|阶段/任务|新增|0.1|差值pp|SF|FS|','|---|---|---:|---:|---:|---:|---:|']
for case,x in c['cases'].items():
 for cell,row in x['cells'].items():lines.append('|%s|%s|%.1f|%.1f|%+.1f|%d|%d|'%(case,cell.replace('_','/'),row['Variant_percent'],row['S_percent'],-row['S_minus_Variant_pp'],len(row['Variant_success_S_fail']),len(row['Variant_fail_S_success'])))
lines+=['','验收证据：final_acceptance_20261103_v1/final_readonly_20261103_124306；旧L005_ACCEPTANCE和L005_FINAL_REPORT永久保留。本报告完成OBJECT部分，后续按已授权协议独立接续GOAL六run，沿用同一监控。']
(R/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
(D/'INTERPRETATION_AUDIT.json').write_text(json.dumps(dict(status='PASS_DESCRIPTIVE_RECOMPUTATION',groups=interpret,all_60_cells_reported=True,primary_comparison_sha256=sha(D/'PRIMARY_S_COMPARISON.json'),report_sha256=sha(R/'FINAL_REPORT.md')),indent=2)+'\n',encoding='utf8')
receipt=dict(status=a['status'],suite='OBJECT',accepted_at=datetime.datetime.now().astimezone().isoformat(),final_directory=str(D.resolve()),evidence_sha256=sha(D/'EVIDENCE.json'),local_final_audit_sha256=sha(D/'LOCAL_FINAL_AUDIT.json'),comparison_sha256=sha(D/'PRIMARY_S_COMPARISON.json'),report_sha256=sha(R/'FINAL_REPORT.md'),completion_source='engineering_cross_lambda_20261102_v1/controller.exit',no_own_processes=True,epochs=2400,physical_optimizer_calls=1171800,accepted_updates=633000,formal_episodes=1200,prior_L005_acceptance_preserved=True,followup='GOAL_AUTHORIZED_PENDING_OWN_VALIDATION_AND_DEPLOYMENT',cleanup='PENDING_POST_PUBLICATION',all_experiment_scope_complete=False)
(R/'ACCEPTANCE.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
print(json.dumps(interpret,indent=2))
