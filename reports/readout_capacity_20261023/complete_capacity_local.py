"""Bind actual final audits and write the complete, negative capacity result. Local only."""
from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parent;F=R/'final_A_CAPACITY_20261023';S=R/'snapshot_20261023_181823';E=R/'final_scalars_readonly_20261023_181901';C=R/'closure_recovery_readonly_20261023_182107'
def js(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=js(F/'LOCAL_FINAL_AUDIT.json');d=js(F/'CAPACITY_DIAGNOSTICS.json');c=js(C/'EVIDENCE.json');e=js(E/'EVIDENCE.json');s=js(S/'EVIDENCE.json')
assert a['status']=='PASS_LOCAL_COMPLETE_A' and a['scientific_decision']=='STOP_STATE_GATE' and a['selected'] is None
assert a['complete_cases']==18 and len(e['checkpoint_sha256'])==90 and a['scalar_error_rows_recomputed']==77814
assert c['status']=='PASS_COMPLETE_RECOVERY_BINDING_AND_CLOSURE' and c['no_own_process'] and c['all_phase_exit0']
assert c['contract_sha256']==e['contract_sha256']=='8ad12b59271e3d6478c4b8e6a2480adba5ccc55b8b38ec5fe1fe6d4e305eb9b0'
assert c['best_cp_sha256']==e['checkpoint_sha256'][str(Path(js(R/'ACTIVE_ROOT.json')['root'])/'output/private/checkpoints/S_task7_fold4_b200.pth').replace('\\','/')]
assert s['controller_exit']=='0' and not s['gpu'].strip() and not s['hash_bad']
assert all(not v['passed'] for model in a['checks'].values() for v in model['checks'])
closure=js(F/'CLOSURE_INDEX.json');closure.update(recovery_audit_sha256=sha(C/'EVIDENCE.json'),recovery_audit_source_sha256=sha(R/'closure_recovery_readonly.py'),recovery_sha256=c['recovery_sha256'],all_full_epoch_fields_bound=True,wrapper_repairs_required=False)
(F/'CLOSURE_INDEX.json').write_text(json.dumps(closure,indent=2),encoding='utf8')
now=datetime.datetime.now().astimezone().isoformat();exit_time=datetime.datetime.fromtimestamp(s['status']['unix']).astimezone().isoformat()
metrics={(v['temporal_ffn'],v['task'],v['kind'],v['target']):v for v in d['pooled_test'] if v['budget']==200}
lines=['\n\n## 最终结果：完整验收，严格A失败，结束诊断','',f'本地闭合时间{now}。controller122601于{exit_time}自然exit0，全部阶段exit0，无GPU/自身进程。189项冻结源和父合同不变。30条轨迹全部6000epoch、49800正式更新；90个50/100/200验证best检查点实际CPU strict通过，完整epoch、earliest验证best、分区身份、父Patch最终SHA通过。本地18case、77814误差行的分位数/CI/动态视觉条件独立复算一致。selected=null，8个任务目标准入检查全部未过，B保持0策略更新/0环境。','',
'final_A_CAPACITY_20261023/LOCAL_FINAL_AUDIT、CAPACITY_DIAGNOSTICS与CLOSURE_INDEX均实际PASS。最终full-epoch恢复状态另以closure_recovery_readonly_20261023_182107核实：最后S_task7_fold4 epoch199→200、1600更新、43个优化器参数状态、model/optimizer/best/selected/cursor/allRNG/grads/modes/requires_grad/父PatchSHA绑定；验证best epoch191与已验收所选CP一致。该检查0前向/优化器/CUDA/环境/服务器写入。最终包装路径首次真实使用全部通过，无包装修复或科学重跑。','',
'### 200epoch范围内验证选优：heldout pooled median / P90，mm','',
'G为同查询读出器的全局视觉token输入，S为冻结空间Patch输入；除两层Temporal FFN宽度外条件保持。以下为完整留出演示合并误差，不是每折统计的平均。','',
'| 模型 | 任务 | 目标 | 原FFN512 | FFN128 |','|---|---|---|---:|---:|']
for task in (6,7):
 for kind in ('G','S'):
  for target,label in [('object_eef','物体—末端'),('object_goal','物体—目标')]:
   old=metrics[512,task,kind,target];new=metrics[128,task,kind,target]
   lines.append(f"| {kind} | {task} | {label} | {old['median_mm']:.2f} / {old['p90_mm']:.2f} | {new['median_mm']:.2f} / {new['p90_mm']:.2f} |")
lines += ['', 'G Task6物体—末端median9.51mm，但P90=25.500747mm仍超过25mm，不能舍入过关。四个物体—目标median仍20.27–24.64mm，P90为48.92–64.04mm。G的Task7两种位置指标均变差，S的Task7物体—末端有所改善但仍不达标；不能仅挑改善单元报告。全部目标、夹爪、50/100/200、各分区与所有配对差保存在CAPACITY_DIAGNOSTICS.json。', '',
'### 泛化、容量与动态读出的结论', '',
'缩小FFN没有一致降低未见演示的定位误差。8个G/S×两任务×两位置目标的宽度配对demo探索CI中7个跨0；G Task7物体—末端的“旧512误差减新128误差”均差−2.2712mm，CI [−4.1900,−0.8482]，方向为变差。该比较使用重复检查的演示，且存在多重探索，不能写成独立确认性显著发现。', '',
'原10/25mm精度之外，G Task7物体—末端以及S两任务物体—末端的动态前一目标优势不为正；多个对小P0的视觉收益CI跨0。最新视觉替换优势虽为正，并不等于当前状态已被精确恢复，更不能代替完整门或闭环控制价值。', '',
'物体—目标训练/验证/测试的每折median均值如下（与上表pooled口径不同）：', '',
'| 宽度 | 模型 | 任务 | 训练 | 验证 | 测试 |','|---|---|---|---:|---:|---:|']
for v in d['partitions']:
 if v['budget']==200 and v['kind'] in ('G','S'):
  z=v['mean_per_fold_not_pooled'];lines.append(f"| {v['temporal_ffn']} | {v['kind']} | {v['task']} | {z['train']['object_goal']['median_mm']:.2f} | {z['validation']['object_goal']['median_mm']:.2f} | {z['test']['object_goal']['median_mm']:.2f} |")
lines += ['', '小模型仍能拟合训练演示（G目标4.57/4.23mm，S6.45/5.34mm），但留出演示约20–24mm。证据支持“仅增加预算或降低这两层FFN容量不足以解决跨演示定位”；不能据此唯一归因CLIP缺失物理信息、Temporal无效或探针不可训练。对象/目标查询的语义对齐、几何可辨识性、演示覆盖等仍未由本对照独立分开。原512→128输入压缩保持不变，因此本实验也未检验取消该输入瓶颈。', '',
'### 计算成本、停止边界与保留证据', '', '| 宽度 | 模型 | 参数 | 正式更新 | fit秒 |','|---|---|---:|---:|---:|']
for width,rows in d['compute'].items():
 for kind,v in rows.items():lines.append(f"| {width} | {kind} | {v['parameters']} | {v['updates']} | {v['seconds']:.3f} |")
total=sum(v['seconds'] for v in d['compute']['128'].values());oldtotal=sum(v['seconds'] for v in d['compute']['512'].values())
lines += ['',f'本次总fit {total:.6f}秒（{total/60:.2f}分钟），旧512为{oldtotal/60:.2f}分钟。G/S参数均减少197376（约33.00%），但本次fit墙钟只减少约{(oldtotal-total)/oldtotal*100:.2f}%；S单组略慢，不能据此声称等算力或普遍速度收益。准备、准入与审计不含在fit时间。新版本9隔离优化器调用和6基线一致性前向另列；首版报告工程失败前同样9隔离更新与6一致性前向保留为额外成本。首版0正式epoch/更新，未造成科学重训。', '',
'本阶段完整验收，科学A失败。按拟合前登记，这是最后一个自动容量值：发布核验后删除patch监控，不自动扫描64/256、更多epoch或正则；严格A未通过，B不启动。进一步研究需要新的研究决策，原失败与取消任务均保留。私有Patch、模型、原始数据未下载或删除；仅发布代码、报告、SHA与误差标量。最终data root约5.456GB、system普通6.645GB，满足2GiB/1GiB保留。']
with (R/'REPORT.md').open('a',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
state=js(R/'RUN_STATUS.json');state.update(status='COMPLETE_DIAGNOSTIC_A_FAILED',phase='COMPLETE_LOCAL_ACCEPTANCE_STOP_STATE_GATE',observed=now,latest_snapshot=S.name,progress=js(S/'LOCAL_AUDIT.json'),controller_exit='0',controller_exit_time=exit_time,gpu_train=None,final_path='ACTUAL_PASS_90_CP_18_CASES_FULL_RECOVERY_CLOSURE',final_audit='final_A_CAPACITY_20261023/LOCAL_FINAL_AUDIT.json',diagnostics='final_A_CAPACITY_20261023/CAPACITY_DIAGNOSTICS.json',closure='final_A_CAPACITY_20261023/CLOSURE_INDEX.json',local_acceptance_pass=True,scientific_decision='STOP_STATE_GATE',selected=None,next_check_minutes=None,training_eta_reference=None,monitor_action='DELETE_AFTER_VERIFIED_PUBLICATION',automatic_science_continuation=False,policy_updates=0,formal_episodes=0)
(R/'RUN_STATUS.json').write_text(json.dumps(state,indent=2),encoding='utf8')
(R/'MONITOR_HANDOFF.md').write_text('''# 诊断已闭合，禁止再部署或轮询

容量对照30fits6000epoch49800更新、90CP/18case77814误差行及最终完整恢复状态已真实独立验收PASS，科学A为STOP_STATE_GATE，selected=null。所有8个门检查未过，B未启动。控制器自然exit0，无自身进程；原失败保留。最终资料见REPORT.md、RUN_STATUS.json和final_A_CAPACITY_20261023。此为最后一个自动容量取值，不自动开启宽度/正则/预算扫描，不恢复旧取消任务。最终发布核验后删除patch监控；没有后续定时工作。若意外收到旧heartbeat，只核发布receipt/监控删除状态，不重复服务器检查/训练/下载/最终验收。
''',encoding='utf8')
print(json.dumps(dict(status=state['status'],local_acceptance=True,scientific_decision=state['scientific_decision'],fit_seconds=total,closure_sha256=sha(F/'CLOSURE_INDEX.json')),indent=2))
