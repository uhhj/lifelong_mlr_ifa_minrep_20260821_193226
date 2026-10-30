"""Record completed acceptance and reviewed interpretation; no remote execution."""
from pathlib import Path
import datetime
import hashlib
import json

R = Path(__file__).resolve().parent
D = R / 'final_readonly_20261030_111546'

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def write(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf8')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

a = read(D/'LOCAL_FINAL_AUDIT.json')
c = read(D/'PRIMARY_C_COMPARISON.json')
e = read(D/'EVIDENCE.json')
assert a['status'] == 'PASS_COMPLETE_LOCAL_ACCEPTANCE'
assert c['status'] == 'PASS_ACCEPTED_C_S_600_EPISODE_PAIRING'
assert a['formal_episodes'] == c['paired_episodes'] == 600
assert a['epochs'] == 1200 and a['main_updates'] == 228300
assert a['physical_training_optimizer_calls'] == 426600
assert not e['processes'] and not e['canceled_mechanism_processes']
assert a['evidence_sha256'] == sha(D/'EVIDENCE.json')
assert len(c['negative_cells']) == 7
assert c['mean_delta_pp']['FinalAll'] == 5
for seed, cell, value in [('41','9_6',25),('42','9_8',20),('43','9_9',40),('43','9_6',-15)]:
    assert c['seeds'][seed]['cells'][cell]['delta_pp'] == value

interpretation = '''
## 科学解释与局限（最终证据人工复核）

本轮完整方法在三seed均值上优于已有完整C：最终全部任务成功率80%→85%，AUC82.50%→87.26%，新任务获得指标FWT85%→89.58%，旧任务最终80.56%→83.89%。但NBT仅从5.56降至5.28个百分点，改善0.28个百分点，不能据此声称明显解决遗忘。这里的接受指执行与证据通过验收，不表示每个seed或任务都改善。

seed41的AUC下降1.875个百分点、FWT下降1.25个百分点，NBT增加3.333个百分点；最终Task6相对C提高25个百分点，却伴随Task8下降15、Task9下降10，最终总体与C持平。seed42没有负单元，最终增益来自Task8提高20个百分点，最终总体提高5个百分点。seed43最终Task9提高40、Task8提高20，但Task6下降15、Task7下降5；最终总体提高10个百分点而旧任务最终均值持平。因此平均收益包含新任务学习增强，不能全部解释为旧任务保持改善。

上述比较保留全部7个负单元（见主C表），没有按测试结果筛选seed、重试或补采。S内部从任务首次获得到最终的旧任务配对共有20次SF（成功变失败）和9次FS（失败变成功）；例如seed41 Task6为18/20→13/20，SF5、FS0，seed43 Task8为17/20→16/20，SF4、FS3。净成功数可能遮蔽样本身份变化，这与C/S同阶段的配对差异是两种不同比较。

训练代理也不能代替闭环：seed41 Task9第100轮首batch的执行误差约0.007153、支持加权保持约0.000868，但正式Task8/9仍比C差；此处保持梯度与BC、执行误差梯度的余弦分别约−0.1189和−0.0764，显示局部方向冲突。seed42 Task7第50→100轮首batch，BC从−26.6518降至−31.6365，执行误差却从0.03316升至1.02127，两个代理并不等价。seed43 Task8保持代理在第1/10/50/100轮分别约0.1042/0.00339/0.04217/0.14048，并非随epoch单调改善。门控约束是同一块起点的C/Q局部比较，不是跨epoch或正式轨迹的全局保证。Task7起点保持项接近数值零，其梯度余弦不作为可靠方向证据。所有36条方向记录及108次额外autograd已单独计账。

三个seed共9678/20100块选择Q（48.15%）；这一比例和训练数据上的固定探针改善，只说明候选选择行为。q来自各阶段自身成功缓存的支持度，不能解释为独立测试成功概率。seed41各旧任务q全1，seed42的Task6为0.9，seed43后期Task8为0.8；其余为1。支持度、门控接受率与最终收益之间不能据三seed建立因果关系，原取消的机制实验也没有被恢复。

C/S全部600回合初态字节、噪声和环境seed配对通过，但训练交互与算力不等：S额外120训练采集，生产优化器调用426600对C的228300，约1.869倍，准入/方向诊断另记。结果支持“当前完整方法在这三个GOAL开发seed上的平均闭环表现改善”，不支持“单独门控造成增益”“等成本优越”或统计显著性结论。每单元20回合、仅三seed，保留seed和任务差异，不扩实验或据负结果调参。

最终实际CPU验收已完成12个阶段CP、3个完整恢复状态、12个缓存teacher重放、600个初态/noise配对及20100块标量复算；66冻结项保持，控制器和各train/evaluate自然exit0且无自身进程。验收没有新增训练或环境回合。原STATIC_ONLY包装审查保留为历史，不代替本次实际验收。
'''
report = R/'FINAL_REPORT.md'
text = report.read_text(encoding='utf8')
marker = '报告发布前仍需人工核对解释文字：逐项说明新旧任务变化与负结果，不能仅凭自动表格宣布机制有效。'
assert marker in text, 'Run once; do not overwrite reviewed interpretation'
text = text.replace(marker, (D/'PRIMARY_C_COMPARISON.md').read_text(encoding='utf8')+'\n'+interpretation)
report.write_text(text, encoding='utf8')
now = datetime.datetime.now().astimezone().isoformat()
review = dict(status='PASS_FINAL_INTERPRETATION_REVIEW', time=now,
    actual_final_observation=e['observed'], audit_sha256=sha(D/'LOCAL_FINAL_AUDIT.json'),
    primary_c_comparison_sha256=sha(D/'PRIMARY_C_COMPARISON.json'), report_sha256=sha(report),
    all_negative_cells_retained=7, causal_claim=False, significance_claim=False,
    limits=['three development seeds', '20 episodes per cell', 'unequal interaction and compute', 'training proxies are not held-out validation'])
write(D/'INTERPRETATION_REVIEW.json',review)
s = read(R/'RUN_STATUS.json')
s.update(status='COMPLETE_ACCEPTED', observation=e['observed'],
    latest_evidence=D.name, final_CPU_acceptance=a['status'],
    scientific_conclusion='Positive mean closed-loop comparison with heterogeneous seeds and minimal mean NBT improvement; unequal costs; no causal or significance claim',
    primary_c_pairing=c['status'], next_check_minutes=None, controller_exit=0,
    formal_extra_autograd=a['formal_extra_autograd'], interpretation_review=D.name+'/INTERPRETATION_REVIEW.json',
    snapshot_sha256=sha(D/'EVIDENCE.json'), publication='See separately verified PUBLICATION_RECEIPT; acceptance is independent of publication')
for row in s['counts'].values():
    row['last_training_state'] = row.pop('current')
    row['current'] = dict(phase='complete',train_exit=0,evaluate_exit=0)
write(R/'RUN_STATUS.json',s)
active = read(R/'ACTIVE_ROOT.json')
active.update(status='COMPLETE_ACCEPTED',observed=e['observed'],latest_evidence=D.name,controller_exit=0,processes=[])
write(R/'ACTIVE_ROOT.json',active)
summary = '''# 最新完整S GOAL三seed已完成

2026-10-30实际独立验收通过，seed41/42/43各400轮、40训练采集、200正式回合全部完成。主对照为已有完整C，600回合初态/noise逐项配对通过，无新训C、重训、重评或恢复旧机制。

三seed平均最终全部任务成功率C80%→S85%（+5个百分点），AUC82.50%→87.26%（+4.76），旧任务最终80.56%→83.89%（+3.33）。NBT仅改善0.28个百分点，seed41 AUC下降、最终总体持平，所有7个负单元保留。不能声称全面减少遗忘或单机制因果收益；训练交互及算力不等。

完整解释见[FINAL_REPORT.md](FINAL_REPORT.md)，实际验收见[LOCAL_FINAL_AUDIT.json](final_readonly_20261030_111546/LOCAL_FINAL_AUDIT.json)，主C配对见[PRIMARY_C_COMPARISON.md](final_readonly_20261030_111546/PRIMARY_C_COMPARISON.md)。发布以本地PUBLICATION_RECEIPT的remote SHA核验为准。
'''
(R/'REPORT.md').write_text(summary,encoding='utf8')
h = R/'MONITOR_HANDOFF.md'
h.write_text(summary+'\n全部科学运行、实际验收和解释已完成；发布remote SHA核验后删除s-goal-seed。禁止再次运行训练、评估、final_readonly或恢复旧机制。下方历史记录仅供追溯。\n\n'+h.read_text(encoding='utf8'),encoding='utf8')
bindings = {str(p.relative_to(R)).replace('\\','/'):sha(p) for p in [D/'EVIDENCE.json',D/'LOCAL_FINAL_AUDIT.json',D/'PRIMARY_C_COMPARISON.json',D/'INTERPRETATION_REVIEW.json',report,R/'REPORT.md',R/'RUN_STATUS.json',R/'ACTIVE_ROOT.json']}
write(R/'FINAL_ACCEPTANCE_RECEIPT.json', dict(status='COMPLETE_ACCEPTED_INTERPRETED',time=now,files=bindings,production_epochs=1200,formal_episodes=600,publication_receipt='Recorded separately after verified non-force publication'))
print(json.dumps(dict(status='COMPLETE_ACCEPTED_INTERPRETED', files=len(bindings)),ensure_ascii=False))
