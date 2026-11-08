"""Write the complete report only from actually accepted, paired results."""
from pathlib import Path
import datetime,hashlib,json,statistics,sys
R=Path(__file__).resolve().parent;D=Path(sys.argv[1])
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=read(D/'LOCAL_FINAL_AUDIT.json');c=read(D/'PRIMARY_S_COMPARISON.json');z=read(D/'INTERPRETATION_AUDIT.json')
assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE' and z['status']=='PASS_ACCEPTED_RESULT_INTERPRETATION_RECOMPUTATION'
assert z['sources']['final_audit_sha256']==sha(D/'LOCAL_FINAL_AUDIT.json')
assert z['sources']['comparison_sha256']==sha(D/'PRIMARY_S_COMPARISON.json')
assert z['means']['Full S']['FinalAll']<z['means']['Fixed q=1']['FinalAll']
keys=['AUC','FWT','NBT','OldFinal','NewFinal','FinalAll']
lines=['# OBJECT支持度校准消融：三个seed最终验收与已有完整S比较','',
 '状态：PASS_COMPLETE_LOCAL_ACCEPTANCE。新增固定q=1三个run已全部独立验收；已有完整S只读复用，没有重训或重评。',
 '', '## 结论','',
 '本次OBJECT三seed结果没有显示支持度校准带来平均性能收益。去掉校准、将非空缓存权重固定为1后，平均AUC、FWT、最终旧任务成功率和最终总成功率都更高；完整S仅在平均NBT上更低。收益方向随seed变化，因此既不能把校准写成已验证的稳定增益，也不能据此宣称固定q=1普遍更好。',
 '主结果仍保留原完整S的预定配置；本次是消融结果，不以这些开发集成绩选择新配置，不追加seed或重跑负结果。',
 '', '## 完整指标','',
 '单位均为百分比；NBT越低越好，其余越高越好。所有差值为完整S−固定q=1，单位为百分点。NewFinal为最终Task9成功率。',
 '', '|seed|方法|AUC|FWT|NBT↓|OldFinal|NewFinal|FinalAll|','|---|---|---:|---:|---:|---:|---:|---:|']
for seed in (41,42,43):
 for label in ['Full S','Fixed q=1']:
  m=z['cases'][f'Q1_{seed}']['methods'][label]['metrics']
  lines.append('|%d|%s|'%(seed,label)+'|'.join(f'{m[k]:.4f}' for k in keys)+'|')
for label in ['Full S','Fixed q=1']:
 lines.append('|均值|'+label+'|'+'|'.join(f"{z['means'][label][k]:.4f}" for k in keys)+'|')
lines+=['|均值差|S−q1|'+'|'.join(f"{z['S_minus_fixed_q1_pp'][k]:+.4f}" for k in keys)+'|','',
 '最终总成功率方面，完整S相对q1在seed41为+2.50pp，seed42为−10.00pp，seed43为−1.25pp；平均劣势主要来自seed42。每seed最终80个配对回合，三seed最终240个回合。AUC按既定任务均衡公式复算，不与600回合直接混合平均混淆。',
 '', '## 新旧任务与NBT解释','',
 '完整S的平均NBT更低，但其平均FWT也更低，平均最终旧任务成绩亦低于q1。NBT以刚学完的对角成绩为起点；较低起点会影响后续差值，不能把NBT优势直接解释为最终旧任务更好。负NBT表示按该定义出现平均后向提升，不是负成功率。',
 '', '|最终任务|固定q=1|完整S|S−q1|','|---|---:|---:|---:|']
for t,m in z['final_task_means'].items():lines.append(f"|Task{t}|{m['Fixed q=1']:.4f}|{m['Full S']:.4f}|{m['Full S']-m['Fixed q=1']:+.4f}|")
lines+=['', '## 配对成败与所有阶段任务单元','',
 'SF=完整S成功而q1失败；FS=完整S失败而q1成功。初态文件、初态字节、Normal噪声、env_seed和attempt逐回合匹配；600回合跨任务/阶段重复使用评估设置，不能作为600个独立训练重复。',
 '', '|范围|回合|SF|FS|共同成功|共同失败|','|---|---:|---:|---:|---:|---:|']
for name,k in [('全部阶段','paired_all'),('最终阶段','paired_final')]:
 m=z[k];lines.append(f"|{name}|{m['episodes']}|{m['SF']}|{m['FS']}|{m['both_success']}|{m['both_fail']}|")
cc=z['cell_counts'];lines+=['',f"30个阶段×任务×seed单元中，完整S胜{cc['S_cell_wins']}、平{cc['cell_ties']}、负{cc['S_cell_losses']}。以下保留全部30单元，未筛选负结果。",'',
 '|seed|阶段/任务|固定q=1|完整S|S−q1|SF|FS|','|---|---|---:|---:|---:|---:|---:|']
for seed in (41,42,43):
 for st,m in z['cases'][f'Q1_{seed}']['cells'].items():
  lines.append(f"|{seed}|{st.replace('_','/')}|{m['Variant_percent']:.1f}|{m['S_percent']:.1f}|{m['S_minus_Variant_pp']:+.1f}|{len(m['Variant_fail_S_success'])}|{len(m['Variant_success_S_fail'])}|")
lines+=['', '## 支持度确实被改变，缓存各自产生','',
 'q1所有非空旧任务缓存有效权重实际为1，经验成功率仍记录；原S按成功数/10加权。候选损失、门控旧评分与梯度诊断共用该权重。每个保持阶段实际GPU准入验证q1与均匀保持更新逐位相同，且伪造低经验支持度不改变q1结果。',
 '', '|seed|方法|Task6成功/10|Task7成功/10|Task8成功/10|Task9成功/10|','|---|---|---:|---:|---:|---:|']
for seed in (41,42,43):
 for label in ['Full S','Fixed q=1']:
  m=z['cases'][f'Q1_{seed}']['methods'][label]
  lines.append(f'|{seed}|{label}|'+'|'.join(str(x) for x in m['collection_successes'])+'|')
lines+=['', 'Task9缓存不会再进入后续训练。早期Task6支持度低且在多个后续阶段复用，所以这不是q基本恒等于1的无效消融；但不同方法各自产生缓存，缓存差异也参与后续轨迹演化，不能作为共同中间状态下的单步加权因果实验。',
 '', '## 门控取舍和训练代理','',
 'Q接受率是内部保持候选被保留的频率，不是环境成功率；两种方法的门控旧目标权重也不同，不能把较多接受直接解释为门控更好。梯度指标来自预定epoch0/9/49/99的训练探针，并非独立测试指标。',
 '', '|seed|方法|Task7 Q接受%|Task8 Q接受%|Task9 Q接受%|总Q接受/9100|','|---|---|---:|---:|---:|---:|']
for seed in (41,42,43):
 for label in ['Full S','Fixed q=1']:
  m=z['cases'][f'Q1_{seed}']['methods'][label]
  lines.append(f'|{seed}|{label}|'+'|'.join(f"{m['stage_proxies'][str(t)]['Q_accept_percent']:.2f}" for t in (7,8,9))+f"|{m['total_Q_accepts']}|")
lines+=['', '|seed|方法|阶段|保持梯度范数均值|保持损失均值|保持/BC梯度cos均值|','|---|---|---|---:|---:|---:|']
for seed in (41,42,43):
 for label in ['Full S','Fixed q=1']:
  for t,m in z['cases'][f'Q1_{seed}']['methods'][label]['stage_proxies'].items():
   lines.append(f"|{seed}|{label}|{t}|{m['gradient_proxies']['retain']:.6g}|{m['retention_loss_proxy']:.6g}|{m['retain_bc_cosine_mean']:.6g}|")
lines+=['', '原始逐探针梯度与CPU离线专家成本/路由后悔值保留在LOCAL_FINAL_AUDIT、RECOMPUTED文件和INTERPRETATION_AUDIT中。它们描述各自训练状态，不能单独解释闭环收益或支持校准的必然优势。',
 '', '## 实际验收与成本','',
 '真实controller.exit0且无自身进程之后执行独立CPU验收：12检查点、3完整恢复及optimizer/scheduler加载、12成功缓存teacher重放、600初态与噪声、27300门控标量及预算全部通过。CPU验收0optimizer更新、0环境调用，审计前后全部轻量科学证据SHA完全一致。全部12采集与3正式评估环境锁区间互斥。',
 '新run合计1200轮、316500接受更新、585900生产optimizer调用；273自身隔离optimizer、123准入autograd、108方向诊断autograd单列；120训练采集、600正式。三个seed逐阶段生产与隔离optimizer预算均与已有S相等。训练采集次数相同，缓存内容、FLOPs和墙钟并不保证相同；已有S历史工程成本保留原账，不混入本轮。',
 '', '## 能支持的结论与边界','',
 '可以支持：在本次OBJECT三个开发seed和已固定协议下，没有观察到支持度校准的平均闭环收益；原S的NBT均值优势没有伴随更好的平均最终保持。这个消融应如实作为混合/负结果呈现。',
 '不能支持：q1在所有任务或seed普遍更优，支持度校准无论何时都无效，或者所有差异都由q唯一造成。这里仅三个训练seed，没有宣称显著性或普适追上概率，也不能向GOAL外推。',
 '本次使用RTX3090，已有S来自历史硬件。三个Task6模型tensor均与原S完全相等，但Task6正式成功率q1为45/40/70%，原S为35/45/70%；Task6没有保持项，这些差异不能归于q。Task6训练采集成功数也不同，提示硬件推理/环境执行差异与缓存路径限制。初始化tensor/RNG及评估初态/noise匹配不能消除此边界。',
 '开发评估集曾反复使用，预训练来源审计尚未闭合；本次不是新的独立确认集，也不能声称全局无数据泄露。',
 '', '## 可追溯文件','',
 f'- 最终证据：{D.name}/EVIDENCE.json',f'- 独立验收：{D.name}/LOCAL_FINAL_AUDIT.json',
 f'- 600配对及全部负单元：{D.name}/PRIMARY_S_COMPARISON.json',
 f'- 完整指标CSV：{D.name}/COMPLETE_METRICS.csv',f'- 解释复算：{D.name}/INTERPRETATION_AUDIT.json']
(R/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
accept=dict(status='PASS_COMPLETE_OBJECT_Q1_ACCEPTANCE',observed=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    final_directory=D.name,contract=a['contract'],accepted_seeds=[41,42,43],checkpoints=12,recoveries=3,caches=12,
    formal_episodes=600,training_collection_episodes=120,gate_blocks=27300,optimizer_updates_in_final_audit=0,environment_calls_in_final_audit=0,
    sources={n:sha(D/n) for n in ['EVIDENCE.json','LOCAL_FINAL_AUDIT.json','PRIMARY_S_COMPARISON.json','INTERPRETATION_AUDIT.json']},
    report_sha256=sha(R/'FINAL_REPORT.md'),conclusion='No average closed-loop benefit of support calibration demonstrated in this three-seed OBJECT comparison; NBT alone favors full S. Hardware/development-set boundaries retained.')
(R/'ACCEPTANCE.json').write_text(json.dumps(accept,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print(json.dumps(accept,indent=2,ensure_ascii=False))
