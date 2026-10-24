"""Local evidence binding for completed training and ongoing formal evaluation."""
import json,hashlib,datetime
from pathlib import Path
R=Path(__file__).resolve().parent;D=R/'check_readonly_20261025_041448';C=R/'capacity_readonly_20261025_041513'
js=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=js(D/'LOCAL_AUDIT.json');ev=js(D/'EVALUATION_PROGRESS_AUDIT.json');e=js(D/'EVIDENCE.json');cap=js(C/'EVIDENCE.json')
assert a['failure'] is None and a['phase']['phase']=='evaluate' and a['isolated_optimizer_calls']==70
minutes={};totals={k:0 for k in ['valid','invalid','old_eligible']}
for arm in 'BCNV':
 assert (D/f'light/train_{arm}.exit').read_text().strip()=='0'
 t=js(D/f'light/TRAIN_{arm}.json');assert (t['epochs'],t['steps'],t['gradient_autograd_calls'])==(400,76100,48)
 assert js(D/f'light/TASK6_PARITY_{arm}.json')['equal'];minutes[arm]=t['summed_epoch_seconds']/60
 gradients=js(D/f'light/GRADIENTS_{arm}.json');assert {(g['stage'],g['epoch']) for g in gradients}=={(s,p) for s in [7,8,9] for p in [0,9,49,99]}
 assert len(gradients)==12 and sum(g['extra_autograd_calls'] for g in gradients)==48
 for row in js(D/f'light/EPOCHS_{arm}.json'):
  for k in totals:totals[k]+=row['totals'][k]
assert totals==dict(valid=2857600,invalid=181600,old_eligible=1082000)
for v in cap['volumes'].values():assert v['free']-v['remaining_bound']-v['reserve']==v['headroom']>=0
(C/'LOCAL_AUDIT.json').write_text(json.dumps(dict(status='PASS_NOT_FINAL',evidence_sha256=sha(C/'EVIDENCE.json'),volumes=cap['volumes'],scope='stat-only stable unique inode reserve arithmetic'),indent=2),encoding='utf8')
rollup=dict(status='PASS_TRAINING_METADATA_PENDING_FINAL_CPU_AND_EPISODES',epochs=1600,updates=304400,isolated_optimizer_updates=70,formal_extra_gradient_calls=192,epoch_minutes=minutes,epoch_minutes_total=sum(minutes.values()),population_totals=totals,teacher_bindings=len(a['teacher_bindings']),source_audit_sha256=sha(D/'LOCAL_AUDIT.json'))
(D/'TRAINING_COMPLETION_METADATA_AUDIT.json').write_text(json.dumps(rollup,indent=2),encoding='utf8')
ts=datetime.datetime.fromtimestamp(a['observation'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
s=js(R/'RUN_STATUS.json');s.update(status='PARTIAL_EVALUATING',observation=ts,observation_unix=a['observation'],counts=a['counts'],own_admissions=a['admissions'],isolated_optimizer_calls=70,teacher_bindings=a['teacher_bindings'],formal_episodes=a['formal_episodes'],phase=a['phase'],latest_evidence=D.name,capacity=e['capacity'],capacity_evidence=C.name,next_check_minutes=90,eta=dict(target='all800 formal episodes completion',observed_completed_vectors=8,remaining_vectors=72,mean_linear_minutes=ev['remaining_mean_minutes'],max_observed_linear_minutes=ev['remaining_max_observed_minutes'],planning_minutes=90,limitations='Only8 early Task6/7 batches, three early terminations; future task/model runtimes unknown. Not a guaranteed upper bound; final acceptance extra.'))
(R/'RUN_STATUS.json').write_text(json.dumps(s,indent=2,ensure_ascii=False),encoding='utf8')
note=f'''# 最新2026-10-25 04:17 CST交接

实际观察{ts}，{D.name}/LOCAL_AUDIT、EVALUATION_PROGRESS_AUDIT及TRAINING_COMPLETION_METADATA_AUDIT实际PASS_PROGRESS_NOT_FINAL。四组B/C/N/V各400完整epoch76100主步全部结束，合计1600epoch304400主更新，train_B/C/N/V.exit均0，16阶段CP记录SHA绑定。各Task6实际tensor同原ALIGN；全部70自身隔离GPU更新准入通过，正式额外192梯度autograd核12组stage/epoch取点。BC合法2857600/屏蔽181600、旧合法累计1082000逐epoch复算通过；12本组前阶段教师身份/tau/w/hash绑定。B/C/N/V epoch计算分钟{minutes}，合计{sum(minutes.values()):.6f}，额外准入/缓存/保存/评估另计，不称等算力。真正全CP独立CPU验收仍待全部队列结束。
65冻结项及父源不变，controller20530/evaluate21481健康，phase evaluate；外置进度显示优先controller的evaluate以免旧TRAIN_STATUS误显示仍训练，此仅本地展示修复，不改服务器。正式80/800，8完成vector、4关闭cell、当前B_s8_t7_g0；逐记录/attempt/终止、输出分片与全集、事件顺序/策略CP/配对初态记录及独立重生成noiseSHA通过，无快照前进差异，不算部分性能。独立初态文件字节及全部CP CPU严格加载留待最终。
评估自己缓存准入长度1–8实际PASS，16视觉前向及16策略参数调用、0环境/更新另列，最大feature差9.298325e-6、参数差1.192093e-6。正式8批均{ev['mean_seconds']:.6f}秒，含3个约20秒提前终止批，原时长全部保留；快照当前批已{ev['current_elapsed']:.6f}秒。按均值余{ev['remaining_mean_minutes']:.6f}min；按已观测最慢72.400123秒余{ev['remaining_max_observed_minutes']:.6f}min，两者仅早期Task6/7样本外推、不是保证上界。加阶段加载，暂参考05:20–05:50评估结束，最终验收另计。按用户整段完成点规则，同一监控下一90分钟检查全部评估结束附近，不短周期守候。
{C.name}/LOCAL_AUDIT PASS_NOT_FINAL：system普通free1456017408B扣余新增369422663B及1GiB后余12852921B；data rootfree2527154176B扣余新增361773957B及2GiB后余17896571B。稳定唯一inode信用，仍保守预留原完整保存峰值；只stat、0payload/写入/前向/更新/环境，未删除/付费。此前临时容量review原证据保留。
唯一engineering02 root /data/mlr_ifa/experiment_workspaces/CCBP_V2_FULL_GOAL41_20261024_v1_engineering02、合同56ec3ca98bde2767e3b0a27170b2e8a71a633fbee2ba028b8a0630e7ff47ff71、controller20530保持。早期B6完成后设备比较故障已版本化修复、实际GPU完整状态restore和100epoch精确继承证据保留，完成epoch重复0，未复制额外旧恢复payload。绝不重跑deploy/build/controller/训练或部分正式评估。
下一次check_readonly→audit_progress→audit_evaluation_progress。如果controller.exit0且无自身进程，则立即remote.py final_readonly.py→audit_final_local.py FINAL。新最终路径仍仅静态，外置错误留证修，不重训重评；必须真实16CPU strict/完整恢复/teacher源/800初态字节noise/全部标量/全矩阵和SF-FS及成本验收。完全接受分析发布后删除同一监控，不自动扩seed。整体PARTIAL。

'''
(R/'REPORT.md').write_text(note+(R/'REPORT.md').read_text(encoding='utf8'),encoding='utf8')
old=(R/'MONITOR_HANDOFF.md').read_text(encoding='utf8');contract=old[old.index('继续用户授权CCBP-v2完整GOAL'):]
start=contract.index('最新2026-10-25 00:09:51');end=contract.index('remote.py check_readonly.py',start)
contract=contract[:start]+contract[end:]
(R/'MONITOR_HANDOFF.md').write_text(note+'以下科学合同与限制继续有效。\n\n'+contract,encoding='utf8')
r=js(R/'MONITOR_RECEIPT.json');r.update(interval_minutes=90,basis='All formal evaluation: early8 batches mean remaining63.07min; slowest observed remaining86.68min plus phase loading, plan90min; not guaranteed',latest_evidence=D.name)
(R/'MONITOR_RECEIPT.json').write_text(json.dumps(r,indent=2,ensure_ascii=False),encoding='utf8')
print(json.dumps(dict(training=rollup,episodes=a['formal_episodes'],next_minutes=90)))
