"""Local-only round closure, no training or environment calls."""
from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parent;D=R/'check_readonly_20261025_010917';C=R/'capacity_readonly_20261025_011027'
js=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=js(D/'LOCAL_AUDIT.json');e=js(D/'EVIDENCE.json');c=js(C/'EVIDENCE.json')
assert a['failure'] is None and (D/'light/train_B.exit').read_text().strip()=='0'
b=js(D/'light/TRAIN_B.json');assert b['epochs']==400 and b['steps']==76100 and b['gradient_autograd_calls']==48
assert js(D/'light/TASK6_PARITY_C.json')['equal']
for v in c['volumes'].values():assert v['free']-v['remaining_bound']-v['reserve']==v['headroom']>=0
(C/'LOCAL_AUDIT.json').write_text(json.dumps(dict(status='PASS_NOT_FINAL',evidence_sha256=sha(C/'EVIDENCE.json'),volumes=c['volumes'],scope='stat-only capacity bound arithmetic; no payload reads'),indent=2),encoding='utf8')
remaining=76100-a['counts']['C']['steps'];seconds=remaining*a['rates_last10']['C']['7']/139
ts=datetime.datetime.fromtimestamp(a['observation'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
s=js(R/'RUN_STATUS.json');s.update(status='PARTIAL_TRAINING',observation=ts,observation_unix=a['observation'],counts=a['counts'],own_admissions=a['admissions'],isolated_optimizer_calls=a['isolated_optimizer_calls'],teacher_bindings=a['teacher_bindings'],formal_episodes=a['formal_episodes'],phase=a['phase'],latest_evidence=D.name,capacity=e['capacity'],capacity_evidence=C.name,next_check_minutes=55,eta=dict(target='C whole training',remaining_steps=remaining,linear_minutes=seconds/60,own_rate_samples=1,planning_extra='save/cache/preparation; stages8/9 C rates unmeasured',not_total_queue_eta=True))
(R/'RUN_STATUS.json').write_text(json.dumps(s,indent=2,ensure_ascii=False),encoding='utf8')
note=f'''# 最新2026-10-25 01:11 CST交接

实际观察{ts}，{D.name}/LOCAL_AUDIT PASS_PROGRESS_NOT_FINAL。65冻结项及父源码不变，controller20530/GPU20727健康，已自然进入C Task7。B完整400epoch76100主步、四阶段CP记录与train_B.exit0通过；epoch纯计算{b['summed_epoch_seconds']/60:.6f}分钟，48额外梯度autograd另列，保存/准入/教师准备不算在该时间内。B最终CP仍待全部完成后的CPU独立最终验收，不算闭环收益。
C Task6完整100epoch10000步且服务器实际tensor与原ALIGN相等；Task7下载1完整epoch139步，合计101epoch10139步。自己的C6/C7隔离4/5更新准入实际PASS，包括完整下一步恢复、状态化梯度测量开关、旧C损失更新精确。累计B/C自身28隔离更新、0环境；N/V未准入，正式0。四个已生成阶段教师身份、tau/w与自身前阶段CP/hash本地复核通过。没有本轮恢复重启或改科学参数。
C剩{remaining}主步，仅首轮6.100154秒/139batch，线性{seconds/60:.6f}分钟；B后段近期实测速率提供约49分钟量级交叉参考，C后续阶段尚未测。保存/缓存/准备另计，下一55分钟在C整组训练结束附近检查，约02:05–02:10；不是全队列完成预测，不内部阶段短轮询。
{C.name}/LOCAL_AUDIT PASS_NOT_FINAL，稳定唯一inode分配量抵扣剩余新增后，system普通保留1GiB之外余12955321B，data root保留2GiB之外余21812347B；余量紧但仍满足预登记峰值，逐epoch原容量保护继续。只stat、0payload读取/服务器写入/前向/更新/环境，无新增删除或付费。已分配B9硬链接仅计一次，不能按全额重复预留误报不足。
当前唯一engineering02 root/controller20530/合同56ec3ca98bde2767e3b0a27170b2e8a71a633fbee2ba028b8a0630e7ff47ff71不变。先前B6后设备比较失败及完整恢复证据保留；新进度不改变恢复历史。不要重跑deploy/build/controller。全部四组训练后才800正式环境，最终wrapper仍仅静态待真实调用。整体PARTIAL。

'''
(R/'REPORT.md').write_text(note+(R/'REPORT.md').read_text(encoding='utf8'),encoding='utf8')
old=(R/'MONITOR_HANDOFF.md').read_text(encoding='utf8');old=old.split('\n最新观察：')[0]
(R/'MONITOR_HANDOFF.md').write_text(note+'以下保留科学合同和历史工程交接；旧进度与ETA以上述最新段落为准。\n\n'+old,encoding='utf8')
receipt=js(R/'MONITOR_RECEIPT.json');receipt.update(interval_minutes=55,basis='C remaining65961 steps / own first stage7 batch rate ~48.25min plus preparation; B full-run timing is comparable reference',latest_evidence=D.name)
(R/'MONITOR_RECEIPT.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding='utf8')
print(json.dumps(dict(observation=ts,B_epoch_minutes=b['summed_epoch_seconds']/60,C_remaining_linear_minutes=seconds/60,next_minutes=55)))
