"""Local-only progress, capacity review and handoff; no scientific execution."""
from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parent;D=R/'check_readonly_20261025_020414';CAP=R/'capacity_readonly_20261025_020512';FIRST=R/'capacity_readonly_20261025_020440'
js=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=js(D/'LOCAL_AUDIT.json');e=js(D/'EVIDENCE.json');c=js(CAP/'EVIDENCE.json');prior=js(FIRST/'EVIDENCE.json')
assert a['failure'] is None and a['formal_episodes']==0
times={}
for arm in 'BC':
 assert (D/f'light/train_{arm}.exit').read_text().strip()=='0'
 t=js(D/f'light/TRAIN_{arm}.json');assert (t['epochs'],t['steps'],t['gradient_autograd_calls'])==(400,76100,48)
 times[arm]=t['summed_epoch_seconds']/60
assert a['counts']['N']['epochs']==0 and a['admissions']['ADMISSION_N_6.json']==4
assert a['isolated_optimizer_calls']==42
for v in c['volumes'].values():assert v['free']-v['remaining_bound']-v['reserve']==v['headroom']>=0
resolution=dict(status='PASS_LATER_CAPACITY_SAMPLE_NOT_FINAL',first_evidence_sha256=sha(FIRST/'EVIDENCE.json'),later_evidence_sha256=sha(CAP/'EVIDENCE.json'),first_status=prior['status'],later_status=c['status'],root_free_recovered=c['volumes']['/root']['free']-prior['volumes']['/root']['free'],stable_credit_unchanged=c['volumes']['/root']['stable_unique_allocated']==prior['volumes']['/root']['stable_unique_allocated'],limitation='First stat sample excluded temporary files and cannot establish a persistent shortfall. Transient saving is consistent with recovered free space but exact temporary file was not recorded. Both observations retained; no deletion/restart/change.',operations=0)
(CAP/'LOCAL_AUDIT.json').write_text(json.dumps(resolution,indent=2),encoding='utf8')
ts=datetime.datetime.fromtimestamp(a['observation'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
s=js(R/'RUN_STATUS.json');s.update(status='PARTIAL_TRAINING',observation=ts,observation_unix=a['observation'],counts=a['counts'],own_admissions=a['admissions'],isolated_optimizer_calls=a['isolated_optimizer_calls'],teacher_bindings=a['teacher_bindings'],formal_episodes=0,phase=a['phase'],latest_evidence=D.name,capacity=e['capacity'],capacity_evidence=CAP.name,next_check_minutes=65,eta=dict(target='N whole training',remaining_steps=76100,own_throughput_measured=False,comparable_C_epoch_minutes=times['C'],planning_minutes=65,extra='save/cache/admission/preparation; new N auxiliary route throughput unmeasured',not_total_queue_eta=True))
(R/'RUN_STATUS.json').write_text(json.dumps(s,indent=2,ensure_ascii=False),encoding='utf8')
note=f'''# 最新2026-10-25 02:06 CST交接

实际观察{ts}，{D.name}/LOCAL_AUDIT实际PASS_PROGRESS_NOT_FINAL。65冻结项及父源不变；controller20530/GPU20976正常，C已自然完成并接续N Task6。B/C各400epoch76100主步，train_B/C.exit均0，合计800epoch152200步；各自4个CP记录SHA绑定。B/C epoch纯计算分别{times['B']:.6f}/{times['C']:.6f}分钟，各48额外梯度autograd单列，保存/教师/准入另计。B/C均已实际Task6 tensor同原ALIGN；全部16CP最终独立CPU加载与800闭环仍待后续，不计算离线损失为收益。
N自己Task6 GPU4隔离更新/0环境、100键21633611参数、完整下一步恢复及原ALIGN更新精确、长度1–8和部署语义准入实际PASS，snapshot时0完整正式epoch。N真实旧阶段状态化梯度测量须待各自Task7–9准入，不能用Task6报告代替。V未准入。累计自身42隔离更新，6份B/C阶段教师身份/tau/w与前阶段CP绑定通过，无本轮恢复重启或科学修改，正式环境0。
N自身吞吐尚未实测，以最近C完整epoch54.757810分钟作可比基线，保存/缓存/准入/准备另计，暂约60–65分钟量级。下一65分钟在N整组训练完成附近检查，参考03:05–03:15 CST；不是四组总结束ETA，不为内部Task切换短轮询。全部四组训练后才启动800正式评估。
本轮容量首次capacity_readonly_20261025_020440为REVIEW_CAPACITY：root扣预留余-54632775B；未立即判持续不足或操作。一次只读复查{CAP.name}实际PASS_NOT_FINAL：稳定已分配信用未变，free恢复67551232B，system普通保留1GiB及剩余峰值后余12918457B，data root保留2GiB后余20743291B。两份原证据与SHA及解释绑定LOCAL_AUDIT；与临时原子保存相容但未记录具体临时文件，不能宣称已证明原因。没有删除/重启/降预算/付费。下一轮需继续核实际容量，不把单次临时排除信用的负值直接当持续容量失败；原逐epoch容量保护继续。
唯一engineering02 root /data/mlr_ifa/experiment_workspaces/CCBP_V2_FULL_GOAL41_20261024_v1_engineering02、合同56ec3ca98bde2767e3b0a27170b2e8a71a633fbee2ba028b8a0630e7ff47ff71、controller20530不变。早期设备比较失败/完整恢复证据保留，不重跑deploy/build/controller。最终wrapper仍只静态待真实运行。整体PARTIAL。

'''
(R/'REPORT.md').write_text(note+(R/'REPORT.md').read_text(encoding='utf8'),encoding='utf8')
old=(R/'MONITOR_HANDOFF.md').read_text(encoding='utf8')
(R/'MONITOR_HANDOFF.md').write_text(note+'以下旧进度与ETA由顶部最新段落覆盖；科学合同和恢复限制继续有效。\n\n'+old,encoding='utf8')
r=js(R/'MONITOR_RECEIPT.json');r.update(interval_minutes=65,basis='N own throughput unmeasured; recent C full epoch compute54.757810min plus save/cache/admission/preparation',latest_evidence=D.name)
(R/'MONITOR_RECEIPT.json').write_text(json.dumps(r,indent=2,ensure_ascii=False),encoding='utf8')
print(json.dumps(dict(observation=ts,epoch_minutes=times,next_minutes=65,capacity_status=c['status'])))
