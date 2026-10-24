"""Close the local progress round; never mutate server science."""
import json,hashlib,datetime
from pathlib import Path
R=Path(__file__).resolve().parent;D=R/'check_readonly_20261025_030945';C=R/'capacity_readonly_20261025_031011'
js=lambda p:json.loads(p.read_text(encoding='utf8'))
a=js(D/'LOCAL_AUDIT.json');e=js(D/'EVIDENCE.json');cap=js(C/'EVIDENCE.json')
assert a['failure'] is None and a['formal_episodes']==0
times={}
for arm in 'BCN':
 assert (D/f'light/train_{arm}.exit').read_text().strip()=='0'
 t=js(D/f'light/TRAIN_{arm}.json');assert (t['epochs'],t['steps'],t['gradient_autograd_calls'])==(400,76100,48)
 assert js(D/f'light/TASK6_PARITY_{arm}.json')['equal']
 times[arm]=t['summed_epoch_seconds']/60
assert a['counts']['V']['epochs']==70 and a['admissions']['ADMISSION_V_6.json']==4 and a['isolated_optimizer_calls']==58
for v in cap['volumes'].values():assert v['free']-v['remaining_bound']-v['reserve']==v['headroom']>=0
(C/'LOCAL_AUDIT.json').write_text(json.dumps(dict(status='PASS_NOT_FINAL',evidence_sha256=hashlib.sha256((C/'EVIDENCE.json').read_bytes()).hexdigest(),volumes=cap['volumes'],scope='stat-only stable unique inode and reserve arithmetic; no final acceptance'),indent=2),encoding='utf8')
hist=js(D/'light/EPOCHS_N.json');later=sum(row['seconds'] for row in hist if row['stage']>6)
linear=(30*a['rates_last10']['V']['6']+later)/60
ts=datetime.datetime.fromtimestamp(a['observation'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
s=js(R/'RUN_STATUS.json');s.update(status='PARTIAL_TRAINING',observation=ts,observation_unix=a['observation'],counts=a['counts'],own_admissions=a['admissions'],isolated_optimizer_calls=a['isolated_optimizer_calls'],teacher_bindings=a['teacher_bindings'],formal_episodes=0,phase=a['phase'],latest_evidence=D.name,capacity=e['capacity'],capacity_evidence=C.name,next_check_minutes=60,eta=dict(target='V whole training, completing all four training arms',remaining_steps=69100,own_task6_rate=a['rates_last10']['V']['6'],comparable_N_later_stage_seconds=later,linear_minutes=linear,planning_minutes=60,extra='save/cache/admission/preparation; V auxiliary stages not yet measured',not_evaluation_completion_eta=True))
(R/'RUN_STATUS.json').write_text(json.dumps(s,indent=2,ensure_ascii=False),encoding='utf8')
note=f'''# 最新2026-10-25 03:11 CST交接

实际观察{ts}，{D.name}/LOCAL_AUDIT实际PASS_PROGRESS_NOT_FINAL。65冻结项及父源不变，controller20530/GPU21239健康，N已自然exit0后接续V Task6。B/C/N各完整400epoch76100主步，共1200epoch228300步，12阶段CP记录及train_B/C/N.exit0绑定；N epoch纯计算{times['N']:.6f}分钟，48额外梯度autograd另列，B/C此前54.142874/54.757810min保留，保存/准入/教师另计。B/C/N各Task6 tensor与原ALIGN实际相同，但全部16CP最终独立CPU审计仍待后续。
V Task6下载70完整epoch7000步；自己4隔离GPU调用、完整下一步恢复、原ALIGN更新精确/100键/21633611参数/1–8长度/部署语义准入实际PASS，不能套给V旧阶段。累计58自身隔离优化器更新/0环境，N Task7–9自己的状态化测量与恢复已实际PASS。九份B/C/N阶段教师身份/尺度/tau/w/自身前阶段CP绑定本地通过。正式环境0，无恢复重启改科学参数。
V Task6最近10轮均{a['rates_last10']['V']['6']:.6f}秒，余30轮；V后续辅助阶段未实测，借N Task7–9纯计算{later/60:.6f}min，合计剩余基线{linear:.6f}min，保存/缓存/准入/准备另计。按用户预计完成点规则，下一60分钟检查V整组主训练结束附近（参考04:05–04:15 CST），届时若自然进入评估按新实际vector吞吐估全段剩余。该ETA不含800仿真/最终独立验收，不因内部Task切换短轮询。
{C.name}/LOCAL_AUDIT实际PASS_NOT_FINAL：system普通free1456066560B、稳定已分配779481088B，扣余新增369422663B及1GiB后余12902073B；data rootfree2791370752B、稳定已分配1394098176B，扣余新增624536453B及2GiB后余19350651B。只stat、0payload/前向/更新/环境/服务器写入，未删除或新增付费。此前短暂REVIEW_CAPACITY和一次复查恢复的原证据保留，不改写为从未出现。
唯一engineering02 root /data/mlr_ifa/experiment_workspaces/CCBP_V2_FULL_GOAL41_20261024_v1_engineering02、合同56ec3ca98bde2767e3b0a27170b2e8a71a633fbee2ba028b8a0630e7ff47ff71、controller20530不变。全部四组训练结束后才800正式；final_readonly/audit_final_local仍只静态，须自然controller.exit0无自身进程后实际运行验收。不得重跑deploy/build/controller或已完成训练。整体PARTIAL，无局部科学收益结论。

'''
(R/'REPORT.md').write_text(note+(R/'REPORT.md').read_text(encoding='utf8'),encoding='utf8')
old=(R/'MONITOR_HANDOFF.md').read_text(encoding='utf8')
# Keep the scientific contract and first repair lineage without accumulating obsolete round ETAs.
contract=old[old.index('继续用户授权CCBP-v2完整GOAL'):]
(R/'MONITOR_HANDOFF.md').write_text(note+'以下历史进度和ETA均由顶部覆盖；科学合同与工程恢复限制保持。\n\n'+contract,encoding='utf8')
r=js(R/'MONITOR_RECEIPT.json');r.update(interval_minutes=60,basis='V remaining Task6 own timing plus N actual later stages; approximately '+str(round(linear,3))+' min computation plus preparation',latest_evidence=D.name)
(R/'MONITOR_RECEIPT.json').write_text(json.dumps(r,indent=2,ensure_ascii=False),encoding='utf8')
print(json.dumps(dict(observation=ts,N_epoch_minutes=times['N'],V_remaining_reference_minutes=linear,next_minutes=60)))
