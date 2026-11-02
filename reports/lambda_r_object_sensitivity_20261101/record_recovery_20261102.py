"""Record observed recovery, preserving prior pause/interruption evidence."""
from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parent;E=R/'engineering_resume_3090_20261102_v1';D=R/'check_readonly_20261102_124955'
def js(p):return json.loads(p.read_text(encoding='utf8'))
def put(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=js(D/'LOCAL_AUDIT.json');v=js(D/'EVIDENCE.json');assert a['status']=='PASS_PROGRESS_NOT_FINAL' and a['epochs']>=261
old=js(R/'check_readonly_20261101_232839/EVIDENCE.json');assert json.loads(v['light']['L005_41/EPOCHS_S.json']['text'])[:260]==json.loads(old['light']['L005_41/EPOCHS_S.json']['text'])
for n,x in old['light'].items():
 if '/GATES_stage' in n:assert v['light'][n]['text'].startswith(x['text'])
h=json.loads(v['light']['L005_41/EPOCHS_S.json']['text']);fresh=h[260:]
st=js(R/'RUN_STATUS.json');st.update(status='PARTIAL_RESUMED_TRAINING_RTX3090',scientific_completion='PARTIAL_NOT_ACCEPTED',automatic_monitor_status='ACTIVE',engineering_state='POST_REBOOT_RECOVERY_RUNNING',current_training_pid=2566,next_check_minutes=30,remaining_time_estimate='Pending stable post-recovery throughput; old slow-host ETA invalid',resume_report='RECOVERY_REPORT_20261102.md',prior_completed_epochs_preserved=260)
st['counts']['L005_41']['current_stage_compute_minutes_estimate']=None
st['counts']['L005_41']['recent_epoch_mean_seconds']=sum(x['seconds'] for x in fresh)/len(fresh)
st['counts']['L005_41']['timing_note']='Only post-recovery completed epochs; small sample, not stable throughput. Recovered epoch uses a fresh timer, no paused wall time.'
put(R/'RUN_STATUS.json',st)
ac=js(R/'ACTIVE_ROOT.json');ac.update(status=st['status'],engineering_state='POST_REBOOT_RECOVERY_RUNNING',current_training_pid=2566,latest_evidence=D.name);put(R/'ACTIVE_ROOT.json',ac)
u=js(R/'USER_RESUME_STATUS.json');u.update(status='RESUMED_VERIFIED_PRODUCTION',observation=a['observation'],latest_evidence=D.name,automatic_monitor_status='ACTIVE',current_completed_epochs=a['epochs'],prior_completed_epochs_preserved=260,all_original_gates_prefix_exact=True);put(R/'USER_RESUME_STATUS.json',u)
validation=js(R/'capacity_readonly_20261102_124634/EVIDENCE.json')['files']['VALIDATION.json']
audit=dict(status='PASS_RECOVERY_PROGRESS_NOT_FINAL',observation=a['observation'],snapshot_sha256=sha(D/'EVIDENCE.json'),original_260_epochs_exact=True,all_original_gate_prefix_exact=True,actual_production_restore=json.loads(v['light'][E.name+'/PRODUCTION_RESTORE.json']['text']),own_GPU_validation_calls=309,own_GPU_validation_autograd=11,own_GPU_validation_environments=0,prior_engineering_calls=7828,prior_engineering_autograd=36,abandoned_partial_recorded_calls=140,abandoned_partial_unrecorded_cost='UNKNOWN',seven_old_gate_records_exact=all(x['all_gate_fields_exact'] for x in validation['old_GPU_comparisons']),post_recovery_epochs=len(fresh),post_recovery_seconds=[x['seconds'] for x in fresh],no_full_run_cross_GPU_equality_claim=True,controller=2560,training=2566,contract=ac['engineering_contract'],future_own_admissions_required=True)
put(D/'RECOVERY_LOCAL_AUDIT.json',audit)
firstfail=js(R/'capacity_readonly_20261102_124408/EVIDENCE.json')['files']['LEDGER.json'];assert firstfail['started']==firstfail['completed']==firstfail['extra_autograd']==0
put(R/'LOCAL_TRANSPORT_RENEWAL_20261102.json',dict(prior_worker_pid=28064,prior_worker_actual_process_absent=True,new_worker_pid=3700,hidden_start=True,pending_requests_before_start=0,STOP_absent=True,credentials_printed=False,active_worker_restarted=False))
report=f'''# 2026-11-02 RTX3090 恢复：PARTIAL，尚未科学最终验收

用户先要求继续训练，发现主机在暂停后重启、GPU由RTX4080SUPER改成RTX3090后，明确授权“允许在3090核验后续训”。原进程全部消失，暂停内存不可恢复。首次SIGCONT包装在任何信号前因旧PID不存在而退出，没有重启或重复生产。CPU检查确认原260轮、checkpoint完整state/优化器/scheduler/grads/modes/requires及全部RNG、历史、probe SHA一致。

独立GPU验证v1因probe scale重算差一个float32 ULP（约2.98e-8）在0optimizer/0autograd/0环境前停止，错误保留。v2精确使用原冻结Task8归一化常数（不是调参或放宽门控），核完整GPU恢复、29调用原准入、7块两条CPU/device快照路径。实际309 started==completed、11额外autograd、0环境、0生产轮；每步标量、两分支恢复、所选完整state/探针/决策精确。7块与旧GPU原记录全部逐字段相等。该观察只覆盖验证路径，不能宣称全run跨GPU逐位等价。

唯一新工程 `{E.name}`，12:48:22启动controller2560、训练2566，合同 `{ac['engineering_contract']}`，8冻结项；135科学及5旧工程冻结项保持。原root和旧工程controller.exit均不存在，最终实际成功来源只使用新工程controller.exit0且无自身进程。禁止重跑旧/新任何dispatch/controller/build/已完成轮。

原checkpoint另存私有interrupted_epoch260.pth，原4167条门控完整归档且原文件全部保留。第261轮前7块重建须逐字段等于已有记录，不重复追加门控行；前260轮不重训。中断部分已记录140候选调用与未记账部分UNKNOWN单列，不能把它说成零；309恢复诊断与此前7828诊断分开，合计8137optimizer/47autograd，不混入1171800生产预算。

实际 {a['observation']}，{D.name} / LOCAL_AUDIT / RECOVERY_LOCAL_AUDIT通过。当前{a['epochs']}轮，接受{a['accepted_updates']}、已完成轮生产物理{a['physical_training_optimizer_calls']}，自身62隔离/30autograd、20采集、正式0。原260轮与全部门控前缀精确保留，controller2560/GPU2566健康；其余五run未启动。恢复后完整轮秒数 {audit['post_recovery_seconds']}，样本不足以推算全队列；旧慢速ETA失效。

后续顺序及loss/λ/阈值/采集/正式预算不变。每阶段自己的GPU准入和设备快照一致性核验继续。新run仍要求官方初始化tensor/RNG相等；用户已授权硬件变化，Task6跨GPU结果实际记录全部差异，不再伪称硬件相同或tensor必相等。L005_41前260轮在旧硬件、后续在3090，其余新run在3090，旧0.1基线不重跑。因此最终报告必须披露硬件混杂，不能称λ单因素硬件完全受控对照。

新工程check/audit_progress/record_progress/final_readonly/audit_final_local已外置适配；最终CPU仍STATIC_ONLY，待全部自然结束才真实24CP/6恢复/24teacher/1200初态noise/54600门控与恢复成本验收。之后根compare_primary_s、三档全结果/新旧任务/SF/FS/负单元/成本/开发集及预训练来源边界解释，发布后关闭监控。

同一object-s-r-run已ACTIVE，下一30分钟一次性核恢复吞吐，再依据当前完整run/整组评估剩余时间调整。现有运输worker3700，先核status/PID，不重启活跃worker或输出凭据。健康静默，只实质变化/完成/新失败/需行动通知。
'''
(R/'RECOVERY_REPORT_20261102.md').write_text(report,encoding='utf8')
header=report.replace('# 2026-11-02 RTX3090 恢复：PARTIAL，尚未科学最终验收','## 最新2026-11-02：用户授权3090核验后已实际续训')+'\n本段覆盖此前PAUSED与旧调度器安排，历史暂停/失败证据永久保留。\n\n'
for n in ['MONITOR_HANDOFF.md','REPORT.md']:(R/n).write_text(header+(R/n).read_text(encoding='utf8'),encoding='utf8')
review=js(E/'STATIC_REVIEW.json');review.update(deployment='deploy_remote_20261102_124815',actual_GPU_validation='capacity_readonly_20261102_124634',actual_production_restore=D.name,final_CPU_acceptance='STATIC_ONLY_PENDING_ACTUAL',files={p.name:sha(p) for p in E.glob('*.py')});put(E/'STATIC_REVIEW.json',review)
print(json.dumps(dict(status=st['status'],epochs=a['epochs'],post_recovery_seconds=audit['post_recovery_seconds'],automatic_monitor='ACTIVE')))
