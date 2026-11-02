"""Record actual recovery evidence and honest incomplete cost, without modifying production."""
from pathlib import Path
import json,hashlib,subprocess,sys,datetime
E=Path(__file__).resolve().parent;R=E.parent;D=R/'check_readonly_20261102_203858'
js=lambda p:json.loads(p.read_text(encoding='utf8'))
def put(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
a=js(D/'LOCAL_AUDIT.json');v=js(D/'EVIDENCE.json');old=js(R/'check_readonly_20261102_201342/EVIDENCE.json')
def lj(n):return json.loads(v['light'][n]['text'])
assert a['status']=='PASS_PROGRESS_NOT_FINAL' and a['counts']['L005_43']['epochs']==292
hist=lj('L005_43/EPOCHS_S.json');assert hist[:289]==json.loads(old['light']['L005_43/EPOCHS_S.json']['text'])
restore=lj(E.name+'/PRODUCTION_RESTORE.json');assert restore['status']=='PASS_FULL_STATE_RESTORED' and restore['step']==65901
assert [x['epoch'] for x in hist[289:]]==[89,90,91]
for n,x in old['light'].items():
 if n.startswith(('L005_41/','L005_42/')):assert v['light'][n]['sha256']==x['sha256']
startup=dict(status='PASS_ACTUAL_RESUME_THREE_COMPLETE_EPOCHS',snapshot=D.name,snapshot_sha256=a['snapshot_sha256'],controller=11239,launcher=11241,training_pid=11246,contract='23f1dda97de2ebd9a1b752db2f34ab23ccd4e2f4ed9461ce4e0d99887a5fa7c4',frozen_engineering_files=15,original_epochs_exact=289,original_partial_gate_prefix_exact=True,original_complete_run_metadata_exact=['L005_41','L005_42'],complete_epochs_after_resume=3,full_restore=restore,original_failures_preserved=a['mitigated_failures_preserved'],diagnostic_optimizer_calls=35418,diagnostic_extra_autograd=94,rendezvous_partial_recorded_optimizer_calls=400,total_recorded_abandoned_partial_optimizer_calls=540,unrecorded_cost='UNKNOWN',final_acceptance=False)
put(D/'RECOVERY_STARTUP_AUDIT.json',startup);put(R/'ENGINEERING_RENDEZVOUS_RECOVERY_STATUS.json',dict(startup,status='PARTIAL_RESUMED_TRAINING',next_check_minutes=30))
recent=[x['seconds'] for x in hist[289:]];mean=sum(recent)/len(recent)
reference={}
for case in ['L005_41','L005_42']:
 hh=lj(case+'/EPOCHS_S.json');rr=[x['seconds'] for x in hh if x['stage']==9][-10:];reference[case]=sum(rr)/len(rr)
timing=dict(status='CONDITIONAL_PLANNING_NOT_COMPLETION',observation=a['observation'],resumed_epoch_seconds=recent,resumed_epoch_mean_seconds=mean,remaining_Task8_epochs=8,remaining_Task8_compute_minutes=8*mean/60,Task9_unmeasured_for_seed43=True,Task9_reference_recent10_seconds=reference,current_run_compute_minutes_range=[8*mean/60+100*x/60 for x in sorted(reference.values())],current_run_training_collection_planning_hours=[2,2.5],L020_training_collection_prior_planning_hours=[9,11],formal_1200_unmeasured_planning_hours=[2,4],all_remaining_training_collection_formal_hours=[13,18],CPU_acceptance_report_publication_extra=True,old_1835_completion_date_superseded_by_interruption=True,limits=['Only 3 resumed Task8 epochs; single-task timing is not parallel acceleration.','Task9 own success caches are not measured yet; historical rates are conditional.','L020 two slots include a single-job tail, preparation and serialized collection; no multiplication of earlier 1.71 ratio.','Formal timing is historical allowance, not measured this round.','Previous failed launcher and recovery gap invalidate the old completion promise.'])
put(D/'TIMING_AUDIT.json',timing)
archive=E/'MONITOR_HANDOFF_BEFORE_RECOVERY.md'
if not archive.exists():archive.write_bytes((R/'MONITOR_HANDOFF.md').read_bytes())
active=js(R/'ACTIVE_ROOT.json');active.update(engineering_state='PASS_ACTUAL_RESUMED_TRAINING',current_training_pid=11246,active_training_pids={'L005_43':11246},hardware_migration='USER_AUTHORIZED_RTX3090');put(R/'ACTIVE_ROOT.json',active)
subprocess.run([sys.executable,'-X','utf8',str(E/'record_progress.py'),str(D),'30'],check=True)
intro=f'''## 2026-11-02 20:39：并行启动器故障已核实并恢复，原失败永久保留

20:13发现旧live controller.exit1，L00543原train_S.exit1。L00542自然400轮40采集exit0之后，PyTorch1.13的--standalone强制共用localhost:29400，43失去TCPStore后由自身torchrun发SIGTERM；不是本次已观察到的GPU故障。完成41/42与43原289轮、135科学及全部历史工程项不变。旧等待器已结束，禁止恢复任何旧controller。

新engineering_rendezvous_recovery_20261102_v1于deploy_remote_20261102_203611唯一部署，controller11239/launcher11241/GPU11246；合同23f1dda97de2ebd9a1b752db2f34ab23ccd4e2f4ed9461ce4e0d99887a5fa7c4，15项。已实际两个独立端口生命周期PASS、869optimizer/11autograd/0环境完整恢复、20原门控块逐字段及下一块双路径精确PASS，诊断已退出。首个下划线参数修复前零成本参数解析失败保留。禁止重复dispatch/controller/validation/build或已完成轮。

最新{D.name}实际审计PASS，203637全部前缀保留；43恢复后的3轮实际完成，现292/400(Task8 92)，66828接受117956完整轮生产、62隔离30准入autograd20采集。41/42各400轮40采集；L020未开始。累计1092轮277828接受508556生产244自身隔离112autograd100采集0正式。原43的289轮和5079门控行精确保留，重建20块不重复追加。完整恢复及前三轮证据见RECOVERY_STARTUP_AUDIT。

原43/train_S.exit1、旧live/controller.exit1及FAILURE.json按合同SHA永久保留，不伪造成功。新43退出来源为新工程L005_43_resume.exit，最终全部完成来源只用新工程controller.exit0且无自身进程。根、旧设备/恢复/检查/原parallel出口仍不存在，旧live失败出口必须存在。只用新工程check/audit/record/final包装；六CPU最终wrapper已静态适配真实新来源和43特殊失败/恢复出口，仍STATIC_ONLY。

工程累计35418optimizer/94autograd单列；旧硬件中断140+本次400=540已记被放弃候选调用，未记账部分仍UNKNOWN。新增私有备份346445621B已计容量。不能声称总实际成本精确。硬件迁移/开发集复用/预训练来源未闭合解释不变。

恢复3轮均{mean:.4f}秒，只是Task8单任务小样本，不作新并行提速结论。当前43余训练采集暂按2–2.5小时，L0209–11小时及1200正式2–4小时均条件参考；总余13–18小时(20:39起约11月3日09:40–14:40)，CPU验收解释发布另计。旧18:35 ETA已被真实故障覆盖。下一30分钟一次性核恢复后稳定吞吐与Task9自己的准入，再按整run/整组正式余时调整，不固定短轮询。

未来L020仍最多两训练并行，显式--rdzv_backend=c10d --rdzv_endpoint=localhost:0 --rdzv_id=唯一UUID，不用--standalone；核实际端口不同。采集hook及18次事件仍在旧live目录，当前4次互斥已通过，其余14次待核；不得提前移除hook。全部最终验收后核无目标collect并核SHA，只清理自有pth和旧live/HOOK_ENABLED，保留证据。未来每阶段自己的GPU准入/BRANCH_COPY_EQ/CHECK_EQ，科学顺序.05全训练→600正式→.2全训练→600正式不变。

先完整读RENDEZVOUS_RECOVERY_REPORT_20261102.md、ENGINEERING_RENDEZVOUS_RECOVERY_STATUS.json、新工程STATIC_REVIEW/AUDIT_WRAPPER_REVIEW及最新证据。运输worker3700先核status/PID，不重启活跃worker、不输出凭据。失败先核进程/冻结/恢复/真实成本，不能盲重启。

'''
for n in ['MONITOR_HANDOFF.md','REPORT.md']:
 p=R/n;p.write_text(intro+p.read_text(encoding='utf8'),encoding='utf8')
for n in ['IMPLEMENTATION_NOTES.md']:
 p=R/n;p.write_text(p.read_text(encoding='utf8')+'\n\n'+intro,encoding='utf8')
put(R/'FINAL_WRAPPER_REVIEW.json',dict(js(E/'AUDIT_WRAPPER_REVIEW.json'),latest_engineering=E.name,actual_final_CPU_acceptance='PENDING',special_case_completion='L00543 original train_S.exit1 remains; new L005_43_resume.exit0 required',prior_failure_mitigation='Exact three path SHA only; no blanket failure exclusion'))
status=js(R/'RUN_STATUS.json');status.update(status='PARTIAL_RESUMED_TRAINING',recovery_startup_audit=str(D/'RECOVERY_STARTUP_AUDIT.json'),timing_audit=str(D/'TIMING_AUDIT.json'));put(R/'RUN_STATUS.json',status)
print(json.dumps(dict(status=status['status'],epochs=a['epochs'],resumed_complete_epochs=3,mean_seconds=mean,next_check_minutes=30),ensure_ascii=False))
