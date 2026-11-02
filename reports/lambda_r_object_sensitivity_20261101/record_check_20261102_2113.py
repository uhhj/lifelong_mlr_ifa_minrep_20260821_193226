"""Read-only evidence bookkeeping; no experiment calls."""
from pathlib import Path
import json,subprocess,sys,hashlib
R=Path(__file__).resolve().parent;E=R/'engineering_cross_lambda_20261102_v1';D=R/'check_readonly_20261102_211342';P=R/'check_readonly_20261102_205347'
js=lambda p:json.loads(p.read_text(encoding='utf8'))
def put(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
a=js(D/'LOCAL_AUDIT.json');v=js(D/'EVIDENCE.json');prior=js(P/'LOCAL_AUDIT.json');pv=js(P/'EVIDENCE.json')
assert a['status']=='PASS_PROGRESS_NOT_FINAL' and not a['failures']
interval=100;wall=(v['observed']-pv['observed'])/60
rates={}
for case in ['L005_43','L020_41']:
 h=json.loads(v['light'][case+'/EPOCHS_S.json']['text']);old=json.loads(pv['light'][case+'/EPOCHS_S.json']['text']);new=h[len(old):]
 rates[case]=dict(new_complete_epochs=len(new),new_accepted=a['counts'][case]['accepted_updates']-prior['counts'][case]['accepted_updates'],new_physical=a['counts'][case]['physical_optimizer_calls']-prior['counts'][case]['physical_optimizer_calls'],completed_epoch_seconds=sum(x['seconds'] for x in new),new_epochs_by_stage={str(st):sum(x['stage']==st for x in new) for st in range(6,10)},recent_current_stage_mean_seconds=a['counts'][case]['recent_epoch_mean_seconds'],current_stage_remaining_compute_minutes=a['counts'][case]['current_stage_compute_minutes_estimate'])
timing=dict(status='OBSERVED_THROUGHPUT_NOT_CONTROLLED_SPEEDUP',observation=a['observation'],snapshot_sha256=a['snapshot_sha256'],previous_snapshot_sha256=prior['snapshot_sha256'],wall_minutes=wall,by_case=rates,total_new_complete_epochs=a['epochs']-prior['epochs'],total_new_accepted=a['accepted_updates']-prior['accepted_updates'],total_new_complete_epoch_physical=a['physical_training_optimizer_calls']-prior['physical_training_optimizer_calls'],complete_epoch_physical_per_wall_minute=(a['physical_training_optimizer_calls']-prior['physical_training_optimizer_calls'])/wall,L00543_current_run_training_collection_planning_minutes=[95,120],next_check_minutes=interval,global_completion_date='NOT_REISSUED_FROM_EARLY_CROSS_LAMBDA_SAMPLES',limits=['The window contains L02041 Task6, acquisition, own admission, and early Task7 plus L00543 Task9; epochs are not equal work.','No matched serial counterfactual; do not call this a speedup ratio or reuse the prior 1.71 ratio.','L02041 Task8/9, L02042/43, and formal evaluation are not yet measured under this schedule.','Current-run pure-compute estimate excludes preparation, saving and last ten acquisition episodes; future resource contention can change it.','Healthy progress; no extra scientific diagnostics, retries or environments.'])
put(D/'TIMING_AUDIT.json',timing)
proof=dict(status='PASS_NEW_L020_TASK7_OWN_ADMISSION_AND_PREFIX',snapshot_sha256=a['snapshot_sha256'],task7_admission=json.loads(v['light']['L020_41/ADMISSION_S_7.json']['text']),task6_gpu_comparison=json.loads(v['light']['engineering_rendezvous_recovery_20261102_v1/L020_41_TASK6_GPU_COMPARISON.json']['text']),environment_mutex=a['environment_mutex_audit'],rendezvous=a['rendezvous_audit'],handoff_complete=False,old_controller11239_frozen=next(x for x in v['processes'] if x['pid']==11239)['state']=='T',no_production_restart=True)
assert proof['old_controller11239_frozen'] and 'engineering_cross_lambda_20261102_v1/HANDOFF_COMPLETE.json' not in v['light']
put(D/'TRANSITION_AUDIT.json',proof)
history=(R/'MONITOR_HANDOFF.md').read_text(encoding='utf8').split('# OBJECT 最新S固定λ_R六run：当前交接')[0]
subprocess.run([sys.executable,'-X','utf8',str(E/'record_progress.py'),str(D),str(interval)],check=True)
intro=f'''## 2026-11-02 21:13 实际复查：跨组训练健康，0.2首个保持阶段自身准入通过

{D.name}及工程audit_progress实际PASS，205347全部epoch/门控/诊断前缀保持。L00543 Task9完整24/100，总324轮77988接受140276生产91隔离41准入autograd30采集；L02041 Task6完整100轮及10采集，Task7完整14轮，总114轮18878接受22056生产33隔离19准入autograd。Task7自己的29/11、完整恢复/10步重放/q1/关保持/测量探针状态不变通过，BRANCH_COPY_EQ/CHECK_EQ实际起点通过，Task6对已有S全部tensor这次实际相等；其他未来阶段不能套PASS。L02041 Task6十次采集一成功、13窗口，q6=.1；训练采集和门控不是正式闭环收益。

累计1238轮307866接受552932完整轮生产306隔离142准入autograd120采集0正式，L00541/42保持各400轮40采集；L02042/43未开始。135科学及全部工程hash不变，controller11852/watchdog11854和GPU11246/11860正常，独立端口42941/42689保持。原等待11239仍T，HANDOFF_COMPLETE尚不存在，严禁手动恢复。6/18真实采集锁完整acquired/released且互斥，未来12采集及6正式锁待核；原失败与540已记放弃调用+UNKNOWN成本、35418/94工程账目均保留。

L00543近10轮均70.1911秒，余76轮约88.91分钟纯计算；保存/最后10采集另计，当前整run规划95–120分钟。L02041 Task7近10均33.2950秒，余86轮47.72分钟纯计算，其余阶段尚未实测。两次观察间{wall:.4f}分钟新增128完整轮和33893完整轮物理调用，是混合Task6/7/9及采集准入窗口；没有同工作量串行对照，不能称新的严格提速倍数或套旧1.71倍，也不据此重发全队列完成保证。TIMING_AUDIT保存实际增量与边界。

下一100分钟在L00543自然训练完成附近复查真实根锁交接、优先L005正式与L02041继续训练/环境互斥；覆盖旧一次性30分钟，后续按当前整run/整组正式实际余时调整，不固定高频、不以时间推断完成。正常推进无新需用户行动，健康静默。以下20:53部署描述仅作工程约束和追溯，其旧进度/间隔以本段覆盖。

'''
for n in ['MONITOR_HANDOFF.md','REPORT.md']:
 p=R/n;p.write_text(intro+history+p.read_text(encoding='utf8'),encoding='utf8')
s=js(R/'RUN_STATUS.json');s.update(status='PARTIAL_CROSS_LAMBDA_TRAINING',timing_audit=str(D/'TIMING_AUDIT.json'),transition_audit=str(D/'TRANSITION_AUDIT.json'));put(R/'RUN_STATUS.json',s)
x=js(R/'ENGINEERING_CROSS_LAMBDA_STATUS.json');x.update(latest_observation=a['observation'],latest_evidence=D.name,latest_progress_status=a['status'],next_check_minutes=interval);put(R/'ENGINEERING_CROSS_LAMBDA_STATUS.json',x)
print(json.dumps(dict(status=s['status'],next_check_minutes=interval,wall_minutes=wall,epochs=a['epochs'])))
