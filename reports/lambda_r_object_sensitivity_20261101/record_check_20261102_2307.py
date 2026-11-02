"""Record the actual natural handoff and measured evaluation timing; no remote calls."""
from pathlib import Path
import json, subprocess, sys, statistics
R=Path(__file__).resolve().parent
E=R/'engineering_cross_lambda_20261102_v1'
D=R/'check_readonly_20261102_230720'
js=lambda p:json.loads(p.read_text(encoding='utf8'))
def put(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
a=js(D/'LOCAL_AUDIT.json');v=js(D/'EVIDENCE.json');l=v['light']
assert a['status']=='PASS_PROGRESS_NOT_FINAL' and not a['failures']
handoff=json.loads(l[E.name+'/HANDOFF_COMPLETE.json']['text'])
assert handoff['root_lock'] and handoff['returncode']==0 and handoff['training_signals']==0
assert l['engineering_rendezvous_recovery_20261102_v1/L005_43_resume.exit']['text'].strip()=='0'
assert not any(p['pid'] in (11239,11241,11246,11854) for p in v['processes'])
for case in ('L005_41','L005_42','L005_43'):
 assert a['counts'][case]['epochs']==400 and a['counts'][case]['training_collection_completed_episodes']==40
admission=json.loads(l['L020_41/ADMISSION_S_8.json']['text'])
assert admission['status']=='PASS_OWN_GPU_ADMISSION'
evaluation=json.loads(l['L005_41/EVAL_ADMISSION.json']['text'])
assert evaluation['status']=='PASS' and evaluation['updates']==evaluation['environments']==0
events=[json.loads(s) for s in l['L005_41/EVAL_EVENTS.jsonl']['text'].splitlines()]
complete=[x for x in events if x['kind']=='complete']
mean=statistics.mean(x['seconds'] for x in complete)
recent=statistics.mean(x['seconds'] for x in complete[-5:])
interval=65
timing=dict(status='MEASURED_PARTIAL_EVALUATION_NOT_FINAL',observation=a['observation'],snapshot_sha256=a['snapshot_sha256'],completed_vector_events=len(complete),outcome_episodes=a['formal_episodes'],mean_vector_seconds=mean,recent5_vector_seconds=recent,remaining_vectors_conservative=60-len(complete),remaining_L005_linear_minutes=[(60-len(complete))*x/60 for x in (mean,recent)],remaining_L005_planning_minutes=[65,90],L02041_task8_remaining_compute_minutes=a['counts']['L020_41']['current_stage_compute_minutes_estimate'],next_check_minutes=interval,global_completion_estimate='NOT_REISSUED',limits=['Snapshot files are read sequentially: 180 outcome episodes but 17 complete vector events; use 43 remaining vectors conservatively.','Different tasks and seeds have different episode termination lengths; evaluation and collection share a global mutex.','L02041 may wait for collection access; its Task9 and remaining two runs have not been measured under this schedule.','No controlled serial comparison or new speedup factor. Final CPU acceptance, interpretation and publication are additional.'])
put(D/'TIMING_AUDIT.json',timing)
proof=dict(status='PASS_ACTUAL_NATURAL_HANDOFF_AND_L005_TRAINING_COMPLETE',snapshot_sha256=a['snapshot_sha256'],handoff=handoff,resumed_L00543_exit=0,old_waiter_and_launcher_absent=True,L02041_task8_admission=admission,L00541_evaluation_admission=evaluation,environment_mutex=a['environment_mutex_audit'],original_failures_preserved=a['mitigated_failures_preserved'],final_acceptance='PENDING')
put(D/'TRANSITION_AUDIT.json',proof)
history=(R/'MONITOR_HANDOFF.md').read_text(encoding='utf8').split('# OBJECT 最新S固定λ_R六run：当前交接')[0]
subprocess.run([sys.executable,'-X','utf8',str(E/'record_progress.py'),str(D),str(interval)],check=True)
intro=f'''## 2026-11-02 23:07 实际复查：0.05全部训练完成，自然交接后正式评估进行中

{D.name}及工程audit_progress实际PASS，211342全部epoch/门控/诊断前缀精确保留。L005三个seed各400轮、105500接受/195300生产、40采集完成。43真实成功来源为前恢复工程L005_43_resume.exit0，原train_S.exit1及旧live失败SHA仍完整保留。cross/HANDOFF_COMPLETE实际root_lock=true、原43 launcher自然returncode0、training_signals0；旧等待11239及43 launcher/GPU、watchdog已不存在，controller11852正常接管。禁止恢复或重跑任何旧新controller。

已优先进入L005正式：41完成180/200，42/43尚未开始，合计180/600。41自己的评估准入PASS，0更新/0环境；这是评估准入的计数，不是正式回合没有环境。L02041继续Task8完整65轮，共265轮58485接受101270生产62隔离30准入autograd20采集，Task8自己的29/11及BRANCH_COPY_EQ/CHECK_EQ通过；42/43未启动。累计1465轮374985接受687170完整轮物理335隔离153准入autograd140采集180正式。135科学及全部工程项不变，controller11852、训练11860及评估12856健康，无新失败或重训。

环境锁已实际核8/18采集完整acquired/released，首个正式锁已取得、尚未释放，合并时间线互斥通过；未发生的10采集及余5正式锁仍待核。L02041后续采集可能正常等待正式释放，不能误判为训练故障。全部接受前保留自有pth和HOOK_ENABLED。

实际17个完整vector事件均{mean:.4f}秒、近5均{recent:.4f}秒。快照逐文件读取，outcome已180回合而事件尾部只完整17个vector；按保守余43vector估L005正式余{timing['remaining_L005_linear_minutes'][0]:.2f}–{timing['remaining_L005_linear_minutes'][1]:.2f}分钟，计任务/seed差异、互斥等待及保存调度规划65–90分钟。L02041 Task8近10轮69.6262秒、余35轮40.62分钟纯计算，Task9及后两个run未测，不重发全队列保证。下一65分钟围绕L005整组正式结束较早侧复查；仍PARTIAL，最终CPU验收、解释发布另计，不按时钟推断完成。

工程35418optimizer/94autograd和540已记放弃候选+UNKNOWN不变，硬件迁移及开发集/预训练来源边界保留。下方21:13与20:53仅历史追溯，HANDOFF尚未发生、旧PID与旧进度/间隔均以本段覆盖。

'''
for name in ('MONITOR_HANDOFF.md','REPORT.md'):
 p=R/name;p.write_text(intro+history+p.read_text(encoding='utf8'),encoding='utf8')
s=js(R/'RUN_STATUS.json');s.update(status='PARTIAL_L005_EVALUATION_L020_TRAINING',timing_audit=str(D/'TIMING_AUDIT.json'),transition_audit=str(D/'TRANSITION_AUDIT.json'));put(R/'RUN_STATUS.json',s)
x=js(R/'ENGINEERING_CROSS_LAMBDA_STATUS.json');x.update(latest_observation=a['observation'],latest_evidence=D.name,latest_progress_status=a['status'],root_lock_handoff='PASS_ACTUAL_NATURAL_EXIT_AND_ROOT_LOCK',latest_handoff=handoff,next_check_minutes=interval);put(R/'ENGINEERING_CROSS_LAMBDA_STATUS.json',x)
x=js(R/'ACTIVE_ROOT.json');x.update(engineering_state='HANDOFF_COMPLETE_L005_EVALUATION_L020_TRAINING',watchdog=None,current_training_pid=11860,active_training_pids={'L020_41':11860},active_evaluation_pid=12856);put(R/'ACTIVE_ROOT.json',x)
print(json.dumps(timing,ensure_ascii=False,indent=2))
