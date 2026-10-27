"""Reconcile all training completion and evaluation progress, metadata only."""
from pathlib import Path
import json,hashlib,datetime,math,tomllib
R=Path(__file__).resolve().parent
D=R/'check_readonly_20261027_201533';L=D/'light';P=R/'check_readonly_20261027_184804/light';K=R/'capacity_readonly_20261027_201558'
js=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=js(D/'LOCAL_AUDIT.json');v=js(D/'EVIDENCE.json');active=js(R/'ACTIVE_ROOT.json');cap=js(K/'EVIDENCE.json')
assert a['status']=='PASS_PROGRESS_NOT_FINAL' and not a['failure'] and a['contract']==active['contract'] and v['children']==active['children'] and v['source_files']==68
assert a['phase']['case']=='GOAL_42' and a['phase']['phase']=='evaluate' and a['phase']['controller']==2559 and v['gpu'].strip()=='3732'
training={};prefixes={};parities={};rows_by={}
for case in ['GOAL_42','GOAL_43','OBJECT_41','OBJECT_42','OBJECT_43']:
 rows=js(L/case/'EPOCHS_C.json');previous=js(P/case/'EPOCHS_C.json');assert rows[:len(previous)]==previous and len(rows)==400
 rows_by[case]=rows;prefixes[case]=len(previous)
 assert (L/case/'train_C.exit').read_text()=='0'
 tr=js(L/case/'TRAIN_C.json');assert tr['epochs']==400 and tr['steps']==(76100 if case.startswith('GOAL') else 105500) and tr['gradient_autograd_calls']==48 and tr['summed_epoch_seconds']==sum(r['seconds'] for r in rows)
 for t in range(6,10):assert tr['checkpoints'][str(t)]==js(L/case/('STAGE_C_%d.json'%t))['sha256']
 grads=js(L/case/'GRADIENTS_C.json');assert {(x['stage'],x['epoch']) for x in grads}=={(t,e) for t in range(7,10) for e in [0,9,49,99]} and sum(x['extra_autograd_calls'] for x in grads)==48
 assert all(math.isfinite(g[k]['l2']) and math.isfinite(g[k]['loss']) for g in grads for k in ['bc','ifa_weighted','exec_weighted','route_weighted'])
 p=js(L/case/'TASK6_PARITY_C.json');assert p['actual_sha256']==tr['checkpoints']['6'];parities[case]=p;training[case]=tr
assert sum(x['epochs'] for x in a['counts'].values())==2000 and sum(x['steps'] for x in a['counts'].values())==468700
assert a['isolated_optimizer_calls']==95 and len(a['teachers'])==15 and a['formal_episodes']==130
assert all(x['historical_ALIGN_equal'] for x in parities.values())
assert cap['status']=='PASS_NOT_FINAL'
for x in cap['capacity'].values():assert x['free']-x['remaining']-x['reserve']==x['headroom']>=0
assert all(cap[k]==0 for k in ['payload_reads','server_writes','forwards','optimizer_updates','environments'])
(K/'LOCAL_AUDIT.json').write_text(json.dumps(dict(status='PASS_NOT_FINAL',evidence_sha256=sha(K/'EVIDENCE.json'),capacity=cap['capacity'],no_payload_model_environment_or_write=True),indent=2),encoding='utf8')
totals={k:sum(r['totals'][k] for rows in rows_by.values() for r in rows) for k in ['valid','invalid','old_eligible']}
audit=dict(status='PASS_PROGRESS_NOT_FINAL',observation=a['observation'],complete_epochs_prefix_preserved=prefixes,completed_training=training,recorded_Task6_parities=parities,own_isolated_optimizer_updates=95,formal_extra_autograd=240,admission_autograd=60,teacher_count=15,population_totals=totals,final_CPU='PENDING',source_sha256={str(p.relative_to(R)):sha(p) for p in [D/'EVIDENCE.json',D/'LOCAL_AUDIT.json',K/'EVIDENCE.json']})
(D/'TRAINING_COMPLETION_METADATA_AUDIT.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
ev=a['evaluations']['GOAL_42'];assert ev['vectors']==12 and ev['cells']==6 and ev['record_event_advance']==10
ea=js(L/'GOAL_42/EVAL_ADMISSION.json');assert ea['status']=='PASS' and ea['frontend_calls']==ea['policy_parameter_calls']==16 and ea['environments']==ea['updates']==0
times=ev['vector_seconds'];mean=sum(times)/len(times);elapsed=a['observation']-ev['ongoing'][0]['time']
# Use complete event batches for elapsed-time extrapolation. A ten-record advance
# is preserved; do not fabricate its not-yet-downloaded completion event.
remaining_mean=(88*mean-elapsed)/60;remaining_slowest=(88*max(times)-elapsed)/60
eval_audit=dict(status='PASS_PROGRESS_NOT_FINAL',evaluations=a['evaluations'],initial_actual_file_bytes='PENDING_FINAL',own_eval_admission=ea,record_event_advance_explanation='130 outcome records; 12 complete event batches plus one start. Sequential metadata download crossed a batch completion; accepted ten-record advance preserved.',mean_batch_seconds=mean,current_batch_elapsed_seconds=elapsed,remaining_mean_minutes=remaining_mean,remaining_observed_slowest_extrapolation_minutes=remaining_slowest,extrapolation_limit='Early GOAL42 only; OBJECT throughput not yet measured; slowest observed is not an upper bound',source_sha256=sha(D/'LOCAL_AUDIT.json'))
(D/'EVALUATION_PROGRESS_AUDIT.json').write_text(json.dumps(eval_audit,indent=2),encoding='utf8')
obs=datetime.datetime.fromtimestamp(a['observation'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
pure_minutes={k:v['summed_epoch_seconds']/60 for k,v in training.items()}
latest=f'''# 最新2026-10-27 20:18 CST即时检查覆盖

用户请求检查进度。实际观察{obs}，{D.name}/LOCAL_AUDIT、TRAINING_COMPLETION_METADATA_AUDIT及EVALUATION_PROGRESS_AUDIT实际PASS_PROGRESS_NOT_FINAL。五组GOAL42/43及OBJECT41/42/43各400完整epoch、共2000epoch468700主更新全部完成，五train_C.exit均0，20阶段CP记录SHA绑定。上轮四组全400轮及OBJECT43前106轮逐字段精确保留，无重复。95自身隔离GPU更新全部准入PASS，240正式额外autograd及60准入autograd另列；15教师身份/源CP/tau/w绑定通过，人口账目{totals}逐epoch复算。五组Task6服务器tensor同本seed历史ALIGN相等记录绑定，完整20CP及恢复态仍待最终独立CPU验收。

纯epoch分钟{pure_minutes}，合计{sum(pure_minutes.values()):.6f}min；教师/准入/保存/评估另计，不称等算力。68冻结项及父源不变，controller2559/evaluate3732健康，phase GOAL42 evaluate，无重启改参。

正式130/1000记录，12个完整vector事件/6关闭cell，当前事件C_s9_t6_g0。OUTCOMES比完整事件前进10条，为顺序下载跨越批次完成的快照差异，保留原记录，不伪补complete事件。全部已保存episode唯一attempt、终止/CP/初态记录及独立重生成noiseSHA、分片和事件顺序核验通过。真实初态文件字节和所有CP加载待最终，不统计部分性能。GOAL42自身评估长度1–8缓存准入PASS，16视觉及16策略参数调用、0更新/环境另列；后续run须各自准入。

12已关闭批均{mean:.6f}秒，含提前终止批；快照当前事件已{elapsed:.6f}秒。剩余按事件批均值外推{remaining_mean:.6f}min，按已观测最慢批外推{remaining_slowest:.6f}min，后者不是保证上界且目前仅早期GOAL样本，OBJECT吞吐尚未测。加载/准备另计，暂参考21:40–22:20完成全段评估，最终独立验收另计。同一监控下一120分钟在全段评估结束附近检查，不按内部run短轮询。

{K.name}/LOCAL_AUDIT实际PASS_NOT_FINAL：system普通留1GiB后余{cap['capacity']['/root']['headroom']}B；data唯一inode抵扣余新增{cap['capacity']['/data']['remaining']}B及2GiB后余{cap['capacity']['/data']['headroom']}B。仅stat，0payload/模型/优化器/环境/服务器写入，无删除付费。此前外置FP32审计修复和原错误保留，本轮无新错误。整体PARTIAL_EVALUATING；所有自然exit0且无自身进程后才final_readonly→audit_final_local→describe_final及机制分析，最终wrapper仍未实际验收。

本段覆盖旧OBJECT43训练进度及85分钟间隔，其他冻结科学协议和恢复发布限制不变。
'''
(R/'MONITOR_HANDOFF.md').write_text(latest+'\n'+(R/'MONITOR_HANDOFF.md').read_text(encoding='utf8'),encoding='utf8')
(R/'REPORT.md').write_text((R/'REPORT.md').read_text(encoding='utf8')+'\n\n'+latest,encoding='utf8')
status=js(R/'RUN_STATUS.json');status.update(status='PARTIAL_EVALUATING',observation=obs,latest_evidence=D.name,counts=a['counts'],formal_episodes=a['formal_episodes'],own_admissions=a['admissions'],next_check_minutes=120,eta=dict(active_phase='all_five_evaluations',measured_batches=12,mean_seconds=mean,remaining_mean_minutes=remaining_mean,remaining_slowest_observed_extrapolation_minutes=remaining_slowest,scope_limit='early GOAL42 only; OBJECT unmeasured',loading_and_final_audit_extra=True))
(R/'RUN_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf8');active.update(status='PARTIAL_EVALUATING',observed=obs,latest_evidence=D.name);(R/'ACTIVE_ROOT.json').write_text(json.dumps(active,indent=2),encoding='utf8')
settings=tomllib.loads(Path('C:/Users/Lenovo/.codex/automations/c-goal42-43-object41-43/automation.toml').read_text(encoding='utf8'))
standing=settings['prompt'][settings['prompt'].index('继续用户授权的C完整GOAL'):]
update=dict(id=settings['id'],mode='update',kind=settings['kind'],name=settings['name'],prompt=latest+'\n以下历史首次进度和间隔由上述最新段落覆盖。\n'+standing,status=settings['status'],rrule='FREQ=MINUTELY;INTERVAL=120',targetThreadId=settings['target_thread_id'],destination='thread')
(R/'MONITOR_UPDATE.json').write_text(json.dumps(update,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(dict(observation=obs,epoch_minutes=pure_minutes,total_epoch_minutes=sum(pure_minutes.values()),population_totals=totals,remaining_mean_minutes=remaining_mean,remaining_slowest_minutes=remaining_slowest,next_minutes=120),indent=2))
