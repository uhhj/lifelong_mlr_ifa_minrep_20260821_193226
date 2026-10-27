"""Reconcile whole OBJECT42 completion and OBJECT43 progress, metadata only."""
from pathlib import Path
import json,hashlib,datetime,math,tomllib
R=Path(__file__).resolve().parent
D=R/'check_readonly_20261027_184804';L=D/'light';P=R/'check_readonly_20261027_172427/light';K=R/'capacity_readonly_20261027_184827'
js=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=js(D/'LOCAL_AUDIT.json');v=js(D/'EVIDENCE.json');active=js(R/'ACTIVE_ROOT.json');cap=js(K/'EVIDENCE.json')
assert a['status']=='PASS_PROGRESS_NOT_FINAL' and not a['failure'] and a['contract']==active['contract'] and v['children']==active['children'] and v['source_files']==68
assert a['phase']['case']=='OBJECT_43' and a['phase']['controller']==2559 and v['gpu'].strip()=='3504'
training={};prefixes={};parities={};rows_by={}
for case in ['GOAL_42','GOAL_43','OBJECT_41','OBJECT_42']:
 rows=js(L/case/'EPOCHS_C.json');previous=js(P/case/'EPOCHS_C.json');assert rows[:len(previous)]==previous and len(rows)==400
 rows_by[case]=rows;prefixes[case]=len(previous)
 assert (L/case/'train_C.exit').read_text()=='0'
 tr=js(L/case/'TRAIN_C.json');assert tr['epochs']==400 and tr['steps']==(76100 if case.startswith('GOAL') else 105500) and tr['gradient_autograd_calls']==48 and tr['summed_epoch_seconds']==sum(r['seconds'] for r in rows)
 for t in range(6,10):assert tr['checkpoints'][str(t)]==js(L/case/('STAGE_C_%d.json'%t))['sha256']
 grads=js(L/case/'GRADIENTS_C.json');assert {(x['stage'],x['epoch']) for x in grads}=={(t,e) for t in range(7,10) for e in [0,9,49,99]} and sum(x['extra_autograd_calls'] for x in grads)==48
 assert all(math.isfinite(g[k]['l2']) and math.isfinite(g[k]['loss']) for g in grads for k in ['bc','ifa_weighted','exec_weighted','route_weighted'])
 p=js(L/case/'TASK6_PARITY_C.json');assert p['actual_sha256']==tr['checkpoints']['6'];parities[case]=p;training[case]=tr
p=js(L/'OBJECT_43/TASK6_PARITY_C.json');assert p['actual_sha256']==js(L/'OBJECT_43/STAGE_C_6.json')['sha256'];parities['OBJECT_43']=p
assert a['counts']['OBJECT_43']['epochs']==106 and a['counts']['OBJECT_43']['steps']==17062 and a['isolated_optimizer_calls']==85 and a['formal_episodes']==0
assert cap['status']=='PASS_NOT_FINAL'
for x in cap['capacity'].values():assert x['free']-x['remaining']-x['reserve']==x['headroom']>=0
assert all(cap[k]==0 for k in ['payload_reads','server_writes','forwards','optimizer_updates','environments'])
(K/'LOCAL_AUDIT.json').write_text(json.dumps(dict(status='PASS_NOT_FINAL',evidence_sha256=sha(K/'EVIDENCE.json'),capacity=cap['capacity'],no_payload_model_environment_or_write=True),indent=2),encoding='utf8')
audit=dict(status='PASS_PROGRESS_NOT_FINAL',observation=a['observation'],complete_epochs_prefix_preserved=prefixes,completed_training=training,recorded_Task6_parities=parities,own_isolated_optimizer_updates=85,teacher_count=len(a['teachers']),formal_episodes=0,final_CPU='PENDING',source_sha256={str(p.relative_to(R)):sha(p) for p in [D/'EVIDENCE.json',D/'LOCAL_AUDIT.json',K/'EVIDENCE.json']})
(D/'OBJECT42_TO_43_TRANSITION_AUDIT.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
obs=datetime.datetime.fromtimestamp(a['observation'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
rate=a['rates_last10']['OBJECT_43']['7'];remaining7=94*rate/60
borrowed={str(t):sum(x['seconds'] for x in rows_by['OBJECT_42'] if x['stage']==t)/60 for t in [8,9]};baseline=remaining7+sum(borrowed.values())
total_epochs=sum(x['epochs'] for x in a['counts'].values());total_steps=sum(x['steps'] for x in a['counts'].values())
latest=f'''# 最新2026-10-27 18:50 CST交接

实际观察{obs}，{D.name}/LOCAL_AUDIT及OBJECT42_TO_43_TRANSITION_AUDIT实际PASS_PROGRESS_NOT_FINAL。68冻结项及父源不变，controller2559/GPU3504健康，已自然进入最后一组OBJECT43 Task7。GOAL42/43及OBJECT41/42各完整400epoch，四run train_C.exit均0，十六CP记录SHA绑定；上轮已完成三组全400轮及OBJECT42前125轮逐字段精确保留，无重复训练。OBJECT42纯epoch计算{training['OBJECT_42']['summed_epoch_seconds']/60:.6f}min，48额外autograd另列；保存/教师/准入另计，最终独立CPU仍待全部自然退出。

OBJECT43已106完整epoch17062步（Task6 100/Task7 6），自己的4+5隔离GPU更新、完整下一步恢复、原ALIGN/原C更新和状态化测量准入实际PASS，Task8/9须待自身准入。累计{total_epochs}epoch{total_steps}主步、85隔离优化器调用、十三份本组教师源CP/身份/tau/w绑定通过，正式0/1000。五run Task6服务器tensor与各自历史ALIGN相等记录绑定，不能替代最终CPU验收。无恢复重启改参；此前外置FP32权重审计修复和原错误保留，本轮未再失败。

OBJECT43 Task7前6轮均{rate:.6f}秒，余94轮{remaining7:.6f}min；Task8/9自身未测，借刚完成同卡OBJECT42实测共{sum(borrowed.values()):.6f}min，剩纯计算参考{baseline:.6f}min。另加10min未实测准备/缓存/保存余量，下一85分钟检查最后整组训练结束附近，暂参考20:00–20:15 CST。此后1000正式及最终验收另计，不把当前整组ETA当全队列完成时间，不为内部Task切换短轮询。

{K.name}/LOCAL_AUDIT实际PASS_NOT_FINAL：system普通留1GiB后余{cap['capacity']['/root']['headroom']}B，data唯一稳定inode抵扣余新增{cap['capacity']['/data']['remaining']}B并留2GiB后余{cap['capacity']['/data']['headroom']}B。仅stat，0payload/前向/优化器/环境/服务器写入，无删除或付费。整体PARTIAL，最终wrappers仍须全部自然exit0且无自身进程后真实运行验收。

本段覆盖旧OBJECT42进度和80分钟间隔；其余冻结科学协议和恢复限制保持。
'''
(R/'MONITOR_HANDOFF.md').write_text(latest+'\n'+(R/'MONITOR_HANDOFF.md').read_text(encoding='utf8'),encoding='utf8')
(R/'REPORT.md').write_text((R/'REPORT.md').read_text(encoding='utf8')+'\n\n'+latest,encoding='utf8')
status=js(R/'RUN_STATUS.json');status.update(observation=obs,latest_evidence=D.name,counts=a['counts'],formal_episodes=0,own_admissions=a['admissions'],next_check_minutes=85,eta=dict(active_case='OBJECT_43',task7_remaining_measured_minutes=remaining7,later_stage_reference='OBJECT42 same3090 completed stage8/9',later_stage_reference_minutes=borrowed,remaining_pure_reference_minutes=baseline,extra_preparation_minutes_planning_only=10,own_later_stage_speed_measured=False,whole_queue_eta='not measured'))
(R/'RUN_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf8');active.update(observed=obs,latest_evidence=D.name);(R/'ACTIVE_ROOT.json').write_text(json.dumps(active,indent=2),encoding='utf8')
settings=tomllib.loads(Path('C:/Users/Lenovo/.codex/automations/c-goal42-43-object41-43/automation.toml').read_text(encoding='utf8'))
# Keep the standing protocol from the first launch, replacing stale schedule history.
standing=settings['prompt'][settings['prompt'].index('继续用户授权的C完整GOAL'):]
update=dict(id=settings['id'],mode='update',kind=settings['kind'],name=settings['name'],prompt=latest+'\n以下历史首次进度和间隔由上述最新段落覆盖。\n'+standing,status=settings['status'],rrule='FREQ=MINUTELY;INTERVAL=85',targetThreadId=settings['target_thread_id'],destination='thread')
(R/'MONITOR_UPDATE.json').write_text(json.dumps(update,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(dict(observation=obs,total_epochs=total_epochs,total_steps=total_steps,remaining_pure_reference_minutes=baseline,next_minutes=85),indent=2))
