"""Read-only metadata reconciliation after both GOAL runs finish."""
from pathlib import Path
import json,hashlib,datetime,math,tomllib
R=Path(__file__).resolve().parent;D=R/'check_readonly_20261027_154716';L=D/'light';P=R/'check_readonly_20261027_144138/light';K=R/'capacity_readonly_20261027_154747'
js=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=js(D/'LOCAL_AUDIT.json');v=js(D/'EVIDENCE.json');active=js(R/'ACTIVE_ROOT.json');cap=js(K/'EVIDENCE.json')
assert a['status']=='PASS_PROGRESS_NOT_FINAL' and not a['failure'] and a['contract']==active['contract'] and v['children']==active['children'] and v['source_files']==68
assert a['phase']['case']=='OBJECT_41' and a['phase']['controller']==2559 and v['gpu'].strip()=='3030'
training={};prefixes={};parities={};rows_by={}
for case in ['GOAL_42','GOAL_43']:
 rows=js(L/case/'EPOCHS_C.json');previous=js(P/case/'EPOCHS_C.json');assert rows[:len(previous)]==previous and len(rows)==400;rows_by[case]=rows;prefixes[case]=len(previous)
 assert (L/case/'train_C.exit').read_text()=='0'
 tr=js(L/case/'TRAIN_C.json');assert tr['epochs']==400 and tr['steps']==76100 and tr['gradient_autograd_calls']==48 and tr['summed_epoch_seconds']==sum(r['seconds'] for r in rows)
 for t in range(6,10):assert tr['checkpoints'][str(t)]==js(L/case/('STAGE_C_%d.json'%t))['sha256']
 grads=js(L/case/'GRADIENTS_C.json');assert {(x['stage'],x['epoch']) for x in grads}=={(t,e) for t in range(7,10) for e in [0,9,49,99]} and sum(x['extra_autograd_calls'] for x in grads)==48
 assert all(math.isfinite(g[k]['l2']) and math.isfinite(g[k]['loss']) for g in grads for k in ['bc','ifa_weighted','exec_weighted','route_weighted'])
 p=js(L/case/'TASK6_PARITY_C.json');assert p['actual_sha256']==tr['checkpoints']['6'];parities[case]=p;training[case]=tr
new=js(L/'OBJECT_41/EPOCHS_C.json');assert len(new)==70 and new[-1]['steps']==10990 and a['isolated_optimizer_calls']==42 and a['formal_episodes']==0
assert cap['status']=='PASS_NOT_FINAL'
for x in cap['capacity'].values():assert x['free']-x['remaining']-x['reserve']==x['headroom']>=0
assert all(cap[k]==0 for k in ['payload_reads','server_writes','forwards','optimizer_updates','environments'])
(K/'LOCAL_AUDIT.json').write_text(json.dumps(dict(status='PASS_NOT_FINAL',evidence_sha256=sha(K/'EVIDENCE.json'),capacity=cap['capacity'],no_payload_model_environment_or_write=True),indent=2),encoding='utf8')
audit=dict(status='PASS_PROGRESS_NOT_FINAL',observation=a['observation'],complete_epochs_prefix_preserved=prefixes,completed_GOAL=training,recorded_Task6_parities=parities,OBJECT41_epochs=70,OBJECT41_steps=10990,own_isolated_optimizer_updates=42,teacher_count=len(a['teachers']),formal_episodes=0,final_CPU='PENDING',source_sha256={str(p.relative_to(R)):sha(p) for p in [D/'EVIDENCE.json',D/'LOCAL_AUDIT.json',K/'EVIDENCE.json']})
(D/'GOAL_TO_OBJECT_TRANSITION_AUDIT.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
obs=datetime.datetime.fromtimestamp(a['observation'],datetime.timezone(datetime.timedelta(hours=8))).isoformat();rate=a['rates_last10']['OBJECT_41']['6'];remaining6=30*rate/60
obj=js(R/'OBJECT_41/PARENT_BINDING.json')['stages'];goal=js(R/'GOAL_43/PARENT_BINDING.json')['stages'];borrowed={str(t):sum(x['seconds'] for x in rows_by['GOAL_43'] if x['stage']==t)/60*obj[str(t)]['batches']/goal[str(t)]['batches'] for t in range(7,10)};baseline=remaining6+sum(borrowed.values())
latest=f'''# 最新2026-10-27 15:50 CST交接

实际观察{obs}，check_readonly_20261027_154716/LOCAL_AUDIT及GOAL_TO_OBJECT_TRANSITION_AUDIT实际PASS_PROGRESS_NOT_FINAL。68冻结项及父源不变，controller2559/GPU3030健康；GOAL42/43各400完整epoch76100步、train_C.exit0、八CP记录绑定。上轮GOAL42全400轮及GOAL43前101轮逐字段精确保留，无重复。纯epoch分钟分别{training['GOAL_42']['summed_epoch_seconds']/60:.6f}/{training['GOAL_43']['summed_epoch_seconds']/60:.6f}，各48额外autograd另列，准备/教师/保存不含在该时间内。最终独立CPU仍待全队列结束。

已自然进入OBJECT41 Task6，70完整epoch10990步，自己的4隔离GPU更新/完整下一步恢复/原ALIGN更新精确及长度1–8准入PASS。OBJECT41旧阶段自己的准入待运行，不能套GOAL；OBJECT42/43未开始。累计870epoch163190主步、42自身隔离更新，六份GOAL教师身份/tau/w/前阶段CP绑定通过，正式0/1000。两GOAL历史Task6 tensor相等记录保留，OBJECT41 Task6未完成，不提前声称其最终tensor相等。无重启改参。

OBJECT41 Task6最近10轮均{rate:.6f}秒，余30轮约{remaining6:.6f}min。OBJECT辅助阶段吞吐尚未实测；借刚完成同卡GOAL43各阶段每batch耗时、按OBJECT Task7/8/9的227/309/362batch外推共{sum(borrowed.values()):.6f}min，剩纯计算参考{baseline:.6f}min。不同旧人口/辅助比例可能影响速度，该估计不作保证上界；缓存/准入/保存另加约10min规划余量。下一90分钟检查OBJECT41整组结束附近，暂参考17:05–17:20 CST；不是OBJECT42/43或全部1000正式完成预测，不内部Task短轮询。

capacity_readonly_20261027_154747/LOCAL_AUDIT实际PASS_NOT_FINAL：system普通留1GiB后余{cap['capacity']['/root']['headroom']}B；data唯一inode抵扣余新增{cap['capacity']['/data']['remaining']}B并留2GiB后余{cap['capacity']['/data']['headroom']}B。仅stat，0payload/服务器写入/模型/优化器/环境，无删除付费。整体PARTIAL；全队列自然退出后才真实最终验收，外置最终wrappers仍未实际运行。

本段覆盖旧GOAL43进度和60分钟间隔，其余冻结协议/恢复/发布权限不变。
'''
(R/'MONITOR_HANDOFF.md').write_text(latest+'\n'+(R/'MONITOR_HANDOFF.md').read_text(encoding='utf8'),encoding='utf8')
(R/'REPORT.md').write_text((R/'REPORT.md').read_text(encoding='utf8')+'\n\n'+latest,encoding='utf8')
status=js(R/'RUN_STATUS.json');status.update(observation=obs,latest_evidence=D.name,counts=a['counts'],formal_episodes=0,own_admissions=a['admissions'],next_check_minutes=90,eta=dict(active_case='OBJECT_41',task6_remaining_measured_minutes=remaining6,aux_reference='GOAL43 same3090 per-stage seconds/batch scaled by OBJECT batches',aux_reference_minutes=borrowed,remaining_pure_reference_minutes=baseline,extra_preparation_minutes_planning_only=10,own_aux_speed_measured=False,whole_queue_eta='not measured'))
(R/'RUN_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf8');active.update(observed=obs,latest_evidence=D.name);(R/'ACTIVE_ROOT.json').write_text(json.dumps(active,indent=2),encoding='utf8')
settings=tomllib.loads(Path('C:/Users/Lenovo/.codex/automations/c-goal42-43-object41-43/automation.toml').read_text(encoding='utf8'))
update=dict(id=settings['id'],mode='update',kind=settings['kind'],name=settings['name'],prompt=latest+'\n'+settings['prompt'],status=settings['status'],rrule='FREQ=MINUTELY;INTERVAL=90',targetThreadId=settings['target_thread_id'],destination='thread')
(R/'MONITOR_UPDATE.json').write_text(json.dumps(update,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(dict(observation=obs,total_epochs=870,total_steps=163190,GOAL_epoch_minutes={k:x['summed_epoch_seconds']/60 for k,x in training.items()},OBJECT41_pure_reference_minutes=baseline,next_minutes=90),indent=2))
