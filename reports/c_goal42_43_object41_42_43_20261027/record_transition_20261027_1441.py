"""Record audited GOAL42 completion and GOAL43 transition; no remote execution."""
from pathlib import Path
import json,hashlib,datetime,math,tomllib
R=Path(__file__).resolve().parent;D=R/'check_readonly_20261027_144138';L=D/'light';P=R/'check_readonly_20261027_133342/light';K=R/'capacity_readonly_20261027_144204'
js=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=js(D/'LOCAL_AUDIT.json');v=js(D/'EVIDENCE.json');active=js(R/'ACTIVE_ROOT.json');cap=js(K/'EVIDENCE.json')
assert a['status']=='PASS_PROGRESS_NOT_FINAL' and a['failure'] is None and a['contract']==active['contract'] and v['children']==active['children'] and v['source_files']==68
assert a['phase']['case']=='GOAL_43' and a['phase']['phase']=='train' and a['phase']['controller']==2559 and v['gpu'].strip()=='2808'
assert (L/'GOAL_42/train_C.exit').read_text()=='0'
old=js(P/'GOAL_42/EPOCHS_C.json');rows=js(L/'GOAL_42/EPOCHS_C.json');assert rows[:len(old)]==old and len(rows)==400
tr=js(L/'GOAL_42/TRAIN_C.json');assert tr['epochs']==400 and tr['steps']==76100 and tr['gradient_autograd_calls']==48
assert tr['summed_epoch_seconds']==sum(r['seconds'] for r in rows)
for t in range(6,10):assert tr['checkpoints'][str(t)]==js(L/('GOAL_42/STAGE_C_%d.json'%t))['sha256']
grads=js(L/'GOAL_42/GRADIENTS_C.json');assert {(x['stage'],x['epoch']) for x in grads}=={(t,e) for t in range(7,10) for e in [0,9,49,99]} and sum(x['extra_autograd_calls'] for x in grads)==48
assert all(math.isfinite(g[k]['l2']) and math.isfinite(g[k]['loss']) for g in grads for k in ['bc','ifa_weighted','exec_weighted','route_weighted'])
r43=js(L/'GOAL_43/EPOCHS_C.json');assert len(r43)==101 and r43[-1]['steps']==10139
assert all(x['sampler_sha256']==y['sampler_sha256'] for x,y in zip(rows,r43))
parities={}
for case in ['GOAL_42','GOAL_43']:
 p=js(L/case/'TASK6_PARITY_C.json');assert p['actual_sha256']==js(L/case/'STAGE_C_6.json')['sha256'] and p['historical_ALIGN_equal'] and p['historical_max_abs']==0
 parities[case]=p
assert a['isolated_optimizer_calls']==28 and a['formal_episodes']==0
for x in cap['capacity'].values():assert x['free']-x['remaining']-x['reserve']==x['headroom']>=0
assert cap['status']=='PASS_NOT_FINAL' and all(cap[k]==0 for k in ['payload_reads','server_writes','forwards','optimizer_updates','environments'])
(K/'LOCAL_AUDIT.json').write_text(json.dumps(dict(status='PASS_NOT_FINAL',evidence_sha256=sha(K/'EVIDENCE.json'),capacity=cap['capacity'],no_payload_model_environment_or_write=True),indent=2),encoding='utf8')
transition=dict(status='PASS_PROGRESS_NOT_FINAL',observation=a['observation'],prefix_complete_epochs_preserved=len(old),new_complete_epochs_repeated=0,GOAL42_training=tr,GOAL42_gradient_points=len(grads),GOAL43_complete_epochs=len(r43),GOAL43_steps=r43[-1]['steps'],recorded_Task6_tensor_parity=parities,own_isolated_updates=28,source_sha256={str(p.relative_to(R)):sha(p) for p in [D/'EVIDENCE.json',D/'LOCAL_AUDIT.json',K/'EVIDENCE.json',L/'GOAL_42/TRAIN_C.json']},final_CPU_acceptance='PENDING')
(D/'TRAINING_TRANSITION_AUDIT.json').write_text(json.dumps(transition,indent=2),encoding='utf8')
obs=datetime.datetime.fromtimestamp(a['observation'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
rate=a['rates_last10']['GOAL_43']['7'];step_linear=(76100-10139)/139*rate/60
task7_remaining=99*rate/60;later=sum(x['seconds'] for x in rows if x['stage'] in (8,9))/60
latest=f'''# 最新2026-10-27 14:44 CST交接

实际观察{obs}，check_readonly_20261027_144138/LOCAL_AUDIT及TRAINING_TRANSITION_AUDIT实际PASS_PROGRESS_NOT_FINAL。68冻结项及父源不变，controller2559/GPU2808健康，GOAL42完整400epoch76100主更新已自然train_C.exit0，四阶段CP记录绑定，前29完整epoch逐字段精确保留，无重训。纯epoch计算{tr['summed_epoch_seconds']/60:.6f}分钟，48正式额外autograd另列，保存/教师/准入另计；最终独立CPU仍待全队列结束。

GOAL43已101完整epoch10139步（Task6 100/Task7 1）；自己的Task6/7 4+5隔离GPU调用及完整下一步恢复、原ALIGN/原C更新、状态化梯度开关PASS。两GOAL Task6服务器tensor各自与本seed历史ALIGN精确相等记录绑定，不替代最终CPU。累计501epoch86239步、28自身隔离更新、四份教师身份/tau/w/前阶段CP绑定通过，正式0/1000；OBJECT三seed未开始，不套GOAL准入。无恢复重启改参。

GOAL43 Task7仅首1轮{rate:.6f}秒/139batch，余65961步按每batch线性{step_linear:.6f}min；按Task7余99轮{task7_remaining:.6f}min加刚完成GOAL42 Task8/9实测{later:.6f}min，剩纯计算参考{task7_remaining+later:.6f}min。自身Task8/9未测，保存/准备/教师另计，下一60分钟在GOAL43整组完成附近检查，参考15:35–15:45 CST；不是OBJECT三seed或1000评估总ETA，不内部Task短轮询。

capacity_readonly_20261027_144204/LOCAL_AUDIT实际PASS_NOT_FINAL：system普通留1GiB后余{cap['capacity']['/root']['headroom']}B，data唯一inode抵扣余新增{cap['capacity']['/data']['remaining']}B且留2GiB后余{cap['capacity']['/data']['headroom']}B。仅stat、0payload/前向/优化器/环境/服务器写入，无删除付费。全部最终独立验收仍待自然结束，整体PARTIAL。旧启动/工程记录保留。

本段覆盖旧13:33进度和65分钟间隔，其余科学合同和恢复限制不变。
'''
oldhandoff=(R/'MONITOR_HANDOFF.md').read_text(encoding='utf8');(R/'MONITOR_HANDOFF.md').write_text(latest+'\n'+oldhandoff,encoding='utf8')
(R/'REPORT.md').write_text((R/'REPORT.md').read_text(encoding='utf8')+'\n\n'+latest,encoding='utf8')
status=js(R/'RUN_STATUS.json');status.update(observation=obs,latest_evidence=D.name,counts=a['counts'],formal_episodes=0,own_admissions=a['admissions'],next_check_minutes=60,eta=dict(active_case='GOAL_43',own_task7_epochs_observed=1,own_step_linear_minutes=step_linear,remaining_task7_plus_recent_GOAL42_task8_9_minutes=task7_remaining+later,preparation_saving_extra=True,whole_queue_eta='not measured'))
(R/'RUN_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf8')
active.update(observed=obs,latest_evidence=D.name);(R/'ACTIVE_ROOT.json').write_text(json.dumps(active,indent=2),encoding='utf8')
settings=tomllib.loads(Path('C:/Users/Lenovo/.codex/automations/c-goal42-43-object41-43/automation.toml').read_text(encoding='utf8'))
newprompt=latest+'\n'+settings['prompt']
update=dict(id=settings['id'],mode='update',kind=settings['kind'],name=settings['name'],prompt=newprompt,status=settings['status'],rrule='FREQ=MINUTELY;INTERVAL=60',targetThreadId=settings['target_thread_id'],destination='thread')
(R/'MONITOR_UPDATE.json').write_text(json.dumps(update,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(dict(observation=obs,counts=a['counts'],GOAL42_epoch_minutes=tr['summed_epoch_seconds']/60,GOAL43_remaining_pure_reference_minutes=task7_remaining+later,GOAL43_step_linear_minutes=step_linear,next_minutes=60),indent=2))
