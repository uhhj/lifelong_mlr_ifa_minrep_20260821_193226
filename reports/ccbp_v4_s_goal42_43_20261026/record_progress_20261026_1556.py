"""Bind full-training metadata and first real evaluation rate; no extra experiment calls."""
import json,hashlib,datetime
from pathlib import Path
R=Path(__file__).resolve().parent;D=R/'check_readonly_20261026_155630';C=R/'capacity_readonly_20261026_155644';js=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=js(D/'LOCAL_AUDIT.json');t=js(D/'TRAINING_TRANSITION_AUDIT.json');c=js(C/'EVIDENCE.json');e=a['evaluations']['42']
assert a['status']==t['status']=='PASS_PROGRESS_NOT_FINAL' and c['status']=='PASS_NOT_FINAL' and all(x['headroom']>=0 for x in c['capacity'].values())
assert a['total_epochs']==800 and a['total_main_updates']==152200 and a['isolated_optimizer_updates']==44 and a['training_collection_episodes']==80 and all(x['complete_training'] for x in t['seeds'].values())
ledger={};v=js(D/'EVIDENCE.json')
for s in (42,43):
 hist=json.loads(v['light'][f'seed{s}/EPOCHS_S.json']['text']);ledger[str(s)]={k:sum(x['totals'][k] for x in hist) for k in ['valid','invalid','old_eligible','anchor_slots','anchor_actual','anchor_empty']}
mean_remaining=(40-e['completed_vectors'])*e['mean_seconds']/60-e['current_elapsed']/60
max_remaining=(40-e['completed_vectors'])*max(e['vector_seconds'])/60-e['current_elapsed']/60
eta=dict(remaining_all_evaluation_mean_minutes=mean_remaining,remaining_all_evaluation_max_observed_minutes=max_remaining,observed_vectors=len(e['vector_seconds']),seed43_speed='UNMEASURED; extrapolated from current seed42',loading_and_final_audit_extra=True,not_guaranteed_upper_bound=True,next_check_minutes=50)
obs=datetime.datetime.fromtimestamp(a['observation'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
(D/'TRAINING_COMPLETION_LEDGER.json').write_text(json.dumps(dict(status='PASS_METADATA_NOT_FINAL_CPU',observation=obs,ledger=ledger,totals={k:sum(x[k] for x in ledger.values()) for k in ledger['42']},formal_extra_autograd=sum(x['gradient_measurement_autograd'] for x in t['seeds'].values()),admission_extra_autograd=52,epoch_seconds={s:x['epoch_seconds'] for s,x in t['seeds'].items()},source_sha256=sha(D/'TRAINING_TRANSITION_AUDIT.json')),indent=2)+'\n',encoding='utf8')
(C/'LOCAL_AUDIT.json').write_text(json.dumps(dict(status='PASS_NOT_FINAL',observation=c['observed'],evidence_sha256=sha(C/'EVIDENCE.json'),capacity=c['capacity'],no_payload_or_forward_or_update_or_environment=True),indent=2)+'\n',encoding='utf8')
state=dict(status='PARTIAL_EVALUATION',observation=obs,latest_evidence=D.name,controller=2033,root=js(R/'ACTIVE_ROOT.json')['root'],contract=a['contract'],seeds={s:dict(counts=x['counts']['S'],admissions=x['admissions'],memory=x['memory']) for s,x in a['seeds'].items()},main_updates=a['total_main_updates'],formal_episodes=a['formal_episodes'],training_collection_episodes=a['training_collection_episodes'],isolated_optimizer_updates=a['isolated_optimizer_updates'],metadata_skew=t['metadata_skew'],next_check_minutes=50,eta=eta,capacity_evidence=C.name,scientific_status='PARTIAL')
(R/'RUN_STATUS.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf8')
active=js(R/'ACTIVE_ROOT.json');active.update(status='ACTIVE_EVALUATION',last_observation=obs,last_evidence=D.name);(R/'ACTIVE_ROOT.json').write_text(json.dumps(active,indent=2)+'\n',encoding='utf8')
monitor=js(R/'MONITOR.json');monitor.update(next_check_interval_minutes=50,last_observation=obs);(R/'MONITOR.json').write_text(json.dumps(monitor,indent=2)+'\n',encoding='utf8')
note=f'''# 最新2026-10-26 15:59 CST进度覆盖

实际观察15:56:30 CST，check_readonly_20261026_155630的LOCAL_AUDIT、TRAINING_TRANSITION_AUDIT、TRAINING_COMPLETION_LEDGER及seed42 EVALUATION_PROGRESS_AUDIT实际PASS_PROGRESS_NOT_FINAL（账本为PASS_METADATA_NOT_FINAL_CPU）。33冻结项及父源不变；controller2033/evaluate4858健康，phase evaluate seed42，无恢复重启改参。两seed各400epoch76100步已自然train_S.exit0，合计800epoch152200主步及80训练采集全部完成。所有44自身隔离GPU更新、16短历史额外autograd/34策略调用实际PASS；正式额外144autograd、准入总52另列。完整人口账本BC合法1426200/屏蔽93400/旧合法538400已复算，8CP记录及源绑定通过，实际最终CPU仍待全队列。

seed42/43各缓存115/115/102/128=460和128/128/102/115=473，成功筛选缺额保留，不补采。全部8次采集父model/optimizer/scheduler/grads/modes/requires/allRNG/step前后精确记录绑定，逐attempt/成功筛选配额/源CP/初态记录hash/noise独立重生成通过。纯epoch42为81.654754min、43为82.836659min；采集分别4.323846/3.175360min，准备/保存/准入另计，不算正式收益。

正式20/400，2完整vector/1关闭cell，当前seed42 S_s7_t6_g0，record_event_advance0。逐episode/attempt/终止/策略CP/记录初态SHA、分片全集与事件顺序通过；seed42评估自身长度1–8缓存准入PASS，16视觉前向/16策略参数调用、0更新/环境，最大feature差9.298325e-6、参数差1.937151e-6。seed43自己的评估准入尚待运行，不能套用。初态真实文件字节与8CPU CP/缓存重放仍待最终，不计算部分性能。

首2批69.867339/68.927909秒、均69.397624秒；快照当前批已17.256140秒。剩余全队列38批（含seed43未实测20批）线性{mean_remaining:.6f}min，按已观测最慢批{max_remaining:.6f}min，不是保证上界，阶段加载额外。暂参考16:40–16:50全部评估结束，最终验收另计。下一50min检查全部正式评估自然退出附近，不内部切换短轮询。

capacity_readonly_20261026_155644实际PASS_NOT_FINAL：system普通保留1GiB后余372781056B，data抵扣余新增并留2GiB后余45167730688B。仅stat，无payload/前向/更新/环境/服务器写入，无删除/付费。上轮外置wrapper修复及原错误全部留证，本轮真实审计无新错误。全部自然controller.exit0且无自身进程后立即final_readonly→audit_final_local→describe_final；外置最终路径仍仅静态，须真实8CP/2恢复/8缓存/80训练400正式验收，错误留证外置修、不重训重评。完整分析发布后删除监控；当前PARTIAL。本段覆盖旧训练进度与60min间隔。

'''
for name in ['REPORT.md','MONITOR_HANDOFF.md']:
 p=R/name;p.write_text(note+p.read_text(encoding='utf8'),encoding='utf8')
(R/'LATEST_MONITOR_NOTE.txt').write_text(note,encoding='utf8');print(json.dumps(dict(status=state['status'],observation=obs,eta=eta,ledger=ledger),indent=2))
