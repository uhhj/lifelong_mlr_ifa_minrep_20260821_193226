"""Record one audited evaluation snapshot and an explicit ETA-based next check."""
import json,sys,hashlib,datetime
from pathlib import Path
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);C=Path(sys.argv[2]);minutes=int(sys.argv[3]);js=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=js(D/'LOCAL_AUDIT.json');v=js(D/'EVIDENCE.json');c=js(C/'EVIDENCE.json');assert a['status']=='PASS_PROGRESS_NOT_FINAL' and c['status']=='PASS_NOT_FINAL' and all(x['headroom']>=0 for x in c['capacity'].values())
assert a['total_epochs']==800 and a['total_main_updates']==152200 and a['training_collection_episodes']==80 and a['isolated_optimizer_updates']==44
assert a['phase']['phase']=='evaluate';seed=str(a['phase']['seed']);e=a['evaluations'][seed]
ev={s:dict(episodes=x['episodes'],vectors=x['completed_vectors'],cells=x['closed_cells'],record_event_advance=x['record_event_advance'],mean_seconds=x['mean_seconds'],vector_seconds_total=sum(x['vector_seconds'])) for s,x in a['evaluations'].items()}
for s,x in ev.items():
 if x['episodes']==200:assert v['light'][f'seed{s}/evaluate.exit']['text']=='0' and x['vectors']==20 and x['cells']==10
obs=datetime.datetime.fromtimestamp(a['observation'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
(C/'LOCAL_AUDIT.json').write_text(json.dumps(dict(status='PASS_NOT_FINAL',observation=c['observed'],evidence_sha256=sha(C/'EVIDENCE.json'),capacity=c['capacity'],no_payload_or_model_or_update_or_environment=True),indent=2)+'\n',encoding='utf8')
state=js(R/'RUN_STATUS.json');state.update(status='PARTIAL_EVALUATION',observation=obs,latest_evidence=D.name,formal_episodes=a['formal_episodes'],evaluation=ev,next_check_minutes=minutes,capacity_evidence=C.name,eta=dict(remaining_active_seed_mean_minutes=e['remaining_mean_minutes'],remaining_active_seed_max_observed_minutes=e['remaining_max_observed_minutes'],scope='Remaining seed43 formal evaluation; final independent acceptance extra',not_upper_bound=True))
(R/'RUN_STATUS.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf8')
active=js(R/'ACTIVE_ROOT.json');active.update(status='ACTIVE_EVALUATION',last_observation=obs,last_evidence=D.name);(R/'ACTIVE_ROOT.json').write_text(json.dumps(active,indent=2)+'\n',encoding='utf8')
monitor=js(R/'MONITOR.json');monitor.update(next_check_interval_minutes=minutes,last_observation=obs);(R/'MONITOR.json').write_text(json.dumps(monitor,indent=2)+'\n',encoding='utf8')
note=f'''# 最新2026-10-26 16:26 CST即时检查覆盖

用户请求检查进度。实际观察{obs}，{D.name}/LOCAL_AUDIT及两seed EVALUATION_PROGRESS_AUDIT实际PASS_PROGRESS_NOT_FINAL。33冻结项及父源不变，controller2033/evaluate10040健康，phase seed43 evaluate；无重启改参。seed42完整200正式/20vector/10cell且evaluate.exit0，逐attempt/终止/策略CP/记录初态SHA/独立重生成noise/事件分片全集核验通过，正式vector共{ev['42']['vector_seconds_total']/60:.6f}min。seed43已130/200、13vector/6关闭cell，当前S_s9_t6_g1，总330/400、33vector/16关闭cell；record_event_advance均0。seed43自己的评估长度1–8缓存准入PASS，0更新/0环境，不套seed42报告。不统计部分科学收益，初态真实字节/8CP CPU/缓存重放仍待最终。

seed43前13批均{e['mean_seconds']:.6f}秒，快照当前批已{e['current_elapsed']:.6f}秒，余7批均值线性{e['remaining_mean_minutes']:.6f}min，按已观测最慢批{e['remaining_max_observed_minutes']:.6f}min，后者不是保证上界。评估结束暂参考16:32–16:36，独立最终验收另计；同一监控改为{minutes}分钟后检查全部自然退出附近，不短周期守候。

两seed各400完整epoch76100步/40训练采集仍完整，合计800/152200/80；44自身隔离更新、144正式额外autograd及52准入autograd另列。seed42/43缓存460/473，成功筛选缺额不补采。前轮训练完成账本和外置wrapper修复保留，本轮没有新错误。

{C.name}/LOCAL_AUDIT实际PASS_NOT_FINAL：system普通保留1GiB后余{c['capacity']['/root']['headroom']}B，data扣余新增及2GiB后余{c['capacity']['/data']['headroom']}B。仅stat，无payload/模型/环境/服务器写入，无删除/付费。全部自然controller.exit0无自身进程后final_readonly→audit_final_local→describe_final，外置最终路径仍仅静态须真实验收，错误留证外置修不重训重评。完整分析发布后删除监控，当前PARTIAL。本段替换旧20/400和50分钟间隔，其余科学约束不变。

'''
for name in ['REPORT.md','MONITOR_HANDOFF.md']:
 p=R/name;p.write_text(note+p.read_text(encoding='utf8'),encoding='utf8')
(R/'LATEST_MONITOR_NOTE.txt').write_text(note,encoding='utf8');print(json.dumps(dict(status=state['status'],observation=obs,evaluation=ev,eta=state['eta'],next_check_minutes=minutes),indent=2))
