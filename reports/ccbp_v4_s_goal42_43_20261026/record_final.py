"""Bind actual acceptance and reports; no remote calls or scientific changes."""
import json,sys,hashlib,datetime
from pathlib import Path
R=Path(__file__).resolve().parent; D=Path(sys.argv[1]); C=Path(sys.argv[2])
js=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=js(D/'LOCAL_FINAL_AUDIT.json');v=js(D/'EVIDENCE.json');p=js(D/'LOCAL_AUDIT.json');c=js(C/'EVIDENCE.json')
assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE' and a['formal_episodes']==400 and not v['processes'] and v['light']['controller.exit']['text']=='0'
assert c['status']=='PASS_NOT_FINAL' and all(x['headroom']>=0 for x in c['capacity'].values())
assert all(c[k]==0 for k in ['payload_reads','server_writes','forwards','optimizer_updates','environments'])
(C/'LOCAL_AUDIT.json').write_text(json.dumps(dict(status='PASS_NOT_FINAL',evidence_sha256=sha(C/'EVIDENCE.json'),capacity=c['capacity'],observation=c['observed'],no_payload_or_model_or_update_or_environment=True),indent=2)+'\n',encoding='utf8')
obs=datetime.datetime.fromtimestamp(v['observed'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
state=js(R/'RUN_STATUS.json');state.update(status='COMPLETE_ACCEPTED_FINITE_EXTENSION',scientific_status='FINITE_RESULTS_ACCEPTED_NO_CAUSAL_OR_SIGNIFICANCE_CLAIM',observation=obs,latest_evidence=D.name,formal_episodes=400,next_check_minutes=None,eta=None,capacity_evidence=C.name,final_audit_sha256=sha(D/'LOCAL_FINAL_AUDIT.json'),metrics={s:x['metrics'] for s,x in a['seeds'].items()},monitor_state='PENDING_REMOVAL_AFTER_PUBLICATION')
state['evaluation']={s:dict(episodes=e['episodes'],vectors=e['completed_vectors'],cells=e['closed_cells'],record_event_advance=e['record_event_advance'],mean_seconds=e['mean_seconds'],vector_seconds_total=sum(e['vector_seconds'])) for s,e in p['evaluations'].items()}
(R/'RUN_STATUS.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf8')
active=js(R/'ACTIVE_ROOT.json');active.update(status='COMPLETE_NATURAL_EXIT_0_NO_OWN_PROCESSES',last_observation=obs,last_evidence=D.name,final_audit_sha256=sha(D/'LOCAL_FINAL_AUDIT.json'));(R/'ACTIVE_ROOT.json').write_text(json.dumps(active,indent=2)+'\n',encoding='utf8')
note=f'''# 最终验收完成：{obs}

S GOAL seed42、43已完成；controller自然exit0，训练/采集/正式评估全部结束且无自身进程。{D.name}/LOCAL_FINAL_AUDIT实际PASS_COMPLETE_LOCAL_ACCEPTANCE，8个CPU严格CP、2完整恢复、8固定成功缓存教师重放和采样、80训练及400正式回合初态字节/噪声/身份/源SHA通过。33冻结项和父源不变，800epoch152200主更新、44隔离GPU更新、144正式额外autograd和52准入autograd已分别记账。最终CPU诊断2523调用7736窗口，0更新0环境；没有补采、重训或重评。此前包装修复及所有历史进度保留。

AUC seed41/42/43为86.875/78.4375/83.020833%，平均82.777778±4.223998%；OldFinal平均80.555556%、FinalAll82.5%。三seed相对RA+ALIGN的AUC平均+1.666667pp，但seed42下降4.270833pp，不能称稳定逐seed提高。相对RA平均AUC仅+0.798611pp，新seed42/43均低于RA。完整矩阵、负单元、同seed跨阶段SF/FS、离线与成本解释见[{D.name}/REPORT.md]({D.name}/REPORT.md)。历史基线非同噪声/等交互，无C/O42/43；seed41原未过C门保留，有限扩展完成不等于证明成功筛选因果或显著。

最新容量{C.name}真实stat审计通过，无删除/付费。最终报告发布并核remoteSHA/localreceipt后删除s-goal-seed42-43监控，结束此有限授权；不得再派发训练、评估或自动扩seed。以下全部旧PARTIAL/ETA/待最终条目仅历史记录，不再作为当前状态。

'''
for name in ['REPORT.md','MONITOR_HANDOFF.md']:
 fp=R/name;fp.write_text(note+fp.read_text(encoding='utf8'),encoding='utf8')
(R/'LATEST_MONITOR_NOTE.txt').write_text(note,encoding='utf8')
names=['EVIDENCE.json','LOCAL_FINAL_AUDIT.json','LOCAL_AUDIT.json','COMPARISON.json','BASELINE_STATISTICS.json','WITHIN_SEED_STAGE_PAIRS.json','REPORT.md']
closure=dict(status='PASS_COMPLETE_ACCEPTANCE_AND_ANALYSIS',observation=obs,contract=a['contract'],artifacts={n:sha(D/n) for n in names},capacity_evidence=C.name,capacity_sha256=sha(C/'LOCAL_AUDIT.json'),publication='PENDING',monitor='PENDING_REMOVAL_AFTER_PUBLICATION',remote_scientific_changes=0,extra_training=0,extra_environment=0)
(D/'FINAL_CLOSURE.json').write_text(json.dumps(closure,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=state['status'],observation=obs,formal_episodes=400,final_audit_sha256=state['final_audit_sha256']),indent=2))
