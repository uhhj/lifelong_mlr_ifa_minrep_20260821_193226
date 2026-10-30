"""Record actual accepted six-case evidence without running scientific work."""
from pathlib import Path
import json, hashlib, sys
R=Path(__file__).resolve().parent;D=Path(sys.argv[1])
js=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
a=js(R/'ACCEPTANCE.json');e=js(D/'EVIDENCE.json');f=js(D/'LOCAL_FINAL_AUDIT.json');c=js(D/'PRIMARY_S_COMPARISON.json');p=js(D/'LOCAL_AUDIT.json')
assert a['status']=='ACCEPTED_SIX_C_EQUAL_OPTIMIZER_BUDGET'
assert a['audit_sha256']==sha(D/'LOCAL_FINAL_AUDIT.json') and a['comparison_sha256']==sha(D/'PRIMARY_S_COMPARISON.json')
assert a['report_sha256']==sha(D/'FINAL_REPORT.md')==sha(R/'FINAL_REPORT.md')
assert f['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE' and not e['processes']
assert e['light'][a['completion_source']]['text']=='0' and 'controller.exit' not in e['light']
g=js(R/'GOAL_ACCEPTANCE.json');gd=Path(g['final_directory']);ge=js(gd/'EVIDENCE.json')
assert sha(gd/'LOCAL_FINAL_AUDIT.json')==g['audit_sha256']
assert sha(gd/'PRIMARY_S_COMPARISON.json')==g['comparison_sha256']
assert sha(R/'GOAL_FINAL_REPORT.md')==g['report_sha256']
checked=0
for k,v in ge['light'].items():
 if k.startswith(('GOAL_41/','GOAL_42/','GOAL_43/')):
  assert e['light'][k]==v;checked+=1
write(D/'GOAL_ACCEPTANCE_PRESERVATION.json',dict(status='PASS_ACCEPTED_GOAL_EVIDENCE_UNCHANGED',accepted_evidence_sha256=sha(gd/'EVIDENCE.json'),current_evidence_sha256=sha(D/'EVIDENCE.json'),light_files_checked=checked))
s=js(R/'RUN_STATUS.json')
s.update(status=a['status'],observation=a['observation'],latest_evidence=D.name,final_CPU_acceptance=f['status'],source_evidence_sha256=sha(D/'EVIDENCE.json'),acceptance=a,counts=p['counts'],final_metrics=c['suites'],next_check_minutes=None,publication='PENDING_FINAL_PUBLISH',monitor='DELETE_AFTER_VERIFIED_PUBLICATION')
s['suite_acceptance']['GOAL']=dict(status='ACCEPTED',original_receipt='GOAL_ACCEPTANCE.json',original_receipt_sha256=sha(R/'GOAL_ACCEPTANCE.json'),final_reconfirmation=a['audit_sha256'])
s['suite_acceptance']['OBJECT']=dict(status='ACCEPTED',receipt='ACCEPTANCE.json',final_audit=a['audit_sha256'])
write(R/'RUN_STATUS.json',s)
active=js(R/'ACTIVE_ROOT.json');active.update(status=a['status'],observed=a['observation'],latest_evidence=D.name,completion_source=a['completion_source'],controller_exit=0,processes=[],acceptance_sha256=sha(R/'ACCEPTANCE.json'));write(R/'ACTIVE_ROOT.json',active)
text=f'''# 六C等优化器预算已完成并接受

实际CPU验收观察：{a['observation']}。最终证据：{D.name}。

六C各700轮，共4200轮、1012500生产optimizer调用、546自身隔离调用、1200正式回合均实际完成。真实工程controller.exit0且无自身进程；原根exit未伪造。CPU24CP、6完整恢复、18teacher缓存及1200初态/noise通过，本地复算与已有S逐阶段等预算及1200配对通过。GOAL原独立接受证据SHA精确保留。

最终四任务成功率三seed均值：GOAL新C80%、S85%；OBJECT新C77.5%、S85%。旧C历史分别80%和82.5%，但硬件及schedule差异不允许把新旧C差值严格归因于增加预算。新C18个扩展阶段训练BC/执行cost继续降低，不等于闭环提高。保留全部负单元、SF/FS及训练代理局限。

详见FINAL_REPORT.md、ACCEPTANCE.json及最终目录的LOCAL_FINAL_AUDIT、PRIMARY_S_COMPARISON、INTERPRETATION_AUDIT。S未重训重评，旧机制永久停止；没有额外训练/补采/重评/扩seed。等optimizer不等FLOPs、墙钟、交互或可用信息。最终发布核验后删除c-goal-object-seed；不恢复旧监控或启动任何科学入口。
'''
(R/'REPORT.md').write_text(text,encoding='utf8')
h=R/'MONITOR_HANDOFF.md';old=h.read_text(encoding='utf8');marker='## 2026-10-31 最终六C已实际接受'
if not old.startswith(marker):h.write_text(marker+'\n\n'+text+'\n以下为历史记录，仅供追溯；全部旧未完成状态与继续运行指令已被上述最终接受状态覆盖。\n\n'+old,encoding='utf8')
print(json.dumps(dict(status=a['status'],GOAL_evidence_preserved=checked,source_sha256=sha(D/'EVIDENCE.json')),ensure_ascii=False))
