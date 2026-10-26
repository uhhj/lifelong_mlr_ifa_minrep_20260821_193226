"""Record audited metadata transition; no remote writes/model/environment calls."""
import json,hashlib,datetime
from pathlib import Path
R=Path(__file__).resolve().parent;D=R/'check_readonly_20261026_145525';C=R/'capacity_readonly_20261026_145549';js=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=js(D/'LOCAL_AUDIT.json');t=js(D/'TRAINING_TRANSITION_AUDIT.json');c=js(C/'EVIDENCE.json');assert a['status']==t['status']=='PASS_PROGRESS_NOT_FINAL' and c['status']=='PASS_NOT_FINAL'
assert all(x['headroom']>=0 for x in c['capacity'].values())
(C/'LOCAL_AUDIT.json').write_text(json.dumps(dict(status='PASS_NOT_FINAL',observation=c['observed'],evidence_sha256=sha(C/'EVIDENCE.json'),capacity=c['capacity'],no_payload_or_forward_or_update_or_environment=True),indent=2)+'\n',encoding='utf8')
repair=R/'wrapper_repair_20261026_1456';(repair/'RESOLUTION.json').write_text(json.dumps(dict(status='PASS_REAL_REAUDIT_OF_EXISTING_SNAPSHOT',original_repair_sha256=sha(repair/'REPAIR.json'),progress_audit_sha256=sha(D/'LOCAL_AUDIT.json'),transition_audit_sha256=sha(D/'TRAINING_TRANSITION_AUDIT.json'),source_contract=a['contract'],server_changes=0,extra_rollouts=0,extra_optimizer_updates=0),indent=2)+'\n',encoding='utf8')
obs=datetime.datetime.fromtimestamp(a['observation'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
state=dict(status='PARTIAL_TRAINING',observation=obs,latest_evidence=D.name,controller=2033,root=js(R/'ACTIVE_ROOT.json')['root'],contract=a['contract'],seeds={s:dict(counts=x['counts']['S'],admissions=x['admissions'],memory=x['memory']) for s,x in a['seeds'].items()},main_updates=a['total_main_updates'],formal_episodes=a['formal_episodes'],training_collection_episodes=a['training_collection_episodes'],isolated_optimizer_updates=a['isolated_optimizer_updates'],metadata_skew=t['metadata_skew'],next_check_minutes=60,eta=t['eta'],capacity_evidence=C.name,scientific_status='PARTIAL')
(R/'RUN_STATUS.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf8')
active=js(R/'ACTIVE_ROOT.json');active.update(status='ACTIVE_TRAINING',last_observation=obs,last_evidence=D.name);(R/'ACTIVE_ROOT.json').write_text(json.dumps(active,indent=2)+'\n',encoding='utf8')
monitor=js(R/'MONITOR.json');monitor.update(next_check_interval_minutes=60,last_observation=obs);(R/'MONITOR.json').write_text(json.dumps(monitor,indent=2)+'\n',encoding='utf8')
note='''# 最新2026-10-26 14:59 CST进度覆盖

实际观察14:55:25 CST，check_readonly_20261026_145525的LOCAL_AUDIT与TRAINING_TRANSITION_AUDIT实际PASS_PROGRESS_NOT_FINAL。33冻结项和两个父源不变，controller2033/GPU3430正常，已自然接续seed43 Task8。seed42完整400epoch76100主步、train_S.exit0和40训练采集全部结束，四CP记录/缓存源/父完整状态不变绑定通过；缓存115/115/102/128共460，成功筛选缺额按原规则保留，不补采。纯epoch81.654754min，四采集4.323846min，保存/准入/准备另计。最终独立CPU检查与400正式仍待后续，不计算训练采集为正式收益。

seed43已251完整epoch33335步（100/100/51/0），自己的Task6/7/8共16隔离GPU调用、完整下一步恢复、关保持=原C和状态化梯度测量准入PASS；自身1–8短历史PASS。已20训练采集及128/128缓存，正式0。两seed累计651epoch109435主步、60训练采集、38自身隔离更新；两seedTask6服务器记录均与本seed历史ALIGN tensor相等，不替代最终CPU验收。元数据自然前进：EPOCHS含Task8第51轮33335步而TRAIN_STATUS仍第50轮33150步，以完整EPOCHS账本计数并保留差异，不重跑。

本轮本地审计首次遇内层memory binding变量覆盖父binding的KeyError，加上子解释器未显式UTF8导致中文路径错误输出解码失败。已保留三份外置旧脚本及原错误摘要，仅重命名局部变量/固定子进程UTF8，对同一快照真实重审PASS；wrapper_repair_20261026_1456/RESOLUTION绑定前后SHA和审计结果。冻结服务器源码/参数/进程未改，0额外环境/优化器，不把包装错误当科学失败。

seed43 Task8最近10轮13.050015秒，余49轮10.657512min；Task9自身尚未测，借刚完成seed42 Task9实际43.059337min，剩余纯计算参考53.716850min，采集/准入/保存额外。下一60min检查seed43整组训练完成附近，暂参考15:50–16:05；不是400正式和最终验收完成时间，不内部Task短轮询。

capacity_readonly_20261026_145549实际PASS_NOT_FINAL：system普通保留1GiB后余372830208B，data按唯一inode已分配抵扣剩余4GiB界并留2GiB后余45168197632B。仅stat，0payload/前向/更新/环境/服务器写入，无删除/付费或恢复重启。整体PARTIAL；完成自然exit0且无自身进程后才final_readonly→audit_final_local→describe_final。外置最终路径仍待真实验收。此段覆盖旧进度与115min间隔，其余科学协议和限制有效。

'''
for name in ['REPORT.md','MONITOR_HANDOFF.md']:
 p=R/name;p.write_text(note+p.read_text(encoding='utf8'),encoding='utf8')
(R/'LATEST_MONITOR_NOTE.txt').write_text(note,encoding='utf8');print(json.dumps(dict(status=state['status'],observation=obs,next_check_minutes=60),indent=2))
