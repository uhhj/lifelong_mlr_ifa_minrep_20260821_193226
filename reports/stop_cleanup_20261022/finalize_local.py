import json,datetime,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
P=R/'plan_cleanup_20261022_134624/EVIDENCE.json'
E=R/'execute_cleanup_20261022_135047/EVIDENCE.json'
p=json.loads(P.read_text());e=json.loads(E.read_text())
assert e['status']=='CLEANUP_COMPLETE' and e['gpu_empty'] and e['experiment_processes_absent']
assert {x['path']:x['sha256'] for x in p['delete_files']}=={x['path']:x['sha256'] for x in e['deleted']}
assert len(e['deleted'])==37 and e['skipped_preserved']==len(p['skipped'])==6
assert sum(e['released'].values())>11_400_000_000
stamp=datetime.datetime.now().astimezone().isoformat()
audit=dict(status='PASS',observed_at=stamp,planned_deleted_identity_match=True,deleted_count=37,retained_contract_references=6,evidence_unchanged=e['evidence_unchanged'],released_bytes=e['released'],source_sha256={x.name+'@'+x.parent.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in (P,E)})
(R/'LOCAL_CLEANUP_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')
status=dict(status='USER_STOPPED_CLEANUP_COMPLETE',observed_at=stamp,training_stopped=True,dispatch_stopped=True,automation_deleted='caet-v2-object41',TAC_AUX8=dict(completed_epochs=182,main_steps=34314,formal_records=0),RSET_EFTR='USER_STOPPED_BEFORE_START',deleted_files=37,released_bytes=e['released'],cleanup_audit='LOCAL_CLEANUP_AUDIT.json',experiments_completed=False)
(R/'RUN_STATUS.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf8')
text=f'''# 用户停止训练与服务器清理

观察时间：{stamp}。状态 USER_STOPPED_CLEANUP_COMPLETE；这是停止与清理完成，不是实验验收完成。

用户要求“停止训练，并清理服务器中废用的数据”。新 TAC_AUX8 调度器及训练子树已停止，无残留实验进程，GPU 空闲；无强制 SIGKILL。caet-v2-object41 自动监控已删除，不得自动恢复任何队列。

TAC_AUX8 留在 182 个完整 epoch、34,314 主更新、0 正式评估；旧七步 TAC 为 297 个完整 epoch、68,373 主更新、0 正式评估。RSET/EFTR 未启动并随队列停止。COVR、LIBERO50 预训练的用户取消保持。中断实验不得记为完成、NO_GAIN 或工程故障。

清理仅涉及 37 个通过预检的 smoke/preflight 序列化测试副本。清理前逐文件校验路径、类型、单链接、大小、mtime、SHA256、无人占用，并检查 209 份合同；6 个被合同引用的候选不删。删除逐项记账，清理后 156 份相关证据 SHA 不变，6 个保留候选 SHA 不变。本地 LOCAL_CLEANUP_AUDIT 独立核对计划与实际删除身份一致。

实测释放空间：数据盘 {e['released']['/data']/1e9:.6f} GB，系统盘 {e['released']['/']/1e9:.6f} GB，合计 {sum(e['released'].values())/1e9:.6f} GB（十进制）。清理后数据盘普通空闲 {e['after']['/data']['ordinary_free']/1e9:.3f} GB，root 全部空闲 {e['after']['/data']['root_free']/1e9:.3f} GB；系统盘普通空闲 {e['after']['/']['ordinary_free']/1e9:.3f} GB。

正式主模型、辅助头检查点、真实训练恢复点、原始数据、官方输入权重、特征、buffer、基线及实验报告均保留。没有清空整个取消实验目录。删除预检副本后，原测试二进制不可再直接检查；原 SHA、预检通过记录与本次删除清单完整留存，未改写历史科学合同。

证据：stop_inventory_20261022_134445/EVIDENCE.json；plan_cleanup_20261022_134624/EVIDENCE.json；execute_cleanup_20261022_135047/EVIDENCE.json；LOCAL_CLEANUP_AUDIT.json。服务器清单及逐项日志在 /data/mlr_ifa/cleanup_20261022/。仅发布代码和轻量元数据。
'''
(R/'REPORT.md').write_text(text,encoding='utf8')
notice=f'''\n\n## {stamp} 用户停止所有训练（覆盖此前继续执行指令）

用户要求停止训练并清理废用数据。所有当前训练与调度已停止，GPU 空闲；caet-v2-object41 自动化已删除。不得恢复 TAC_AUX8、旧 TAC、RSET 或 EFTR 队列。TAC_AUX8 182 完整 epoch/34314 主更新/0 正式；旧 TAC 297 完整 epoch/68373 主更新/0 正式；RSET/EFTR 未启动。状态 USER_STOPPED，不是实验完成或工程失败。

已清理 37 个通过预检的临时序列化副本，实测释放 11.460 GB，6 个合同引用候选保留。正式检查点、实际训练恢复点、数据、特征、buffer 与审计记录保留。清单和独立核对见 ../20261022_stop_cleanup/REPORT.md、LOCAL_CLEANUP_AUDIT.json。历史进度和协议保留但不再是继续运行授权。
'''
for directory in ('20261022_tac_aux8_object_seed41','20261022_caet_modules_object_seed41'):
 root=R.parent/directory
 for name in ('MONITOR_HANDOFF.md','REPORT.md'):
  with (root/name).open('a',encoding='utf8') as f:f.write(notice)
 old=json.loads((root/'RUN_STATUS.json').read_text(encoding='utf8'))
 (R/(directory+'_STATUS_BEFORE_STOP.json')).write_text(json.dumps(old,indent=2)+'\n',encoding='utf8')
 old.update(status='USER_STOPPED',observed_at=stamp,training_active=False,dispatch_active=False,RSET_EFTR='USER_STOPPED_BEFORE_START',automation_status='DELETED',whole_run_eta=None,stop_evidence='../20261022_stop_cleanup/stop_inventory_20261022_134445/EVIDENCE.json')
 old['completed_epochs']=182 if 'aux8' in directory else 297
 old['main_steps']=34314 if 'aux8' in directory else 68373
 old['formal_records']=0
 for key in ('stage_remaining_training_seconds','training_pid','controller_pid','queue_controller_pid','auxiliary_head_updates','auxiliary_head_exposures','formal_updates','stage','current'):
  old.pop(key,None)
 (root/'RUN_STATUS.json').write_text(json.dumps(old,indent=2)+'\n',encoding='utf8')
print(json.dumps(audit,indent=2))
