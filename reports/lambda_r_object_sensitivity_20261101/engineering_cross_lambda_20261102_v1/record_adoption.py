"""Record observed cross-group adoption without claiming future outcomes or speedup."""
from pathlib import Path
import json,sys,subprocess,hashlib
E=Path(__file__).resolve().parent;R=E.parent;D=R/'check_readonly_20261102_205347';js=lambda p:json.loads(p.read_text(encoding='utf8'))
def put(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
a=js(D/'LOCAL_AUDIT.json');v=js(D/'EVIDENCE.json');deploy=js(R/'deploy_remote_20261102_205304/EVIDENCE.json')
assert a['status']=='PASS_PROGRESS_NOT_FINAL' and a['counts']['L005_43']['epochs']==307 and a['counts']['L020_41']['epochs']==3
light=lambda n:json.loads(v['light'][n]['text'])
live=light(E.name+'/LIVE_LAUNCH.json');committed=light(E.name+'/LIVE_COMMITTED.json')
assert live['unchanged_launcher']==11241 and live['unchanged_training_pid']==11246 and live['new_launcher']==11855 and live['training_signals']==committed['training_signals']==0
assert next(x for x in v['processes'] if x['pid']==11239)['state']=='T'
assert all(next(x for x in v['processes'] if x['pid']==pid)['state'] not in ['T','Z','X'] for pid in [11246,11860])
assert len(a['rendezvous_audit']['observed_training_endpoints'])==2
assert len({x['master_port'] for x in a['rendezvous_audit']['observed_training_endpoints']})==2
assert E.name+'/HANDOFF_COMPLETE.json' not in v['light']
startup=dict(status='PASS_ACTUAL_CROSS_LAMBDA_TRAINING_NOT_FINAL',observation=a['observation'],snapshot=D.name,snapshot_sha256=a['snapshot_sha256'],contract=deploy['contract_sha256'],controller=11852,watchdog=11854,original_coordinator_frozen=11239,unchanged_launcher=11241,unchanged_training_pid=11246,new_launcher=11855,new_training_pid=11860,actual_distinct_ports=a['rendezvous_audit']['observed_training_endpoints'],production_signals=0,own_admission_L02041=dict(optimizer_calls=4,extra_autograd=8,status='PASS_TASK6_ONLY'),root_lock_handoff='PENDING_CURRENT_L00543_NATURAL_EXIT',zero_environment_lock_validation=deploy['environment_lock_validation'],scientific_frozen_files=135,engineering_frozen_files=5,additional_optimizer_calls=0,additional_environments=0)
put(D/'CROSS_LAMBDA_STARTUP_AUDIT.json',startup);put(R/'ENGINEERING_CROSS_LAMBDA_STATUS.json',startup)
archive=E/'MONITOR_HANDOFF_BEFORE_CROSS_LAMBDA.md'
if not archive.exists():archive.write_bytes((R/'MONITOR_HANDOFF.md').read_bytes())
active=js(R/'ACTIVE_ROOT.json');active.update(engineering_state='LIVE_CROSS_LAMBDA_PENDING_NATURAL_HANDOFF',watchdog=11854,current_training_pid=11246,active_training_pids={'L005_43':11246,'L020_41':11860});put(R/'ACTIVE_ROOT.json',active)
subprocess.run([sys.executable,'-X','utf8',str(E/'record_progress.py'),str(D),'30'],check=True)
intro='''## 2026-11-02 20:53 用户授权跨λ组并行已实际启动

用户明确“可以把0.05的这一个seed和0.2的一个seed并行吗”。已唯一部署engineering_cross_lambda_20261102_v1（deploy_remote_20261102_205304），合同ee7929e8ae178125508a0feead81c0fa153f7a725f62f46bdf99fe88c9fbcead、5工程项，controller11852/watchdog11854。仅冻结原等待controller11239；L00543原launcher11241/GPU11246未收到任何信号、没有重启。新增L02041 launcher11855/GPU11860，实际端口42689与现有42941不同，都显式独立c10d localhost:0和唯一UUID，不用--standalone。

check_readonly_20261102_205347实际审计PASS，204611全部epoch/门控/诊断前缀保持，135科学及全部工程冻结项不变。L00541/42各400轮40采集；L00543现307轮(Task9 7)、71834接受127968生产91隔离41准入autograd30采集，自己的Task9准入/BRANCH_COPY_EQ/CHECK_EQ通过。L02041完整3轮471接受/生产，自己的Task6四隔离及八autograd、初态tensor/RNG/COPY_EQ/CHECK_EQ通过；后续阶段不得套PASS。总1110轮283305接受519039生产277隔离131准入autograd110采集0正式。L02042/43未启动。

LIVE_COMMITTED/LIVE_LAUNCH已实际核真，但HANDOFF_COMPLETE尚不存在。根锁仍由冻结11239持有；等原43 launcher内核Z/exit0、400轮40采集和prefix/hash通过后，新owner才结束旧等待器、取得同一根锁，写前一恢复工程L005_43_resume.exit的真实结果。严禁手动恢复11239或重复任何controller/dispatch/build/validation/已完成轮，否则可能重复未来任务。若新owner在LIVE_COMMITTED后失效，watchdog保持11239冻结并写OWNER_LOST_AFTER_LIVE_COMMIT，先核现有进程与真实成本，不盲恢复。

用户新调度覆盖旧“.05正式全部完成才允许任何.2训练”的约束：当前仅提前L02041与L00543并行；43自然完成后立即优先L005三个seed600串行正式，已启动的L02041可继续训练；正式与所有训练采集共用原collection全局flock，环境不并行。L005600正式后才启动L02042/43，最多两个训练进程，最后L020600串行正式。训练源码/wrapper/每run任务顺序、loss/阈值/块长/样本和预算均未改变。零环境模拟实际原评估wrapper锁与采集型锁竞争，独立区间无重叠，原evaluator argv/cwd不变；不是新增正式或科学环境验证。新增工程optimizer/环境均0。

实际采集互斥5/18已通过，未来13次继续核对；6个正式评估锁尚未发生，只是静态/零环境机制验证通过，不能提前称实际全部互斥完成。原采集锁事件在engineering_live_parallel_3090_20261102_v1，新正式锁事件在本cross工程，最终合并核全部时间区间不重叠。旧live自有pth和HOOK_ENABLED保留至全部接受后，核无目标collect/原SHA再清理，不提前删除其他环境文件。

只用本cross工程check/audit_progress/record_progress/final_readonly/audit_final_local。唯一最终全部成功源改为本cross工程controller.exit0且无自身进程；前一恢复controller.exit必须保持不存在，原live controller.exit1/43原train_S.exit1和FAILURE原SHA永久保留。43真实训练完成仍要求前一恢复工程L005_43_resume.exit0。六CPU最终wrapper外置适配仍STATIC_ONLY：全部自然退出后真实24CP/6恢复/24teacher/1200noise/54600gate、所有成本及环境互斥验收，再compare_primary_s三档全结果、新旧任务/SF/FS/负单元和硬件/泄露边界解释发布。

工程累计35418optimizer/94autograd不变，旧140+新400=540已记录放弃候选调用及UNKNOWN未记账成本保留；此次无新GPU训练验证成本。此前完整恢复与原289轮/5079门控保留，3090迁移不等λ单因素硬件完全受控，旧0.1只读，不跑GOAL/C/Q/旧机制，不挑最佳λ或扩seed。

当前跨组并行仅刚启动，不能将历史1.71倍或原13–18小时规划直接当新提速/完成保证。下一30分钟一次性核L020首个保持阶段准入与实际并行吞吐，再按当前整run/整组正式余时安排，不固定高频，不按时间断言完成。完整事实见CROSS_LAMBDA_REPORT_20261102.md、ENGINEERING_CROSS_LAMBDA_STATUS.json与最新CROSS_LAMBDA_STARTUP_AUDIT。

运输旧机制目录.session_control worker3700，先核status/PID，不重启活跃worker、不输出凭据；本cross工程check复制旧目录queue，按result真实目录复制完整EVIDENCE，不解析截断output主体或猜目录秒数。每次审计照AGENTS审核轻量证据串行nonforce推MLR核remoteSHA/receipt；健康静默，只实质变化/完成/新失败/需行动通知。全部六run实际验收解释发布后才删除同一监控。

'''
for n in ['MONITOR_HANDOFF.md','REPORT.md']:
 p=R/n;p.write_text(intro+p.read_text(encoding='utf8'),encoding='utf8')
(R/'CROSS_LAMBDA_REPORT_20261102.md').write_text('# 用户授权跨组并行：执行与边界\n\n'+intro,encoding='utf8')
p=R/'IMPLEMENTATION_NOTES.md';p.write_text(p.read_text(encoding='utf8')+'\n\n'+intro,encoding='utf8')
p=R/'PROTOCOL.md';p.write_text(p.read_text(encoding='utf8')+'\n\n## 2026-11-02 用户调度补充（科学逐run协议不变）\n\n用户授权把当前0.05 seed43与0.2 seed41并行。只改变run之间调度：0.05完成后优先进行其600正式，已启动0.2 seed41可继续训练；全部采集与正式共用全局互斥。0.05正式结束后才接续0.2另外两个seed，最多两个训练进程；最后0.2正式。此前组间严格串行顺序被本授权覆盖。旧科学冻结文件不修改，工程合同和实际执行审计单独保留。\n',encoding='utf8')
put(R/'FINAL_WRAPPER_REVIEW.json',dict(status='STATIC_ONLY_NOT_FINAL_ACCEPTANCE',latest_engineering=E.name,embedded_cpu_cases=6,completion_source=E.name+'/controller.exit',L00543_completion='Original failed train_S.exit1 preserved; prior recovery L005_43_resume.exit0 required',cross_lambda_authorized=True,environment_mutex_actual_final_check='PENDING',counts_unchanged=True))
s=js(R/'RUN_STATUS.json');s.update(status='PARTIAL_CROSS_LAMBDA_TRAINING',next_check_minutes=30,cross_lambda_startup_audit=str(D/'CROSS_LAMBDA_STARTUP_AUDIT.json'));put(R/'RUN_STATUS.json',s)
# Preserve two resolved local observation errors, neither repeated any server run.
put(E/'LOCAL_OBSERVATION_NOTES.json',dict(status='RESOLVED_LOCAL_ONLY',events=['Truncated queue output JSON parse failed; actual returned EVIDENCE.json read instead','Audit invocation used request second rather than returned snapshot second; rerun local audit against actual returned directory succeeded'],scientific_calls_added=0,remote_deployment_repeated=False))
print(json.dumps(dict(status=s['status'],controller=11852,epochs=a['epochs'],current_L00543=307,new_L02041=3,distinct_ports=True,next_check_minutes=30)))
