"""Record actual armed boundary handoff and the independent performance trial."""
import json,sys,hashlib,datetime
from pathlib import Path
E=Path(__file__).resolve().parent;R=E.parent;D=Path(sys.argv[1]);interval=int(sys.argv[2]);js=lambda p:json.loads(p.read_text(encoding='utf8'))
a=js(D/'LOCAL_AUDIT.json');v=js(D/'EVIDENCE.json');active=js(R/'ACTIVE_ROOT.json');trial=js(E/'PARALLEL_TRIAL_AUDIT.json')
assert a['status']=='PASS_PROGRESS_NOT_FINAL'
light=v['light'];read=lambda name:json.loads(light[E.name+'/'+name]['text'])
armed=read('ARMED.json');guard=read('GUARD.json');assert armed['status']=='WAIT_L005_42_NATURAL_EXIT' and armed['training_signals']==0
assert E.name+'/HANDOFF_COMPLETE.json' not in light and E.name+'/WATCHDOG_RELEASE.json' not in light and E.name+'/FAILURE.json' not in light
processes={p['pid']:p for p in v['processes']}
assert processes[4347]['state']=='T' and 4689 in processes and processes[4694]['state'] not in ['T','Z','X']
assert active['controller'] in processes and armed['watchdog'] in processes
h=json.loads(light['L005_42/EPOCHS_S.json']['text']);assert hashlib.sha256(json.dumps(h[:guard['epoch_count']],sort_keys=True,separators=(',',':')).encode()).hexdigest()==guard['epoch_prefix_sha256']
g=b''.join(light.get('L005_42/GATES_stage%d.jsonl'%st,{}).get('text','').encode() for st in [7,8,9]);assert hashlib.sha256(g[:guard['gate_prefix_bytes']]).hexdigest()==guard['gate_prefix_sha256']
active.update(engineering_state='ARMED_WAIT_L005_42_NATURAL_EXIT',watchdog=armed['watchdog'],current_training_pid=4694)
(R/'ACTIVE_ROOT.json').write_text(json.dumps(active,indent=2)+'\n',encoding='utf8')
status=dict(status='ARMED_WAIT_L005_42_NATURAL_EXIT',observed=a['observation'],controller=active['controller'],watchdog=armed['watchdog'],old_waiting_controller=4347,old_waiting_controller_state='T',current_training_pid=4694,current_training_signals=0,production_parallel_active=False,full_validation=trial['status'],diagnostic_optimizer_calls=34549,diagnostic_extra_autograd=83,parallel_diagnostic_optimizer_calls=18584,parallel_diagnostic_environments=0,engineering_contract=active['engineering_contract'],engineering_frozen_files=active['engineering_frozen_files'],completion_source=active['completion_source'],max_training_concurrency=2,parallel_group='L020',training_collections='serialized',formal_evaluation='serial',guard_failure='Diagnostic headroom guard before parallel launch; retained 6392 serial calls; versioned continuation completed remaining 6392; original failure preserved',latest_evidence=D.name)
(R/'ENGINEERING_PARALLEL_STATUS.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf8');(D/'PARALLEL_HANDOFF_LOCAL_AUDIT.json').write_text(json.dumps(dict(status='PASS_ARMED_PREFIX_PROCESS_AND_COST_AUDIT',**{k:z for k,z in status.items() if k!='status'}),indent=2)+'\n',encoding='utf8')
timing=trial['cases']['timing'];m=timing['reports'][0];c=a['counts']['L005_42']
header=f'''## 最新 {a['observation']}：并行等价验证通过，调度已部署等待自然边界

用户要求实际尝试多seed。三次隔离试验合计18584 optimizer、0额外autograd、0环境、0生产epoch。两档λ/Task6–9、seed42/43串行对并行完整一轮的逐步标量、C/Q完整状态、探针、门控和保存加载下一步实际逐位相等。Task9去逐块指纹的计时为串行两份{m['serial_two_worker_seconds']:.2f}秒→并行{m['parallel_two_worker_seconds']:.2f}秒，合计吞吐{m['aggregate_throughput_ratio']:.4f}倍、墙钟减少{100*m['wall_reduction_fraction']:.2f}%。生产同时作为背景负载，不能外推整run/所有未来轨迹。并行计时窗口GPU均{timing['gpu_windows']['parallel']['gpu_mean']:.2f}%，显存峰值{timing['gpu_windows']['parallel']['memory_peak_MiB']:.0f}MiB，含生产+两个诊断进程，不是未来双生产利用率承诺。

完整验证的串行参照先完成6392调用；生产进入采集时触发剩余资源保护，诊断在任何并行调用前退出。失败原件保留，16:30只读资源恢复后唯一continue_parallel_v2续缺少6392调用，未重复串行参照或科学轮。累计工程34549 optimizer/83额外autograd单列，旧中断140已记候选+UNKNOWN保留。所有诊断已退出。详见PARALLEL_OPTIMIZATION_REPORT_20261102.md、ENGINEERING_PARALLEL_STATUS.json及工程PARALLEL_TRIAL_AUDIT.json。

新engineering_parallel_3090_20261102_v1合同{active['engineering_contract']}、{active['engineering_frozen_files']}冻结项，controller{active['controller']}/watchdog{armed['watchdog']}已ARMED；仅旧等待调度器4347为T，当前launcher4689/GPU4694未收信号且继续。HANDOFF_COMPLETE尚不存在，不称生产并行已启用。L005_41完整400轮40采集；L005_42现{c['epochs']}轮、{c['accepted_updates']}接受、{c['physical_optimizer_calls']}生产物理、{c['training_collection_completed_episodes']}采集。{D.name}及新audit_progress/PARALLEL_HANDOFF_LOCAL_AUDIT PASS，153532全部epoch/门控前缀及新GUARD前缀精确保留，135科学+5设备+8恢复+6检查及新工程项不变。

调度等L00542自然400轮40采集、launcher内核Z/exit0及prefix/hash通过后，结束旧等待调度器并取得同root锁。然后L00543单独训练→L005三seed600串行正式→L020三seed最多2个训练并行→L020三seed600串行正式；训练采集全局互斥。没有改变loss/阈值/批次/各seed任务顺序/RNG/预算。若一个并行训练失败，不再启动后续任务，让已启动同伴自然结束并报告，不盲重启。

只用新工程check/audit_progress/record_progress/final_readonly/audit_final_local；最终只认新工程controller.exit0且无自身进程。根及设备/恢复/检查工程旧exit保持不存在。全部6份嵌入CPU脚本也已外置适配新完成源，最终仍STATIC_ONLY，真实24CP/6恢复/24teacher/1200noise/54600gate及16次未来采集互斥验收待全部结束。未来每阶段自己的GPU准入/BRANCH_COPY_EQ/CHECK_EQ必须实际通过；3090迁移、开发集反复使用、预训练来源未闭合限制不变。禁止重复任何dispatch/controller/validation/build或完成轮。

下一{interval}分钟在当前完整L00542预计结束较早侧复查，根据新实测调整；后续并行剩余时间按实际并发完成时间估计，不能简单累加或把1.64倍套全部队列。保持同一object-s-r-run，健康静默；全部科学验收解释发布后才删除。运输worker3700先核status/PID，凭据不输出。每次按AGENTS经publish.py轻量审查串行非force推MLR核远端SHA/receipt。

'''
for name in ['MONITOR_HANDOFF.md','REPORT.md']:(R/name).write_text(header+(R/name).read_text(encoding='utf8'),encoding='utf8')
(R/'PARALLEL_OPTIMIZATION_REPORT_20261102.md').write_text('# 3090双任务并行试验与安全调度\n\n'+header+'\n原始轻量证据、诊断源码、独立复算和部署来源均见engineering_parallel_3090_20261102_v1。当前为PARTIAL；没有正式闭环结果或全实验验收。\n',encoding='utf8')
print(json.dumps(status))
