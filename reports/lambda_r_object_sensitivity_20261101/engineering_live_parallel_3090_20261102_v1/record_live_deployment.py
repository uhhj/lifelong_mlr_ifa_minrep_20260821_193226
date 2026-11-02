from pathlib import Path
import json,hashlib,ast,shutil,datetime
R=Path('execution/20261101_lambda_r_object_sensitivity');E=R/'engineering_live_parallel_3090_20261102_v1';D=R/'check_readonly_20261102_172140';G=R/'capacity_readonly_20261102_172100'
js=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=js(D/'LOCAL_AUDIT.json');v=js(D/'EVIDENCE.json');gpu=js(G/'EVIDENCE.json');active=js(R/'ACTIVE_ROOT.json');j=lambda n:json.loads(v['light'][E.name+'/'+n]['text'])
# Convert local redirected text to UTF-8 for the reviewed publication.
p=E/'latest_audit_output.txt';b=p.read_bytes();p.write_text(b.decode('utf16') if b.startswith(b'\xff\xfe') else b.decode('utf8'),encoding='utf8')
old=R.parent/'20261027_support_gate_forks/capacity_readonly_20261102_170423'
if old.exists():shutil.copytree(old,R/old.name,dirs_exist_ok=True)
assert a['status']=='PASS_PROGRESS_NOT_FINAL' and not a['failures']
assert a['counts']['L005_43']['epochs']>=1 and a['counts']['L005_43']['own_isolated_optimizer_calls']==4
assert {p['pid'] for p in v['processes'] if p['state']=='T'}=={4347,8608,8609}
assert all(any(p['pid']==pid and p['state'] not in ['T','Z','X'] for p in v['processes']) for pid in [4689,4694,8996,9000])
assert E.name+'/HANDOFF_COMPLETE.json' not in v['light']
for name in ['controller.exit','engineering_checks_3090_20261102_v1/controller.exit','engineering_parallel_3090_20261102_v1/controller.exit']:assert name not in v['light']
assert j('LIVE_LAUNCH.json')['training_signals']==0
status=dict(status='PASS_ACTUAL_LIVE_PARALLEL_PRODUCTION_PARTIAL',observed=a['observation'],snapshot=D.name,snapshot_sha256=sha(D/'EVIDENCE.json'),controller=8994,watchdog=8995,existing_seed42_launcher=4689,existing_seed42_gpu=4694,new_seed43_launcher=8996,new_seed43_gpu=9000,existing_seed42_original_process_continues=True,production_signals=0,contract=active['engineering_contract'],frozen_files=9,all_scientific_and_prior_engineering_files_unchanged=True,completed_and_current_prefix_exact=True,live_adoption=True,root_lock_handoff='WAIT_EXISTING_L00542_NATURAL_EXIT',old_waiters_frozen=[4347,8608,8609],expected_future_collection_locks=18,production_collection_locks_observed=a['parallel_collection_lock_audit']['acquisitions'],startup_validation='PASS_ZERO_UPDATE_EXACT_STARTUP_LOCK',local_control_flow='PASS_ACTUAL_MAIN_SIMULATED_ORDER_AND_FAILURES',new_startup_validation_optimizer_calls=0,new_startup_validation_environments=0,total_engineering_optimizer_calls=34549,total_engineering_extra_autograd=83,abandoned_partial_recorded_optimizer_calls=140,abandoned_partial_unrecorded_cost='UNKNOWN',GPU_window=dict(source=G.name,evidence_sha256=sha(G/'EVIDENCE.json'),samples=30,mean_utilization=gpu['mean_gpu_utilization'],min_utilization=gpu['min_gpu_utilization'],max_utilization=gpu['max_gpu_utilization'],peak_used_MiB=gpu['peak_used_MiB'],scope='Actual concurrent Task8/Task6, not an equal-work speed comparison'),next_check_minutes=150,completion_source=E.name+'/controller.exit')
(R/'ENGINEERING_LIVE_PARALLEL_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
review=js(R/'FINAL_WRAPPER_REVIEW.json');scripts=next(ast.literal_eval(n.value) for n in ast.parse((E/'final_readonly.py').read_text(encoding='utf8')).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='SCRIPTS')
assert len(scripts)==6
for s in scripts.values():assert "Q.parent/'"+E.name+"'/'controller.exit'" in s;ast.parse(s)
review.update(engineering_completion_source=E.name+'/controller.exit',engineering_contract=active['engineering_contract'],engineering_wrappers={n:sha(E/n) for n in ['check_readonly.py','audit_progress.py','record_progress.py','final_readonly.py','audit_final_local.py']},all_six_embedded_CPU_completion_sources_adapted=True,expected_remaining_startup_locked_collections=18,engineering_final_status='STATIC_ONLY_NOT_ACTUAL_ACCEPTANCE')
(R/'FINAL_WRAPPER_REVIEW.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
header=f'''## 2026-11-02 17:21 用户要求立即并行L005 seed42/43：已实际启动并通过进度审计

唯一部署deploy_remote_20261102_171927，engineering_live_parallel_3090_20261102_v1合同{active['engineering_contract']}、9冻结项，controller8994/watchdog8995。原L00542 launcher4689/GPU4694未收信号、未重启，仍为原完整训练进程；新L00543 launcher8996/GPU9000已实际训练。17:21:41 check_readonly_20261102_172140审计PASS，171951及其164759前缀全部精确保留，135科学+5设备+8恢复+6检查+29原并行+9本工程项实际不变。L00541完整400轮40采集；L00542完整265轮(Task8 65)，58485接受101270生产62隔离30autograd20采集；L00543完整18轮2826生产、自身4隔离8autograd、初态/COPY_EQ/CHECK_EQ通过。合计683轮166811接受299396生产157自身隔离79准入autograd60采集0正式。未来阶段不能套PASS。

旧4347/8608/8609仅等待调度器及其watchdog为T，禁止手动恢复，否则可重复启动43。新LIVE_COMMITTED/LIVE_LAUNCH已验真，真正根锁仍由冻结4347持有，等待现有42自然400轮40采集、launcher内核Z/exit0、prefix/hash通过后新owner才结束旧三个调度进程并取得根锁。HANDOFF_COMPLETE尚不存在，不称此边界已完成。新owner若LIVE_COMMITTED后失败，watchdog保持旧等待器冻结，OWNER_LOST_AFTER_LIVE_COMMIT必须先审计，不盲恢复。禁止再次运行任何dispatch/controller/build/validation或已完成轮。

当前42解释器不改；未来准确路径collect.py子进程通过专属.pth启动锁串行采集，其他进程不匹配/关闭标记均不受影响。零更新/零环境模拟双进程互斥、固定随机输出/sys.path、禁用/不匹配测试实际PASS；最初runpy改变argv0而失败的零成本attempt保留，改为原argv下exec后PASS。本生产快照尚无实际采集锁事件，未来应核42剩Task8/9、43及L020共18次acquired/released不重叠和每次10采集。采集锁在导入torch/环境前等待；训练同时运行不增加采集次数。

复用已实际8阶段双λ完整epoch串并行精确验证，无新增optimizer或环境诊断。工程累计34549optimizer/83autograd单列，旧中断140已记额外候选+未记账UNKNOWN保留。30样本实际双生产GPU窗口均77.13%、范围27–93%、峰值4885MiB(约4.77GiB)；42Task8/43Task6不同工作量，不能把它或之前1.635倍隔离吞吐直接称全run提速。下一150分钟在当前42整run预计完成附近复查自然根锁交接及实际采集互斥，再按当前并行run/整组正式余时调整，不固定高频；不按时间断言完成。

后续顺序L00542/43并行全部训练完成→L005600串行正式→L020最多两训练并行(采集互斥)→L020600串行正式。只用本工程check/audit_progress/record_progress/final_readonly/audit_final_local，最终真实来源仅本工程controller.exit0且无自身进程；根及所有旧工程exit保持不存在。外置final所有六case实际完成来源已静态适配，仍非最终CPU接受。最终实际24CP/6恢复/24teacher/1200noise/54600gate和成本、硬件迁移/开发集重复使用/预训练来源限制验收，再compare_primary_s解释发布。

专属启动文件/data/mlr_ifa/env/lib/python3.8/site-packages/zzz_mlr_lambda_live_collection_20261102.pth已按SHA冻结；全部最终验收后核无目标collect进程，核本工程合同SHA再清理仅这个自有文件和HOOK_ENABLED，保留原文本/hash及清理证据。不得提前删除以免未来collection失去互斥；不碰其他site-packages文件。运输旧目录worker3700，先核status/PID不重启活跃worker不输出凭据。全部最终完成发布后才删除同一object-s-r-run监控，健康静默。

'''
for n in ['MONITOR_HANDOFF.md','REPORT.md']:
 p=R/n;p.write_text(header+p.read_text(encoding='utf8'),encoding='utf8')
report='''# L005 seed42/43即时并行工程观察

用户要求无需等待0.2组，当前seed42不中断，立即启动0.05 seed43。此前等待自然边界是保守调度实现，不是λ_R或实验方法的限制。难点是现有seed42未安装采集互斥；仅让未来seed43加锁不能确保两者采集串行。本次为限定五条实际collect.py路径的未来子解释器安装预导入互斥，不修改已运行seed42的训练状态或冻结科学源码。

'''+header+'''本地直接执行实际controller.main的模拟控制流，覆盖42先完成/43先完成，以及两者任一失败：43仅启动一次，42现有launcher/GPU从未收到信号，根锁仅在42自然退出后取得，失败后不启动评估或后续run。此模拟不是生产完成证明。独立并行训练验证沿用此前八阶段实际全轮标量/候选完整state/门控/保存加载下一步精确验证，结果和18584诊断调用原账保留；没有重复该GPU训练验证。

本次只是调度工程接受与进度审计，六run科学验收仍PARTIAL；保持λ档、所有轮数/门控/数据/预算/模型/任务顺序和配对评估不变。跨GPU历史差异与已知未记账中断成本不因此次并行而消失。
'''
(R/'LIVE_PARALLEL_REPORT_20261102.md').write_text(report,encoding='utf8')
print(json.dumps(status,ensure_ascii=False))
