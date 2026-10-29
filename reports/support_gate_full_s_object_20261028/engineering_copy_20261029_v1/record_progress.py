from pathlib import Path
import json,sys,datetime
E=Path(__file__).resolve().parent;N=E.parent;D=Path(sys.argv[1]);interval=int(sys.argv[2])
def js(p):return json.loads(p.read_text(encoding='utf8'))
def put(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
v=js(D/'EVIDENCE.json');a=js(D/'LOCAL_AUDIT.json');assert a['status']=='PASS_PROGRESS_NOT_FINAL'
def light(n,default=None):
 x=v['light'].get('engineering_copy_20261029_v1/'+n)
 return json.loads(x['text']) if x else default
start=light('STARTED.json');current=light('STATUS.json',{});handoff=light('HANDOFF_COMPLETE.json')
assert start and not v['canceled_mechanism_processes']
status='PARTIAL_OPTIMIZED_SEED43_OR_EVALUATION' if handoff else 'PARTIAL_SEED42_RUNNING_HANDOFF_ARMED'
out=dict(a,status=status,root='/data/mlr_ifa/experiment_workspaces/SUPPORT_GATE_FULL_S_OBJECT41_42_43_20261028_v1',controller=start['pid'],original_controller=3013,original_controller_state='REPLACED_AFTER_SEED42_NATURAL_EXIT' if handoff else 'STOPPED_COORDINATOR_ONLY_SEED42_UNINTERRUPTED',engineering_status=current,latest_evidence=D.name,next_check_minutes=interval,final_CPU_acceptance='PENDING',scientific_conclusion='PENDING_FORMAL_CLOSED_LOOP')
put(N/'RUN_STATUS.json',out)
active=js(N/'ACTIVE_ROOT.json');active.update(controller=start['pid'],original_controller=3013,engineering_contract=v['engineering_contract_sha256'],completion_source=v['completion_source'],latest_evidence=D.name,observed=v['observed'],status=status);put(N/'ACTIVE_ROOT.json',active)
lines=['# 最新完整 S OBJECT 三 seed：当前交接','',f"## 最新 {a['observation']}：用户授权复制优化验证与安全交接",'',
f"实际验证PASS_FULL_TRAINING_PATH_EXACT：Task6–9各一整轮全部batch，157/227/309/362批，C/Q完整候选与两分支恢复、每步标量、探针、门控、所选完整state和保存加载下一步逐位相等。额外诊断3914 optimizer、{a['engineering']['diagnostic_extra_autograd_calls']} autograd、0环境、0生产接受epoch，独立记账；这是完整路径覆盖，不是重跑400epoch或已经验收三seed。原seed41已有CP/JSON及63冻结项实际保持。",'',
f"工程合同 {v['engineering_contract_sha256']}，目录engineering_copy_20261029_v1，controller{start['pid']}，当前 {json.dumps(current,ensure_ascii=False)}。仅CUDA→CPU后的重复CPUclone被去掉，CPU源clone保留。冻结训练文件未修改，wrapper在新seed43解释器导入时替换函数；原各阶段自身GPU准入保留，并核seed43实际阶段开始完整state复制。",'',
('seed42已自然完成400轮和40采集、launcher内核Z/exit_code0被记录后才结束原等待调度器3013；新调度器持有同一根锁，仅train43→evaluate41→evaluate42→evaluate43。' if handoff else 'seed42继续原launcher5055/train5060，不接收任何信号。只有原等待调度器3013暂停以保留seed43交接点。新调度器等seed42自然退出且400轮/40采集核验后接管；owner watchdog在接管前工程owner失败时只释放旧调度器。不可盲SIGCONT旧调度器或启动其他队列。'),
'禁止重跑guard/validation/dispatch/controller/build或已完成epoch；不得恢复旧机制。原root/controller.exit保持不存在，最终真实完成来源为engineering_copy_20261029_v1/controller.exit==0且无自身进程，不可伪造原exit。','']
for seed,c in a['counts'].items():lines.append(f"seed{seed}：{c['epochs']}/400轮，{c['accepted_updates']}接受/{c['physical_optimizer_calls']}已完成轮物理调用，采集{c['training_collection_completed_episodes']}/40，正式{c['formal_episodes']}/200；阶段状态{json.dumps(c['current'],ensure_ascii=False)}。")
lines+=['',f"合计{a['epochs']}/1200完整epoch、{a['accepted_updates']}/316500接受、{a['physical_training_optimizer_calls']}/585900生产物理调用，自身准入{a['own_isolated_optimizer_calls']}/273、{a['own_extra_autograd_calls']}/123额外autograd，采集{a['training_collection_completed_episodes']}/120、正式{a['formal_episodes']}/600。额外3914诊断调用不混入生产计数。",'',
'科学协议仍是官方OBJECT初始化、Task6–9各100轮，Task6无保持，Task7–9自己的成功缓存q与固定10步C参照门控、两个epsilon0、原4槽分母；每阶段10采集20–29不补采，正式0–19原600步5zero配对Normal。内部C/Q属于新S。禁止扩seed、扫参、改变损失阈值或采样预算、因负结果追加实验。旧机制永久停止，14完整组及Q3轮保留，旧未记账中断成本UNKNOWN。','',
'监控沿用旧目录worker4572，先读.session_control/status，不重启或输出凭据。使用本工程目录check_readonly.py复制到旧目录后queue，按result真实目录复制证据回新目录；engineering_copy_20261029_v1/audit_progress.py NEW PREVIOUS，再engineering_copy_20261029_v1/record_progress.py NEW INTERVAL。禁止使用原record_progress覆盖工程实际controller和完成来源。','',
'本工程final_readonly.py/audit_final_local.py已外置静态适配真实工程exit及单独诊断账目，FINAL_WRAPPER_REVIEW仍STATIC_ONLY；全部自然结束且无自身进程后才实际CPU12CP/3全state/12缓存teacher/600初态noise和全部门控标量验收，再运行根describe_final.py并解释新旧任务、SF/FS、负结果和训练代理局限。每个未来阶段必须自己的准入。','',
f"下一{interval}分钟依据实际阶段吞吐复查，不把约14.6%条件潜在节时当实际整seed提速，不根据时间推断完成。仅实质变化/失败/完成/需行动通知，健康静默。每轮按根AGENTS经publish.py审查轻量code/report/hash/scalars/outcomes，串行nonforce推MLR核remoteSHA/localreceipt，禁CP/state/RNG/cache/凭据。全部三个seed独立验收、解释报告和发布完成后才删除同一监控。"]
text='\n'.join(lines)+'\n';(N/'MONITOR_HANDOFF.md').write_text(text,encoding='utf8');(N/'REPORT.md').write_text(text.replace('当前交接','运行报告',1),encoding='utf8')
print(json.dumps(dict(status=status,evidence=D.name,controller=start['pid'],phase=current,handoff_complete=bool(handoff))))
