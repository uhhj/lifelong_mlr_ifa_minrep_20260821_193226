"""Record only actual independently audited snapshots for the GOAL run."""
from pathlib import Path
import json,sys,datetime
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);interval=int(sys.argv[2])
js=lambda p:json.loads(p.read_text(encoding='utf8'))
def put(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
a=js(D/'LOCAL_AUDIT.json');v=js(D/'EVIDENCE.json');active=js(R/'ACTIVE_ROOT.json')
assert a['status']=='PASS_PROGRESS_NOT_FINAL' and a['contract']==active['contract']
assert v['gpu_diagnostic']['exit']==0 and not v['canceled_mechanism_processes']
status=dict(a,status='PARTIAL_TRAINING_OR_EVALUATION',root=active['root'],controller=active['controller'],latest_evidence=D.name,next_check_minutes=interval,final_CPU_acceptance='PENDING',scientific_conclusion='PENDING_FORMAL_CLOSED_LOOP',primary_reference='Existing accepted complete C GOAL41/42/43')
put(R/'RUN_STATUS.json',status)
active.update(observed=v['observed'],latest_evidence=D.name,status=status['status']);put(R/'ACTIVE_ROOT.json',active)
lines=['# 最新完整 S GOAL 三 seed：当前交接','',f"实际观察 {a['observation']}，{D.name}/LOCAL_AUDIT.json={a['status']}。",'',f"唯一root `{active['root']}`，controller {active['controller']}，合同 `{active['contract']}`，{active['frozen_files']}冻结项。唯一部署证据{active['unique_dispatch_evidence']}。禁止再次运行prepare_local/deploy/controller或重复已完成epoch。",'']
for seed,c in a['counts'].items():
 lines.append(f"seed{seed}：{c['epochs']}/400完整epoch，{c['accepted_updates']}/76100接受、{c['physical_optimizer_calls']}/142200已完成轮物理调用；采集{c['training_collection_completed_episodes']}/40、正式{c['formal_episodes']}/200。当前{json.dumps(c['current'],ensure_ascii=False)}。")
 if c['recent_epoch_mean_seconds'] is not None:lines.append(f"当前阶段最近至多10轮均{c['recent_epoch_mean_seconds']:.4f}秒；该阶段纯计算余{c['current_stage_compute_minutes_estimate']}分钟，保存/采集/后续另计，首轮不能当稳定吞吐。")
lines += ['',f"累计{a['epochs']}/1200epoch、{a['accepted_updates']}/228300接受、{a['physical_training_optimizer_calls']}/426600生产物理调用；自己的隔离{a['own_isolated_optimizer_calls']}/273、额外准入autograd{a['own_extra_autograd_calls']}/123。训练方向最多108autograd另记。",'',
'先完整读本目录PROTOCOL.md、IMPLEMENTATION_NOTES.md、RUN_STATUS.json、ACTIVE_ROOT.json和最新EVIDENCE/LOCAL_AUDIT。仅GOAL最新完整S seed41/42/43；OBJECT已独立验收完成，旧机制永久停止，均不得重跑。主对照为已有完整C（GOAL41为accepted v3、42/43为accepted扩展）；不新训C，不以原S替代主对照。', '',
'官方GOAL初始化、Task6–9各100epoch，Task6无保持，7–9自己成功缓存q与固定10步同state C/Q零容差门控、原4槽分母。三seed120采集20–29、600正式0–19，600steps5zero配对Normal；不补采/重评/扫参/扩大seed/因负结果追加实验。内部C/Q是方法组成。', '',
'已复用逐位验证的复制优化，仅CUDA转CPU后的重复clone去掉，CPU源clone保留。每stage新解释器自身初始化capture与原函数对比COPY_EQ，自己的4或29调用准入必须实际PASS。OBJECT既有3914诊断不能计入GOAL，也不能套给尚未准入的GOAL阶段。', '',
'顺序train41→train42→train43→evaluate41→evaluate42→evaluate43。仅自然root/controller.exit0且无自身进程后，才运行本目录final_readonly.py实际CPU12CP/3全state/12缓存teacher/600初态noise/20100门控标量验收，audit_final_local.py FINAL→describe_final.py FINAL→compare_primary_c.py FINAL，解释新旧任务/SF/FS/负结果并发布。FINAL_WRAPPER_REVIEW仅静态，不是接受。', '',
'运输使用旧目录20261027_support_gate_forks现有worker4572，先读.session_control/status.json，不重启活跃worker或输出凭据。本目录check_readonly.py复制至旧目录，queue_remote.py check_readonly.py；按result.output第一行实际目录复制完整证据回来，不猜秒数；audit_progress.py NEW PREVIOUS→record_progress.py NEW INTERVAL。异常先核进程/hash/退出/完整恢复和已耗候选成本，禁止盲重启部分块/环境attempt。', '',
f'下一{interval}分钟复查：初期核首个门控阶段自己的准入和吞吐，之后按整seed训练含采集或全部评估真实剩余时间调整，不固定高频，不靠时钟推断完成。健康进展静默，只实质变化/完成/新失败/需行动通知。', '',
'每次进度审计按根AGENTS经本目录publish.py审查轻量code/report/hash/scalars/outcomes，串行nonforce推MLR核remoteSHA/localreceipt；禁模型/CP/state/RNG/cache/凭据/未审bundle。沿用本轮GOAL新监控（见MONITOR_RECEIPT.json），不是已删除s-object-seed。三个seed独立验收、主C比较/解释报告和发布后才删除本轮监控。']
text='\n'.join(lines)+'\n';(R/'MONITOR_HANDOFF.md').write_text(text,encoding='utf8');(R/'REPORT.md').write_text(text.replace('当前交接','运行报告',1),encoding='utf8')
print(json.dumps(dict(status=status['status'],observation=a['observation'],epochs=a['epochs'],next_check_minutes=interval)))
