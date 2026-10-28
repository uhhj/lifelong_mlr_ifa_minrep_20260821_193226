"""Persist independently audited full-S observation and the user-directed priority."""
from pathlib import Path
import json,sys,datetime,hashlib
R=Path(__file__).resolve().parent;M=R.parent/'20261027_support_gate_forks';D=Path(sys.argv[1]);interval=int(sys.argv[2])
def js(p):return json.loads(p.read_text(encoding='utf8'))
def put(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
v=js(D/'EVIDENCE.json');a=js(D/'LOCAL_AUDIT.json');dep=js(R/'deploy_remote_20261028_195120'/'EVIDENCE.json')
assert a['status']=='PASS_PROGRESS_NOT_FINAL' and a['contract']==dep['contract']
assert not v['canceled_mechanism_processes'] and v['gpu_diagnostic']['exit']==0
assert v['processes'] or v['light'].get('controller.exit',{}).get('text','').strip()=='0'
status=dict(a,status='PARTIAL_TRAINING_OR_EVALUATION',root=dep['root'],controller=dep['controller'],latest_evidence=D.name,next_check_minutes=interval,mechanism_status='USER_STOPPED_PARTIAL_PRESERVED_NOT_RESUMABLE_WITHOUT_NEW_USER_REQUEST',final_CPU_acceptance='PENDING',final_wrapper_adaptation='REQUIRED_BEFORE_FINAL',scientific_conclusion='PENDING_FORMAL_CLOSED_LOOP')
put(R/'RUN_STATUS.json',status)
put(R/'ACTIVE_ROOT.json',dict(root=dep['root'],controller=dep['controller'],contract=dep['contract'],children=dep['children'],frozen_files=dep['frozen_files'],observed=v['observed'],latest_evidence=D.name,status=status['status'],unique_dispatch_evidence='deploy_remote_20261028_195120'))
lines=[f"## 最新 {a['observation']}：最新完整 S 已启动",'',f"独立 root `{dep['root']}`，controller {dep['controller']}，合同 `{dep['contract']}`，{dep['frozen_files']} 冻结项实际通过。唯一部署证据 deploy_remote_20261028_195120。禁止重跑 build/deploy/controller 或重复训练已完成 epoch。",'']
for seed,c in a['counts'].items():
 lines.append(f"seed{seed}：{c['epochs']}/400 完整 epoch，{c['accepted_updates']} 接受更新 / {c['physical_optimizer_calls']} 已完成轮物理训练调用，采集已完成 {c['training_collection_completed_episodes']}/40，正式 {c['formal_episodes']}/200。当前 {json.dumps(c['current'],ensure_ascii=False)}。")
lines += ['',f"合计 {a['epochs']}/1200 完整 epoch，{a['accepted_updates']}/316500 接受、{a['physical_training_optimizer_calls']}/585900 完整轮物理训练调用。自身隔离 {a['own_isolated_optimizer_calls']}、额外 autograd {a['own_extra_autograd_calls']}。当前快照 {D.name}/LOCAL_AUDIT.json 实际 PASS_PROGRESS_NOT_FINAL。",'',
'Task6 无保持；Task7–9 的逐任务 q 来自各 seed 自己的 10 次成功采集缓存，使用固定 10 步 C/Q 完整状态门控。内部 C/Q 候选属于新 S，不是恢复旧机制对照。每 task100epoch、每 task10训练采集20–29、不补采；正式各 stage 已见任务0–19，600steps5zero配对Normal。三 seed120采集+600正式，273隔离更新/123准入autograd另记，训练诊断最多108autograd。原科学阈值、损失、块长和预算不变。', '',
'旧机制已按用户最新指令停止，保留 GOAL三seed12组 + OBJECT41 C/S 共14完整组及 Q3轮，G/U与正式0。停止证据在旧目录 deploy_remote_20261028_193702/EVIDENCE.json，旧 root/user_stop_20261028/RESULT.json；停止后进程为空，56冻结项通过，所有CP/恢复点/日志保留。旧中断及停止时未记账部分成本仍UNKNOWN。不得恢复旧队列，也不等待其最终验收。', '',
'队列顺序：train seed41完整6–9 → train42 → train43 → evaluate41 → evaluate42 → evaluate43。各阶段必须自己的准入通过，不能套首阶段 PASS。最终仅 controller.exit0且无自身进程后独立验收12CP/3全state恢复/12缓存teacher/600初态noise/门控决策/闭环标量。现有旧S final wrappers尚待在本新目录外置适配支持权重、门控和273隔离调用；禁止直接套旧66调用或旧机制1120预算。生成及静态通过不算实际CPU验收。', '',
f'下一{interval}分钟检查首次门控阶段的准入与吞吐，再依据当前整run实际耗时调整；此为初次运行核实安排，不是总完成ETA。只实质变化/完成/新失败/需行动通知，健康无变化静默。', '',
'监控传输仍用旧目录现有worker4572，先读.session_control/status.json，不重启活跃worker或打印凭据。本目录check_readonly.py复制到旧目录同名包装，queue_remote.py check_readonly.py后从实际result.output第一行读真实目录，再把完整轻量证据复制到本目录。运行 audit_progress.py NEW [PREVIOUS]，再 record_progress.py NEW INTERVAL。所有stage/GATES前缀精确核对；异常先核进程/hash/退出/完整恢复和已耗成本，不盲重启环境或部分块。', '',
'按根AGENTS串行运行本目录publish.py以及旧目录停止报告的publish.py，审查轻量文件、nonforce推MLR、remoteSHA/localreceipt。禁止发布CP/状态/缓存/凭据。最终三个seed独立验收、分析报告、发布后再删除同一个s-object-seed监控。']
text='\n'.join(lines)+'\n'
(R/'MONITOR_HANDOFF.md').write_text('# 最新完整 S OBJECT 三 seed：当前交接\n\n'+text,encoding='utf8')
(R/'REPORT.md').write_text('# 最新完整 S OBJECT 三 seed：运行报告\n\n'+text,encoding='utf8')
print(json.dumps(dict(status=status['status'],observation=a['observation'],evidence=D.name,next_check_minutes=interval)))
