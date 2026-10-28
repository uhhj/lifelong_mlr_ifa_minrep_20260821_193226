"""Record audited recovery progress without replacing historical interruption evidence."""
import json,sys
from pathlib import Path
R=Path(__file__).resolve().parent.parent;D=R/sys.argv[1];interval=int(sys.argv[2])
def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
a=read(D/'LOCAL_AUDIT.json');ra=read(D/'RESUME_AUDIT.json');v=read(D/'EVIDENCE.json')
assert ra['status']=='PASS_RESUME_PROGRESS_NOT_FINAL' and a['status']=='PASS_PROGRESS_NOT_FINAL'
assert v['processes'] and v['gpu_diagnostic']['exit']==0
status=read(R/'RUN_STATUS.json');status.update(a)
status.update(status='PARTIAL_RESUMED_TRAINING',controller=2020,original_controller=29838,latest_evidence=D.name,resume=ra)
status['connectivity']=dict(status='RESTORED',observation=a['observation'],latest_evidence=D.name,worker_restarted=False,next_check_minutes=interval)
status['resume']['completion_marker']='engineering_resume_20261028_v1/controller.exit'
status['resume']['final_wrapper_adaptation_pending']=True
status['interruption']['recovery_executed']=True
status['interruption']['next_action']='Resumed from epoch 8; retain original failure and unknown partial-epoch cost'
write(R/'RUN_STATUS.json',status)
active=read(R/'ACTIVE_ROOT.json');active.update(status=status['status'],controller=2020,original_controller=29838,observed=v['observed'],latest_evidence=D.name,engineering_contract=ra['engineering_contract_sha256'],latest_diagnostic_evidence='capacity_readonly_20261028_161926',latest_connectivity_evidence=D.name)
write(R/'ACTIVE_ROOT.json',active)
n=a['counts']['OBJECT_41']['S']['epochs']
text=f'''## 最新 {a['observation']}：恢复后进度复查

用户“已启动”后，16:17 SSH/GPU恢复，16:19 CPU/PCI/驱动诊断通过。16:28:12已唯一启动版本化 engineering_resume_20261028_v1/controller_resume.py，controller2020、GPU进程2037。禁止再次部署/启动原controller或本恢复入口。工程合同 {ra['engineering_contract_sha256']}，原56冻结项及工程4项SHA实际通过。原13组完成epoch文件与S前7轮逐字段精确保留。

最新实际快照 {D.name}，GPU完整state恢复、原29调用准入及下一步双重重放实际PASS，诊断后恢复原checkpoint全state，再从第8轮继续。S现已完整{n}/100轮；累计{a['epochs']}完整epoch、{a['accepted_updates']}接受、{a['physical_training_optimizer_calls']}已记账训练调用，原隔离{a['own_isolated_optimizer_calls']}加本次31=共{ra['total_known_isolated_optimizer_calls']}，恢复额外autograd11、环境0，正式{a['formal_episodes']}/1120。中断时未记账部分轮物理成本仍UNKNOWN，不能声称总实际成本精确或为零。原checkpoint已另存私有 interrupted_20261028_epoch7.pth，原失败日志/工程attempt保留。

恢复后已完成{ra['resumed_complete_epochs']}轮，最近至多10轮均{ra['resumed_recent_epoch_mean_seconds']:.6f}秒；恢复后轮时有波动，不沿用旧50分钟预测。当前线性纯计算估计{ra['S_remaining_compute_minutes_estimate']:.2f}分钟仅作暂定，不含保存/准备或Q/G/U与正式评估。下一{interval}分钟在当前S整run预计结束附近复查，再按实际当前run吞吐更新同一监控，不恢复固定短周期；这是检查安排而非完成承诺。单次GPU利用率采样不能判断持续瓶颈。容量已把额外346305333B私有备份计入峰值，root留1GiB后余{a['capacity']['/root']['spare']}B，data扣剩余和2GiB后余{a['capacity']['/data']['spare']}B。

后续队列为S剩余→原Q/G/U各自准入与100轮→原4case正式评估。原根controller.exit保持不存在；最终成功标记改为 engineering_resume_20261028_v1/controller.exit==0 且无自身进程。现有 final_readonly.py/audit_final_local.py/describe_final.py 尚需外置版本化适配这一真实完成来源和恢复额外31调用、UNKNOWN成本，不能伪造根exit；仅全部训练/评估完成后才执行真实最终验收。当前record_progress.py仍硬编码旧deployment，不使用；使用audit_progress→audit_transition→engineering_resume_20261028_v1/audit_resume.py，再record_resume.py SNAPSHOT INTERVAL记录。capacity_readonly.py仍固定next_epoch7，续训后不要原样执行。

此次工程恢复未改变科学损失/阈值/采样/预算，没有重训任何已完成轮；前7轮记录相同不等于能补回中断未保存部分成本。整体PARTIAL_RESUMED_TRAINING，科学最终验收与闭环结论待后续。机制验收分析发布后仍须构建部署并完成仅最新完整S OBJECT41/42/43，不能删除监控。本段覆盖此前失联/GPU不可见与原controller29838活跃描述；下方历史记录只作追溯。

'''
p=R/'MONITOR_HANDOFF.md';old=p.read_text(encoding='utf8');title,body=old.split('\n',1);p.write_text(title+'\n\n'+text+body.lstrip(),encoding='utf8')
p=R/'REPORT.md';p.write_text(text+p.read_text(encoding='utf8'),encoding='utf8')
print(json.dumps(dict(status=status['status'],evidence=D.name,next_check_minutes=interval),ensure_ascii=False))
