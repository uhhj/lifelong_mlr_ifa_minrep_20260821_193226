"""Record an actually audited GOAL snapshot without changing production files."""
from pathlib import Path
import json,sys,hashlib
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);interval=int(sys.argv[2])
def js(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
a=js(D/'LOCAL_AUDIT.json');e=js(D/'EVIDENCE.json');active=js(R/'ACTIVE_ROOT.json')
assert a['snapshot_sha256']==hashlib.sha256((D/'EVIDENCE.json').read_bytes()).hexdigest()
assert a['contract']==active['contract'] and a['status']=='PASS_PROGRESS_NOT_FINAL'
v=js(R/'ENGINEERING_VALIDATION.json');assert v['optimizer_completed']==5704
out=dict(a,status='PARTIAL_GOAL_TRAINING_OR_EVALUATION',latest_evidence=D.name,production_launched=True,object_closed=True,next_check_minutes=interval,engineering_optimizer_calls=5704,engineering_extra_autograd_calls=36,engineering_environment_calls=0,final_acceptance='NOT_EXECUTED')
write(R/'RUN_STATUS.json',out)
prep=js(R/'PREPARATION_STATUS.json');prep.update(status='PRODUCTION_DEPLOYED_PARTIAL',production_launched=True,remaining=['own per-stage admissions and exact production prefixes','natural controller exit0 and no own processes','actual final CPU acceptance, paired comparison, interpretation and publication'])
write(R/'PREPARATION_STATUS.json',prep)
lines=['# GOAL 固定 λ_R 六run：当前交接','',f"实际观察 {a['observation']}，{D.name}/LOCAL_AUDIT={a['status']}。",'',f"唯一root `{active['root']}`；controller {active['controller']}；合同 `{active['contract']}`。禁止重复compose/freeze/deploy/controller/工程验证或已完成轮。",'', '|case|完整轮/400|接受更新|生产物理|采集/40|正式/200|','|---|---:|---:|---:|---:|---:|']
for case,c in a['counts'].items():lines.append('|%s|%d|%d|%d|%d|%d|'%(case,c['epochs'],c['accepted_updates'],c['physical_optimizer_calls'],c['training_collection_completed_episodes'],c['formal_episodes']))
lines+=['',f"累计{a['epochs']}/2400轮、{a['accepted_updates']}/456600接受、{a['physical_training_optimizer_calls']}/853200生产、{a['own_isolated_optimizer_calls']}/546自身隔离、{a['own_extra_autograd_calls']}/246准入autograd、{a['training_collection_completed_episodes']}/240采集、{a['formal_episodes']}/1200正式。方向最多216autograd另记。",'',f'下一{interval}分钟复查，初期一次性核首个保持阶段自身准入及吞吐，后续按整run或整组正式实测剩余时间调整，不固定高频、不按时钟推断完成。','',
'先完整读PROTOCOL、IMPLEMENTATION_NOTES、RUN_STATUS、ACTIVE_ROOT、SOURCE_REVIEW、FINAL_WRAPPER_REVIEW、OBJECT_COMPLETION_BINDING、ENGINEERING_ACCEPTANCE和最新实际EVIDENCE/LOCAL_AUDIT。只新增GOAL .05/.2各41/42/43；旧GOAL .1的600正式只读两次配对，不是1200独立基线。每任务100轮、原10步双候选零epsilon、自己成功缓存q与4槽分母、原RA/损失其余项/采样预算保持；同λ准入。禁止扩seed、扫参、补采重评或按成绩选λ。',
'最多两训练，显式c10d localhost:0独立UUID，不用--standalone。L005优先训练、空槽可L02041；L005训练齐全优先600串行正式，已启动L02041可继续；之后L02042/43最多两训练，最后L020600正式。全部24采集和6评估共享environment_locked全局flock，生产事件须逐项审计互斥，无site-packages hook。每阶段自己的GPU准入/COPY_EQ/BRANCH_COPY_EQ/CHECK_EQ必须实际通过，不套未来PASS。官方初态tensor/RNG同一；跨GPU Task6记录实际比较，不预称相等或λ单因素硬件完全受控。',
'GOAL工程独立验证实际PASS_GOAL_FULL_PATH_BOTH_LAMBDAS，两λ八阶段全batch逐位相等，5704隔离optimizer/36autograd/0环境，退出0无诊断进程，环境锁零环境模拟PASS。证据engineering_validation_20261103_v1/capacity_readonly_20261103_133804，ENGINEERING_ACCEPTANCE绑定123项生产源码UTF8 LF文本SHA。此前两个部署前包装零成本失败保留，不重复任何验证入口。工程5704/36不混生产或OBJECT历史35418/94及540+UNKNOWN。',
'OBJECT已全接受/报告发布/按SHA清理自有pth与HOOK_ENABLED，OBJECT_COMPLETION_BINDING及原OBJECT_CLOSURE为闭环；不再跑依赖已清理pth的旧cross检查，不重训重评。C/Q/旧机制永久停止，已有S只读。',
'最终仅GOAL本root/controller.exit0且无自身进程后，才运行final_readonly→audit_final_local→compare_primary_s，实际CPU24CP/6恢复/24teacher/1200初态noise/40200门控及完整预算/环境互斥验收。FINAL_WRAPPER_REVIEW仅STATIC_ONLY，不套OBJECT的scale例外。包装失败保留证据外置版本化修复，不重训重评。随后解释三档、新旧任务、SF/FS、全部负单元、训练代理与成本、硬件迁移、开发集反复使用和预训练来源未闭合限制；不称独立确认或全局无泄露。',
'运输旧机制目录.session_control现worker3700，先读status并核PID，活跃不重启不输出凭据；本目录check_readonly复制旧目录queue，按result.output第一行真实目录取证，audit_progress.py NEW PREVIOUS→record_progress.py NEW INTERVAL。任何异常先核hash、进程/退出、完整恢复及实际已耗成本，禁止盲恢复部分块/环境attempt。',
'每次检查按AGENTS经本目录publish.py审核轻量code/report/hash/scalars/outcomes，串行nonforce推MLR核remoteSHA/receipt；禁CP/model/state/RNG/cache/凭据/未审bundle。不足容量不删旧模型或付费。健康静默，仅实质变化/完成/新失败/需行动通知。沿用object-s-r-run同一监控，GOAL也全部独立接受、解释发布后才关闭。']
(R/'MONITOR_HANDOFF.md').write_text('\n\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps(dict(status=out['status'],observed=a['observation'],epochs=a['epochs'],next_check_minutes=interval)))
