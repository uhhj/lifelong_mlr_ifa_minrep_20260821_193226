"""Local evidence recording only; no remote or scientific execution."""
from pathlib import Path
import json, hashlib

R = Path(__file__).resolve().parent
D = R / 'check_readonly_20261101_191813'
E = R / 'engineering_device_snapshot_20261101_v1'
js = lambda p: json.loads(p.read_text(encoding='utf8'))
def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')

a = js(D/'LOCAL_AUDIT.json')
assert a['status'] == 'PASS_PROGRESS_NOT_FINAL'
v = js(D/'EVIDENCE.json')
h = json.loads(v['light']['L005_41/EPOCHS_S.json']['text'])
t = [x['seconds'] for x in h if x['stage'] == 7]
mean, recent = sum(t)/len(t), sum(t[-10:])/10
timing = dict(observation=a['observation'], evidence=D.name,
    evidence_sha256=a['snapshot_sha256'], current_case='L005_41',
    current_production_optimization_active=False, stage7_epochs=len(t),
    stage7_full_mean_seconds=mean, stage7_recent10_seconds=recent,
    stage7_remaining_minutes=(100-len(t))*recent/60,
    hypothetical_remaining_pure_minutes=[(100-len(t)+100*(309+362)/227)*x/60 for x in [recent,mean]],
    formula='(Task7 remaining epochs + 100*(309+362)/227) * Task7 seconds / 60',
    next_check_minutes=240,
    limits=['Task8/9 throughput not measured; own old-cache probes grow with stages.',
            'Excludes preparation, saving and remaining 30 collections.',
            'Diagnostic sharing ended; recent improvement is NOT production optimization speedup.',
            '240-minute next check is near the early side of current whole-run planning, not a fixed short poll.',
            'No reliable completion date for all six runs from this observation.'])
write(D/'TIMING_AUDIT.json', timing)
s = js(R/'ENGINEERING_VALIDATION_STATUS.json')
s.update(engineering_dispatch_state='ARMED_WAIT_CURRENT_RUN_NATURAL_EXIT',
    engineering_dispatch_evidence='deploy_remote_20261101_191752',
    engineering_controller=3899, watchdog=3900,
    engineering_contract=a['engineering']['contract_sha256'],
    production_activation=False, latest_production_evidence=D.name)
write(R/'ENGINEERING_VALIDATION_STATUS.json', s)
status = js(R/'RUN_STATUS.json')
status.update(engineering_state='ARMED_WAIT_CURRENT_RUN_NATURAL_EXIT',
    engineering_validation=dict(status='PASS_PAIRED_FULL_EPOCH_PATH_BOTH_LAMBDAS',
        evidence='capacity_readonly_20261101_191620', optimizer_started=7828,
        optimizer_completed=7828, extra_autograd=36, environments=0,
        production_epochs=0, production_activation=False, diagnostic_process_exited=True))
write(R/'RUN_STATUS.json', status)

latest = '''## 2026-11-01 19:18：完整等价验证通过，安全交接已ARMED，当前训练未切换

capacity_readonly_20261101_191620实际读取VALIDATION PASS：两档λ的Task6–9共8阶段，各完整一轮157/227/309/362个batch，全部每步标量、C/Q分支完整state、探针、门控决定、所选state、保存加载下一步及方向诊断逐项相等。诊断进程3441已退出，实际7828 started==completed、36额外autograd、0环境、0生产epoch，单独记账。原135科学项及诊断使用的只读基线CP/cache SHA精确保留。覆盖完整路径，不等于验证全部400轮，也不是科学最终接受。

已完整审查工程payload及check/audit/record/final的真实完成来源适配。19:17:58唯一部署engineering_device_snapshot_20261101_v1，controller3899/watchdog3900，工程合同b9ebd306e7ac3ce1fba2ce107ba5b2fa6b4fd20f62164790bf7ced4897b6086c、5工程冻结项。禁止再次执行dispatch_handoff、任何原/新controller、验证dispatch或已完成轮。

check_readonly_20261101_191813/LOCAL_AUDIT及ENGINEERING_HANDOFF_LOCAL_AUDIT实际PASS，184739全部epoch/门控/诊断前缀精确保留。原等待调度器2163仅被PID级SIGSTOP，当前launcher2164/训练2169未接收信号且继续运行；新owner3899与watchdog3900为S，ARMED存在、HANDOFF_COMPLETE尚不存在，无FAILURE/WATCHDOG_RELEASE。当前仍L005_41原训练代码，优化尚未用于任何生产run。

新owner等待当前L005_41自然完成400轮40采集、launcher内核Z且exit_code0，并核完整记录/前缀/hash后才结束旧等待调度器、取得同一root锁、记录真实train_S.exit0并接续五个未来run。仅替换各新解释器内四次门控块快照，保持各tensor原设备独立clone，轮末CP和准入仍原CPU格式；每阶段实际BRANCH_COPY_EQ及自身GPU准入保留。原root/controller.exit必须不存在；最终唯一完成来源为engineering_device_snapshot_20261101_v1/controller.exit0且无自身进程。若WATCHDOG_RELEASE出现，应审计并通知，不能宣称交接成功或盲重复部署。

本次L005_41 Task6完整100轮、Task7完整95轮，共195轮37265接受58830生产、33自身隔离19准入autograd、10采集0正式；其他五run未开始。当前Task7近10轮57.1295秒、全95轮110.8830秒，尚余5轮约4.76分钟纯计算。Task8/9未测，按批次数比例的当前整run剩余纯计算条件范围286–556分钟，保存/准备/余30采集另计，不能将近期变快归因于尚未启用的优化。全部六run目前没有可靠完成日期。下一240分钟在当前整run结束预计较早侧复查，覆盖旧20分钟验证等待间隔，不固定短轮询。详见本快照TIMING_AUDIT。

运输使用旧目录.session_control现有worker28064，先读status核PID，不重启活跃worker或输出凭据。以后只用工程目录check_readonly.py复制至旧目录queue；按result实际目录复制回本目录，再工程audit_progress.py NEW PREVIOUS→工程record_progress.py NEW INTERVAL。根旧记录器/旧final包装不再适用。工程final包装仍仅静态，全部自然完成后才实际CPU24CP/6恢复/24teacher/1200noise/54600门控验收，随后根compare_primary_s和完整解释发布。7828诊断调用/36autograd必须单列，不混科学预算。原源、原证据及失败记录保留。

只做OBJECT λ=.05/.2各三seed，已有.1只读；科学损失、门控阈值、批次、采样及预算不变。两成潜在整体节时仍是条件推测，不能用92.05%复制微基准节时冒充整run提速。健康静默，只实质变化/完成/新失败/需行动通知；全部六run独立验收解释发布后才删除同一object-s-r-run监控。

'''
for name in ['MONITOR_HANDOFF.md', 'REPORT.md']:
    p = R/name
    p.write_text(latest+p.read_text(encoding='utf8'), encoding='utf8')
p=R/'PROFILE_OPTIMIZATION_REPORT_20261101.md'
p.write_text('# 最新状态：完整路径等价通过，已部署等待自然边界\n\n'+latest+
             '\n---\n\n以下为18:53以前定位与验证历史，进度由上文覆盖。\n\n'+
             p.read_text(encoding='utf8'), encoding='utf8')
review=js(R/'FINAL_WRAPPER_REVIEW.json')
review.update(engineering_completion_source='engineering_device_snapshot_20261101_v1/controller.exit',
    engineering_final_status='STATIC_ONLY_NOT_ACTUAL_ACCEPTANCE',
    engineering_wrappers={n:hashlib.sha256((E/n).read_bytes()).hexdigest()
        for n in ['check_readonly.py','audit_progress.py','record_progress.py','final_readonly.py','audit_final_local.py']})
write(R/'FINAL_WRAPPER_REVIEW.json',review)
print(json.dumps(dict(status='RECORDED_ARMED_NOT_SWITCHED',observation=a['observation'],next_minutes=240)))
