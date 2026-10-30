"""Record an audited observation; no scientific dispatch or resume."""
from pathlib import Path
import json,sys,datetime
E=Path(__file__).resolve().parent;R=E.parent;D=Path(sys.argv[1]);interval=int(sys.argv[2])
js=lambda p:json.loads(p.read_text(encoding='utf8'));a=js(D/'LOCAL_AUDIT.json');e=js(D/'EVIDENCE.json')
dep=js(R/'ACTIVE_ROOT.json');assert a['contract']==dep['contract']
now=datetime.datetime.fromtimestamp(e['observed'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
s=dict(status='PARTIAL_C_TRAINING_OR_EVALUATION' if a['status']=='PASS_PROGRESS_NOT_FINAL' else a['status'],observation=now,root=dep['root'],controller=dep['controller'],contract=a['contract'],frozen_files=e['source_files'],latest_evidence=D.name,epochs=sum(x['epochs'] for x in a['counts'].values()),accepted_updates=sum(x['steps'] for x in a['counts'].values()),production_physical_calls=sum(x['steps'] for x in a['counts'].values()),own_isolated_calls=a['isolated_optimizer_calls'],new_formal_episodes=a['formal_episodes'],new_training_collection=0,new_S_runs=0,counts=a['counts'],rates_last10=a['rates_last10'],evaluations=a['evaluations'],phase=a['phase'],failure=a['failure'],processes=a['processes'],next_check_minutes=interval,completion_source='engineering_suite_order_20261030_v1/controller.exit',engineering_contract=a['engineering_contract'],final_CPU_acceptance='PENDING_ACTUAL_FINAL',source_evidence_sha256=a['source_evidence_sha256'])
(R/'RUN_STATUS.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
dep.update(status=s['status'],latest_evidence=D.name,observed=now);(R/'ACTIVE_ROOT.json').write_text(json.dumps(dep,indent=2)+'\n')
lines=['# 仅C匹配已有完整S预算：当前交接','',f'实际观察 {now}，{D.name}/LOCAL_AUDIT.json={a["status"]}。','',f'唯一root `{dep["root"]}`，controller{dep["controller"]}，合同`{a["contract"]}`，{e["source_files"]}冻结项。禁止重跑prepare/deploy/controller或已完成轮。用户最新明确S不重跑，只跑C；已有GOAL/OBJECT六S仅只读配对，旧机制永久取消。','', '|case|完整epoch/700|生产调用|正式/200|','|---|---:|---:|---:|']
for case,x in a['counts'].items():lines.append('|%s|%d|%d|%d|'%(case,x['epochs'],x['steps'],a['evaluations'][case]['episodes']))
lines += ['',f'累计{s["epochs"]}/4200轮、{s["production_physical_calls"]}/1012500生产调用、{s["own_isolated_calls"]}/546隔离调用、新正式{s["new_formal_episodes"]}/1200。没有新S或训练环境采集。', '',
 '先完整读PROTOCOL.md、IMPLEMENTATION_NOTES.md、RUN_STATUS.json、ACTIVE_ROOT.json、LOCAL_BUILD_REVIEW.json和最新EVIDENCE/LOCAL_AUDIT。C每seedTask6=100轮、Task7–9各200轮；GOAL142200、OBJECT195300生产调用，逐阶段匹配已接受S的双候选成本；自身4/29隔离=91，C每seed12准入autograd和48方向autograd另记。延长原线性调度终点为实际C更新数，warmup500不变；实际SCHEDULE/COPY_EQ/初始化/Task6对已有S全部tensor需通过。不能套未来stage准入。', '',
 '用户最新顺序：GOAL41/42/43训练→GOAL41/42/43评估→OBJECT41/42/43训练→OBJECT41/42/43评估。正式初态0–19、600步5zero原Normal噪声，不能补采重评或测试选优。S额外240历史采集与历史3914工程验证/中断UNKNOWN保持原账；等optimizer不等FLOPs/墙钟/交互。', '',
 '运输使用旧目录20261027_support_gate_forks现有.session_control worker，先读status并核进程，不重启活跃worker或输出凭据。本工程目录engineering_suite_order_20261030_v1/check_readonly.py复制旧目录后queue，按result真实目录复制证据回来；engineering_suite_order_20261030_v1/audit_progress.py NEW PREVIOUS→该工程record_progress.py NEW INTERVAL。失败先核冻结hash、进程/退出、完整恢复与原失败和已耗成本，不盲恢复或覆盖。', '',
 '仅工程目录engineering_suite_order_20261030_v1/controller.exit0且无自身进程后执行该工程目录final_readonly.py→audit_final_local.py FINAL→根compare_primary_s.py FINAL。原root/controller.exit保持不存在。外置final包装目前仅STATIC_ONLY，须实际24CP、6完整恢复、18teacher缓存CPU重放、1200初态/noise及预算验收；与已有六S的实际1200正式逐项配对，解释两套件三seed、新旧任务/SF/FS、全部负单元及成本边界。然后完成FINAL_REPORT、接受状态和发布。不因负结果追加实验。包装错误保留证据后外置版本化修复，不重训重评。', '',
 f'下一{interval}分钟复查，初期核首个扩展阶段准入/200轮scheduler，随后依据当前整seed或全部评估剩余时间调整，禁止固定高频和按时钟推断完成。健康进展静默，只完成/新失败/需行动通知。', '',
 '每轮按根AGENTS经publish.py审查轻量code/report/hash/scalars/outcomes，串行nonforce推MLR并核remoteSHA/localreceipt；禁CP/model/state/cache/RNG/凭据。六C实际独立验收、已有S比较解释和发布后才删除本新监控；不恢复旧s-goal-seed或s-object-seed。']
(R/'MONITOR_HANDOFF.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
(R/'REPORT.md').write_text('\n'.join(lines[:14])+ '\n\n整体PARTIAL，科学最终验收和C/S比较待全部实际结束。详见PROTOCOL及MONITOR_HANDOFF。\n',encoding='utf8')
print(json.dumps({k:s[k] for k in ['status','observation','epochs','production_physical_calls','own_isolated_calls','new_formal_episodes','phase','next_check_minutes']},ensure_ascii=False))
