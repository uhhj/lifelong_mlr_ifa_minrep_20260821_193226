"""Persist an actual Q snapshot; publishing PARTIAL is not acceptance."""
from pathlib import Path
import json,sys,hashlib
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);interval=int(sys.argv[2])
js=lambda p:json.loads(p.read_text(encoding='utf8'))
def put(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
a=js(D/'LOCAL_AUDIT.json');v=js(D/'EVIDENCE.json');active=js(R/'ACTIVE_ROOT.json')
assert a['status']=='PASS_PROGRESS_NOT_FINAL' and a['contract']==active['contract']
assert hashlib.sha256((D/'EVIDENCE.json').read_bytes()).hexdigest()==a['snapshot_sha256']
status=dict(a,status='PARTIAL_TRAINING_OR_EVALUATION',root=active['root'],controller=active['controller'],latest_evidence=D.name,next_check_minutes=interval,final_CPU_acceptance='PENDING',scientific_conclusion='PENDING_FORMAL_CLOSED_LOOP',primary_reference='Six existing accepted latest full S; no rerun')
put(R/'RUN_STATUS.json',status);active.update(observed=v['observed'],latest_evidence=D.name,status=status['status']);put(R/'ACTIVE_ROOT.json',active)
lines=['# 完整Q对最新S：GOAL和OBJECT各三seed','',f"实际观察 {a['observation']}；{D.name}/LOCAL_AUDIT={a['status']}。",'',f"唯一活动root `{active['root']}`；controller {active['controller']}；合同 `{active['contract']}`；{active['frozen_files']}冻结项；唯一部署证据 {active['unique_dispatch_evidence']}。禁止重复部署或启动任何controller/prepare/train入口。",'', '|case|完整轮/400|接受=生产物理调用|采集/40|正式/200|','|---|---:|---:|---:|---:|']
for case,c in a['counts'].items():
 lines.append(f"|{case}|{c['epochs']}|{c['accepted_updates']}|{c['training_collection_completed_episodes']}|{c['formal_episodes']}|")
lines+=['',f"总{a['epochs']}/2400轮、{a['accepted_updates']}/544800生产、{a['own_isolated_optimizer_calls']}/546隔离、{a['own_extra_autograd_calls']}/246准入autograd、{a['training_collection_completed_episodes']}/240采集、{a['formal_episodes']}/1200正式。方向诊断最多216autograd另计。",'',f'下一{interval}分钟复查，随后按整seed或整套件正式评估实际余时调整，不固定高频。尚未开始阶段不能套准入或速度。', '', '先完整读取PROTOCOL.md、IMPLEMENTATION_NOTES.md、RUN_STATUS.json、ACTIVE_ROOT.json、FINAL_WRAPPER_REVIEW.json及最新实际EVIDENCE/LOCAL_AUDIT。', '', '用户选择每任务100轮，Q保留逐任务支持度保持、关闭生产C/Q门控；各自成功缓存，原4槽分母。GOAL三seed训练→GOAL三seed评估→OBJECT三seed训练→OBJECT三seed评估。已有S1200回合只读配对，不重训不重评，C和旧机制不恢复。', '', 'v1在首次准入/更新/环境前检查属性错误退出，真实零成本审计与失败均保留；只运行v2，新目录修复仅检查表达式，不改scheduler或科学参数。禁止重跑v1，禁止重跑已完成epoch。', '', '使用本目录check/audit/record，运输沿用旧机制目录现有worker，先读status核PID。按实际result目录复制，不猜秒数。不重启活跃worker或输出凭据。故障先审计完整状态/已耗成本，禁止盲恢复部分轮或环境attempt。', '', '自然v2 controller.exit0且无自身进程后，才final_readonly.py实际CPU24CP/6恢复/24缓存teacher/1200初态noise，再audit_final_local.py、compare_primary_s.py。外置包装仅静态不能套PASS。必须完成FINAL_REPORT解释新旧任务/SF/FS/全部负单元与成本边界，接受和发布后才删除新Q监控。', '', '每次审计按根AGENTS经publish.py审核code/report/hash/scalars/outcomes，串行nonforce推MLR核remoteSHA/localreceipt；禁CP/model/state/cache/RNG/凭据/未审bundle。健康静默，仅完成/新失败/需行动通知。']
lines+=['','数据泄露边界见leakage_audit_20261031/REPORT.md：直接新增缓存/正式混入未发现，开发集重复使用与预训练来源未闭合，不得写成全局无泄露或独立确认。最终报告须保留该限制，不追加实验。']
text='\n'.join(lines)+'\n';(R/'MONITOR_HANDOFF.md').write_text(text,encoding='utf8');(R/'REPORT.md').write_text(text,encoding='utf8')
print(json.dumps(dict(status=status['status'],observation=a['observation'],epochs=a['epochs'],next_check_minutes=interval)))
