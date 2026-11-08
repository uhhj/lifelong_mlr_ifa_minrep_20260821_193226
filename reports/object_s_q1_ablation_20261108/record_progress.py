"""Record an actual audited snapshot, preserving all earlier evidence."""
from pathlib import Path
import json,sys,datetime
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);a=json.loads((D/'LOCAL_AUDIT.json').read_text(encoding='utf8'))
assert a['status']=='PASS_PROGRESS_NOT_FINAL'
status=dict(a,status='PARTIAL',latest_evidence=D.name,scope='OBJECT full S effective q=1 only, seeds41/42/43; accepted calibrated S read-only',next_check_minutes=int(sys.argv[2]) if len(sys.argv)>2 else 30)
(R/'RUN_STATUS.json').write_text(json.dumps(status,indent=2,ensure_ascii=False),encoding='utf8')
lines=['# OBJECT完整S固定q=1：当前交接','',f"实际观察：{a['observation']}。进度审计PASS，不是最终接受。",'', '|case|完整轮/400|生产optimizer|采集/40|正式/200|','|---|---:|---:|---:|---:|']
for k,v in a['counts'].items():lines.append(f"|{k}|{v['epochs']}|{v['physical_optimizer_calls']}|{v['training_collection_completed_episodes']}|{v['formal_episodes']}|")
lines+=['',f"最新证据 {D.name}；合同 {a['contract']}。下一检查参考 {status['next_check_minutes']} 分钟，后续按整run/整组正式实际剩余时调整。",'',
'先读PROTOCOL、IMPLEMENTATION_NOTES、ACTIVE_ROOT、RUN_STATUS、PREDEPLOY_REVIEW、SOURCE_REVIEW、ENGINEERING_REUSE、FINAL_WRAPPER_REVIEW及最新EVIDENCE/LOCAL_AUDIT。唯一root S_Q1_OBJECT41_42_43_20261108_v1。禁止重复prepare/freeze/deploy/controller或已完成轮。',
'用户只授权OBJECT三个seed，保持lambda=.1，完整S仅有效q=1，空缓存贡献0，候选保持/门控旧评分/方向诊断一致；raw经验q仍记录。原S只读、不重训重评，不恢复旧机制。自己的缓存与原S不必相同。',
'最多两个训练，独立c10d localhost:0/UUID；全部12采集和3评估共用环境锁。三训练自然结束后600正式。每stage自身GPU准入、实际Q1等于原均匀保持、COPY_EQ/BRANCH_COPY_EQ/CHECK_EQ必须实际通过。未来阶段不可套PASS。',
'仅真实controller.exit0且无自身进程后提交一次final_readonly，按实际返回目录audit_final_local→compare_primary_s。12CP/3恢复/12teacher/600noise/27300门控验收；包装目前STATIC_ONLY，不继承旧scale例外，不因包装失败重训重评。最后解释全三seed新旧任务/SF/FS/负单元/门控/支持度/硬件和开发集边界，发布ACCEPTANCE/FINAL_REPORT。',
'直接transport.py使用既有Credential Manager，不输出凭据、不恢复旧worker请求。请求中断后先查进程/完成文件，不盲重提交。所有失败、成本与前缀保留。每次审计经publish.py发布轻量证据到MLR，禁CP/model/state/cache/RNG/凭据。健康进展不高频打扰用户，仅实质变化/完成/失败/需行动通知。']
(R/'MONITOR_HANDOFF.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
