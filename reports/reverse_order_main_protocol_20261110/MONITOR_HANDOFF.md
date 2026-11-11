# 当前逆序任务交接

先读PROTOCOL、IMPLEMENTATION_NOTES、ACTIVE_ROOT、RUN_STATUS和最新证据。唯一root见SPEC，禁止重复deploy/build/prepare/controller及完整epoch、采集或正式attempt。

最新实际观察2026-11-11T16:26:24.798042+08:00，证据check_readonly_20261111_162623。下一检查2026-11-11T19:30:24.798042+08:00，范围CURRENT_WHOLE_RUN_TRAINING_AND_COLLECTION。transport.py check_readonly.py → audit_progress.py NEW PREVIOUS → record_progress.py NEW → publish.py。最终自然exit0且无自身进程后才final_readonly.py → audit_final_local.py → describe_final.py。所有wrapper错误保留原attempt后外置修复，不重训重评；不自动扩seed调参。

当前评估修订见EVALUATION_PROTOCOL_AMENDMENT.json：MLR与DEBR共同完整DEBR评估；训练case合同保持原样；原顺序历史MLR不重评。凭据仅Windows Credential Manager/内存；不得回显、写参数日志。公开仅代码报告/元数据SHA/标量/outcomes。
