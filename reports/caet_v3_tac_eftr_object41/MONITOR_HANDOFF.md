# CAET-v3 三组独立对照交接

2026-10-22 14:16:39 CST，PARTIAL。先读 USER_METHOD.txt、PROTOCOL.md、IMPLEMENTATION_NOTES.md、REPORT.md、RUN_STATUS.json、COLLATE01_RECOVERY.md。用户明确确认三组独立方案，只 OBJECT seed41，TAC→EFTR→ALIGN；不复活此前取消任务，不扩 seed。

当前远端 Q=/data/mlr_ifa/experiment_workspaces/CAETv3_TAC_EFTR_ALIGN_OBJECT41_20261022_collate01，controller84841；worker Q/TAC、Q/EFTR、Q/ALIGN，输出 /data/mlr_ifa/experiments/CAETv3_{variant}_OBJECT_seed41_20261022_collate01。当前 TAC train84912。SSH root@jq1.9gpu.com:11920，凭据仅内存/getpass，不入文件/日志。

启动证据 launch_20261022_141317，合同SHA 4e74adbf7b52426b8c3e2b3bc5b7302f80bc0a64ec4550a557c544497b1b5a3f。旧无collate01队列因准入任务ID接口错误退出1，2隔离主调用0正式0环境；progress_20261022_141217 已证实旧进程全停、725hash一致。releases/initial 和旧远端源保留，不能覆盖旧退出或恢复旧dispatcher。新版只修observe的tuple→tensor转换，科学合同另见协议。

最新 progress_20261022_141630：TAC19epoch2983主步、2983头更新、27968效果曝光、0正式；725源/调度hash、活动输入hash、初始化同父和本地完整epoch账目PASS。8隔离GPU调用5主3辅助、0环境已PASS，不重复准入。Task6最近10epoch4.655秒，余约6.3分钟，仅该阶段估时。EFTR/ALIGN尚未准入，不把TAC的PASS套用。

每轮先 python -X utf8 check_progress.py，只读保存新快照；再 audit_progress.py <目录>。核真实进程GPU磁盘、队列/worker退出、冻结hash、逐epoch主步、辅助/候选账目和正式record/attempt。正常运行不重启。一轮检查保存报告并发布后结束，不等待整组训练。

TAC/ALIGN改变BC时间目标，Task6最终张量不要求同原RA；EFTR Task6须精确同RA。三者初始tensor/RNG、所有IFA和原RA成员/载荷保持。TAC效果/对比联合训练，不能套用旧v2预测器留出门或H_aux逻辑。EFTR真实三候选只提交一次，额外调用另计，预算不保证闭环。完整epoch恢复须含辅助头/优化器、全部RNG/modes/grads/requires_grad、cursor、teacher和支持hash。

最终在有效supervisor/pipeline退出0、无残留后：package_final.py VARIANT --manifest-only；resume_light_download.py <包目录>；全部SHA通过后 audit_final.py <包目录> VARIANT。新版最终脚本目前仅静态检查，包装/汇总缺陷外置留证修复，不重跑环境。绑定队列receipt和完整200episode/20attempt/10cell、四ep99严格100键、原成员/IFA、初始化/教师/辅助/候选账目。分别与RA及旧v2比较，三组之间TAC对Align、Align对RA、EFTR对RA解释归因。不能用训练结束或准入测试声称算法成功。

每轮按AGENTS运行本目录 publish.py，精确审查轻量文件、类型大小secret/hash后经publish_mlr_branch非force推MLR并核远端SHA/本地receipt。禁权重、demos、原动作/状态、latent、buffer、RNG二进制、未审查归档。数据root保留2GiB，system仅普通空闲，不删旧资产/新付费。

新监控 caet-v3-tac-eftr-align-seed41，每30分钟，健康无变化安静，只实质阶段变化/恢复失败/完成/需行动通知。三组全部本地验收发布后删除该监控。旧任务保持取消。


## 2026-10-22 16:24 CST 检查

progress_20261022_162140：TAC四阶段全部400epoch、105500主更新完成，train6–9/retain6–9及已完成准备阶段均exit0；controller84841、pipeline84899和evaluate85327正常，未恢复或重复准入。725冻结源/调度和288输入hash、初始化同父、本地逐epoch计步PASS。TAC Task6张量与RA不同符合已冻结BC目标变更，保留实际差异，不能误报工程失败。

逐步原始账目汇总：辅助头105493更新，效果曝光1002100；Task8/9对比锚点曝光17800/20600，额外伙伴前向85800/99100（合计184900）。这是178/206独立当前时刻经100epoch重复曝光，不是38400个独立样本；Task6/7跨任务对比0，效果回归仍执行。上述为账目汇总，身份/伙伴选择的完整独立复算仍纳入最终验收，不能称已证明迁移收益。主epoch累计64.739分钟，额外准备/存储/评估另计。

evaluation_readonly_20261022_162355：130/200正式记录、13完成vector/14已开始、6完整cell、0技术异常；外置audit_evaluation_progress.py使用原formal_audit独立检查逐episode初态hash/index、envseed、600步上限、唯一attempt/成功向量及已完成cell摘要PASS。未计算部分AUC。13批均232.513秒，剩评估线性27.13分钟，含最终包装/审计预计16:55–17:20 CST仅参考；后续批次耗时有变动。仍待4最终CP严格加载、完整200/20及有效退出无残留验收。

EFTR/ALIGN仍排队且尚未GPU准入，不能套用TAC的通过。数据普通13.895GB/root18.766GB、system普通7.979GB，data保留2GiB条件满足。正常训练/评估不重启。下一轮先检查是否自然完成/自动转EFTR，TAC完成后只下载白名单轻量包并独立最终验收；新版包装/最终脚本仍待实证。总体PARTIAL。
