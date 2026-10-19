# GOAL seed41 三组交接

先读 PROTOCOL.md、ACTUAL_SPEC.yaml、SOURCE_DIFF.json、RUN_STATUS.json、CURRENT_SNAPSHOT.json、DEPLOY_COMMAND.json及恢复补充。用户已明确授权补齐seed41三组，不复用OBJECT41。此次仅新增GOAL41，不扩展其他seed。旧42–44结果只读保留。

本地根 execution/20261019_ra_coreset_goal_seed41；远端Q=/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed41_three_arm_20261019。B=/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed41_20261019；R=/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed41_20261019。Q/baseline/source与Q/ra/source分开；GOAL系统盘授权已涵盖本轮。SSH沿用本对话授权，密码仅getpass内存输入，不记录。

检查运行本地 python execution/20261019_ra_coreset_goal_seed41/check_progress.py；真实远程命令 /data/mlr_ifa/env/bin/python Q/monitor_snapshot.py。检查queue/wrapper.exit及commands.jsonl、各阶段exit/log、GPU/磁盘/实际进程、epochs/recovery/storage/trials/attempt和冻结合同。进程正常则留存快照、按实测估时，结束；不守候，不重复启动。

队列依次baseline软件18、configure、suite audit、GPU preflight(4optimizer+1零动作接口0策略rollout)、supervise(导出、Random/Coreset各训练评估、8CP加载验收)；随后RA admission(软件8、真实身份绑定、全离线选样/删除代价审计、freeze、4optimizer恢复0环境)、supervise(训练/评估/audit/4CP加载/退出验收)、aggregate、queue退出、final_queue_audit、wrapper退出。所有预算与模式见协议，不重复已通过预检。

源码核心与前轮字节一致；seed41新增baseline common/configure/env/pipeline/summarize标签，RA只新增common/prepare准入参数。原始source.patch和SOURCE_DIFF明确记录。ORCHESTRATION_SHA256使用绝对路径，兼容两份verify；不重贴合同。新共享feature/modal由新导出真实生成，不复制42特征改标签。RA只绑定本次完成的seed41父基线。共享Task6实际tensor及初始化RNG需验收。

恢复：先确认相关进程停、原失败证据完整及合同/输入一致。正式训练仅完整epoch checkpoint恢复model/optimizer/scheduler/grads/modes/全部RNG/cursor，预提交storage名单幂等，不重复计步/入库。部分global RNG评估拒绝盲重启，正常失败不重试；技术恢复若无法保证原全局RNG/环境执行顺序和固定预算则BLOCKED。不能盲重启整条queue、覆盖原exit/log或修改冻结科学代码；只读汇总故障可外置修复留证。

基线软件/configure/audit预检exit在Q，formal pipeline/supervisor exit在B/R。queue与wrapper均0、FINAL_QUEUE_ACCEPTANCE PASS、600records/60vector/1200epoch228300step/12CP严格加载凭证、全部hash/Task6/配额/锚点/实际成员/尝试及成本成立才最终COMPLETED。本地须下载轻量manifest包验证后独立按逐episode复算三组所有任务、精确AUC/FWT/NBT/OldFinal/FinalAll、RA-C/RA-R。不能把启动或训练完成冒充完整验收。

Q/ra/package_light.py可用于两根输出，运行前分别设CORESET_OUT=B或R（其它环境不需调用模型）；两次命令隔开避免同秒目录冲突。仅下载其打印目录内metadata及PACKAGE_MANIFEST并逐文件验hash。不下载checkpoint/RNG二进制/latent/buffer/原actions；bindings仅sanitized binding_metadata。额外收集Q/AGGREGATE.json/FINAL_QUEUE_ACCEPTANCE/queue与wrapper退出及新快照。原raw binding含last_action，严禁下载发布。

自动化goal-seed41，每2小时；近结束可调30分钟同ID不新建。全部验收、报告与push成功后删除。不按性能改变三组固定预算，不自动新增实验。负结果正常交付。每次检查按AGENTS用publish_mlr_branch推送本轮相关代码报告轻量证据至MLR，审查敏感信息/类型体积/hash，精确stage，非force，核对远端SHA存receipt；无变化不空提交。

初始部署命令 /data/mlr_ifa/env/bin/python /tmp/ra_goal41_install.py，退出0；launcher PID9005仅用于启动证据，以实际进程为准。最初本地build_sources.py因Windows GBK读中文失败exit1，已在PREPARATION_REPAIR.json披露，发生于未部署、0optimizer/0环境；UTF8草稿修复后退出0，不是科学运行恢复。

限制：看到42–44结果后新增seed41，属探索性；同stage global动作RNG未逐轨迹隔离、原文件排序会改变实际batch、IFA随真实记忆变化、官方pretrain state元数据unknown、调度补全保留。约6小时为先前同GPU吞吐推算，需按本轮实测更新，不承诺准点。
