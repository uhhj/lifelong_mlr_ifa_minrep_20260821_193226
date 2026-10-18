# ZR-v2 seed43 运行与后续判定

最新已核实：2026-10-18 15:45:31 CST，四阶段40epoch/10550step/105240曝光已完成，4快照已保存、16结构检查全部PASS、calibrate.exit=0；evaluate PID36038已启动原200条闭环。15:47观察GPU100%、stage6/task6正常，尚未提交完成向量组。不得重训或重复集成。自动化ID为zr-v2-seed43，每2小时检查。外置final_process_audit.py额外核验父200条episode文件SHA与准入绑定相同；未改冻结运行源码。

2026-10-18新任务。先读附件 PROTOCOL.md/spec.yaml、EXECUTION_DECISIONS.md、launch_evidence及最新进度。用户授权有效后继续其他seed，已授权本轮及扩展新产物放系统盘。不得因附件旧“不自动扩展”再请求已经给出的权限。

SSH root@jq1.9gpu.com:11920 已授权，凭据仅内存/getpass，不进入文件或日志。Python /data/mlr_ifa/env/bin/python。

- W=/root/mlr_ifa/experiment_workspaces/ZRv2_OBJECT_seed43_20261018
- O=/root/mlr_ifa/experiments/ZRv2_OBJECT_seed43_20261018
- 父P=/data/mlr_ifa/experiments/Seed43_Random_Coreset_repeat_20261016_v1，coreset组。
- 父源码=/data/mlr_ifa/experiment_workspaces/Seed43_three_arm_repeat_20261016_v1/rc/source，保持只读。

2026-10-18 15:42 CST：真实父来源/四CP/缓存/记忆及全部批次语言门准入PASS；25项合成软件PASS；真实GPU集成4次隔离optimizer、0环境步骤PASS；真实单分支更新对其他任务输出及成本补充PASS（0更新）。不重复这些测试或追加预算。当前pipeline PID35977、calibrate PID35978；已完成Task6 epoch0–2、471步。PID仅历史线索，后续核查实际进程/开始时间。

真实启动：source $W/env.sh；/data/mlr_ifa/env/bin/python -m zr2_exp.freeze；nohup /data/mlr_ifa/env/bin/python -u $W/pipeline.py > $O/pipeline.log 2>&1 < /dev/null &。
pipeline顺序校准→单独进程原闭环→只读audit。正常不守候。运行 `$W/monitor_snapshot.py` 查看真实进程、GPU、磁盘、epochs、records、attempt、exits。不能重复启动。

本轮不是ZR-v1公共共同训练。四个公共repeat1 Ck只读；独立官方T0冻结eval，rank8七维UV校准；FP64双向门，旧UV自身Coreset重新校准，真实任务ID仅约束梯度路径，推理全数值门不路由。每阶段10完整epoch、batch10、当前全数据+已入库旧buffer、sampler0、seed43，40epoch/10550step/105240曝光。阶段新buffer精确复制父成员/顺序/文件名/哈希，不能额外选择。各阶段calibration ep9快照独立。

全部四阶段校准完成后200条原协议闭环、20向量attempt，原0–19初态/10lane/env43,44/600步/5零动作/argmax分量Normal、每阶段global RNG43。无基线重训或新基线评估，无9000方案。构造额外支路保存恢复全局RNG。仅空间编码一次，同一历史送Ck和T0。

冻结合同包含源代码/原父输入/缓存/语言门数组/编排哈希，禁止悄改或重贴合同。若工程堵塞先确认进程退出、留存失败、核对输入，再仅显式恢复本run完整epoch模型/UV/optimizer/scheduler/grads/modes/全部RNG/cursor。epoch9后用预提交父清单幂等入库。保留原始batch尝试和物理重复成本；当前audit对无恢复fresh执行计数严格，发生恢复需外置只读审计核对提交权威，不能删失败记录凑数。部分闭环不能盲重启；正常失败不重试，无法保证原环境/RNG顺序时BLOCKED并报告。

pipeline.exit=0且无残留后执行外置 `$W/final_process_audit.py`；检查4CP重载、40epoch/10550step/105240曝光、16正式结构检查、200records/20attempts、输入hash、独立指标复算。状态AUDIT_PASS_METRIC_NOT_PASSED表示工程完成但门槛未全过；ELIGIBLE_FOR_REVIEW表示四条件通过，用户已另授权自动扩展。

精确Fraction门槛全部满足才扩展：AUC>151/192，FWT>=7/10，NBT<−61/360，OldFinal>=13/15，加工程与退出验收PASS。不能仅AUC改善就扩展，不改门槛。报告所有任务和取舍。

若通过，按已授权五seed研究补40、41、42、44；每seed独立绑定自身Coreset+IFA原父四CP/记忆/同seed数据，只调对应seed及路径，原T0/门/UV机制/预算不改，新的预算40epoch10550step200rollout每seed。先核验每seed完整来源、磁盘；输入缺失BLOCKED，不重训父基线。seed43必须用repeat1，其他seed用对应原父运行。43+task的UV局部初始化如何扩展须在对应spec按实验seed+task这一参考实现规则记录，不作结果驱动调参。

每轮轻量证据/报告/代码发布至本地publish_mlr_branch→远端MLR，遵守AGENTS。不上传下载权重、latent、buffer载荷、语言数组、私密信息。临时模型/数组只留服务器。所有证据必须标时间/PARTIAL或真实验收状态。模型新增T0成本、原全局动作流未隔离、repeat1 RTX4090对当前RTX3090、pretrain模态unknown、额外校准预算都披露。完整交付且不再有授权后续工作时删除对应自动化。
