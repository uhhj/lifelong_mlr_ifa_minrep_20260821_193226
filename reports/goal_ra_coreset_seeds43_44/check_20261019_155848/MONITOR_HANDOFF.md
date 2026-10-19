# RA-Coreset GOAL 43/44 扩展交接

先完整读EXTENSION_PROTOCOL.md、RUN_STATUS.json、SOURCE_DIFF.json、各seed ACTUAL_SPEC.yaml、当前progress与恢复证据。用户2026-10-19明确要求两seed都跑，复用对应已验收Random/Coreset基线；旧42只读。保持42算法/实验协议，不因43指标改变44预算。已有GOAL系统盘授权。

SSH root@jq1.9gpu.com:11920已有授权，凭据只内存/getpass；禁止文件日志。Q=/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_extension_20261019，W=Q/seed{s}，O=/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed{s}_20261019；P=/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed{s}_20261017。环境/data/mlr_ifa/env/bin/python，source W/env.sh。队列wrapper PID2156已启动，先准入43；不重复安装或启动。

用Q/monitor_snapshot.py查真实process/GPU/disk/queue与wrapper退出、各seed准入/训练/评估/audit/pipeline/supervisor退出，epochs/CP/recovery/storage/正式记录attempt及冻结hash。也可本地check_progress.py（getpass）保存轻量证据；--package仅对GPU预检已通过组运行原verify和package_light，验证manifest，不下载权重/latent/buffer/原actions/RNG二进制。当前新workspaces均不可重建，失败必须先读真实日志。所有工程脚本由ORCHESTRATION_SHA256.json锁定并进入各seed合同，不得悄改重新贴标签。

queue顺序：43 admission.sh(8软件、prepare真实身份、offline独立删除代价、freeze、4次隔离GPU更新恢复/0环境)→supervise.sh(train/evaluate/audit/process exit)→44同流程→aggregate_three_seeds.py；wrapper在queue退出0后final_queue_audit.py。通过的准入不重复。原科学train/evaluate/model/selector与42字节一致，只有prepare/freeze/audit/offline中的seed/provenance参数化。身份通过实际8×7末端动作窗口匹配，不跨seed套用42绑定。

每seed400完整epoch/76100step/4ep99CP/200episode/20vector；新增800epoch/152200step/8CP/400条。三seedRA合计1200epoch/228300step/12CP/600条，旧Random/Coreset各600条只读。训练/Random锚点/全局评估RNG用当前s，env[s,s+1]，sampler0/距离标定42不变；stateTrue，q499/445/452/985，锚点249/222/226/492，原KCenter+最少交换，原BC+.1meanIFA/gamma.7/warmup500逐步线性、路径排序concatshuffle。无Event/SC/残差/stable-slot/E1，不扩suite/seed/参数。原0–19初态10lane600步5zero，global动作RNG每stage，argmax分量Normal，不加独立Generator/680/9000。Task6tensor与初始化全RNG须匹配各自父Coreset。

工程恢复授权：确认进程停、合同hash一致，留存原退出/log；只能本seed完整epoch恢复model/optimizer/scheduler/grads/modes/allRNG/cursor。store按已提交清单幂等复制，不重新抽样或重复步。不能盲重启queue/fullpipeline或部分评估；原globalRNG和环境顺序无法保证则BLOCKED。正常失败不重试。只读汇总错误可外置留证修复，不重跑环境。若43工程失败，先恢复完成43，然后仅启动尚未运行44；不得重复prepare/admission等已过步骤。

完成需每seed严格CP加载/训练计数/200records20attempt/真实入库离线ID顺序一致/锚点配额最少交换载荷/输入hash/Task6tensorRNG，全部exit0且无残留。queue.exit0和wrapper.exit0及FINAL_QUEUE_ACCEPTANCE PASS后收集两个seed package_light、三seed汇总/计时/退出凭证。本地独立复算全部矩阵与精确AUC/FWT/NBT/OldFinal/FinalAll、逐seed对Coreset/Random差、三seed均值及样本SD(ddof1)，不只汇报正结果。不下载权重；CP实际加载用服务器AUDIT证据明确披露。

每次检查/实验轮次按AGENTS检查publish_mlr_branch分支和远端，精确stage本轮已审查代码报告轻量证据、非force push、核对SHA、保留receipt。未完成发布PARTIAL+时间。冻结科学服务器源不为发布修改。正常保存进度与实测ETA后结束，不一直守候；2小时检查，临近完成可30分钟，不建重复任务。全部验收和发布后删除本自动化，不启动下一轮。披露看到42后探索性扩展/globalRNG/原batch排序/IFA间接变化/pretrain state unknown/调度补全等边界。

2026-10-19 14:18检查：43已400epoch76100step/4CP/200records20attempt及全部exit0，服务器和本地audit_progress复算PASS。AUC84.6875/FWT83.75/NBT-3.333333/OldFinal85/FinalAll86.25；相对Coreset AUC+6.25pp。44已准入PASS且TRAINING，包内65epoch6500step/0正式记录；Task6最终tensor当时尚未到检查点，记PENDING。两seed冻结hashPASS，queue/wrapper仍运行无失败。预计15:50–16:25结束，仍按2小时检查。证据progress_20261019_141825，不能将43完成写成整体完成。

2026-10-19 15:40检查：43完成验收不变；44已400epoch/76100step、train.exit0，评估120/200条、12/20vector，无技术异常。四任务实际有序入库与offline一致，交换129/102/108/246；Task6tensor同父。两seed冻结hash及轻量包manifest、本地复算PASS。按已完成匹配vector耗时比0.9917，预计16:04–16:17完成，仍为PARTIAL；最新ETA用progress_20261019_154009/EVALUATION_PROGRESS.json，优先于audit_progress内未扣除已评估预算的训练阶段参考ETA。因剩余不到1小时，同一自动化已改为30分钟，更新凭证在该progress/AUTOMATION_UPDATE.json。不得重复启动。

2026-10-19 15:59检查：44训练完成不变，评估180/200条、18/20vector，剩stage9/task9两vector，0技术异常；queue/wrapper仍活动无exit。四任务实际storage/冻结哈希/包manifest和本地复算PASS。最新ETA以progress_20261019_155848/EVALUATION_PROGRESS.json为准，预计16:06–16:12，仍为PARTIAL；自动化已30分钟，无重复启动。
