# SC-Coreset GOAL seed42：监控、验收与条件扩展

2026-10-18 用户授权新方案，先seed42。用户已确认扩展门槛：AUC严格高于对应原Coreset且OldFinal不降低，工程全部通过后才补43/44。42门槛为AUC>63/80、OldFinal>=41/60；Random仅参考，不替换门槛。旧ZR-v2已经结束，不恢复；本轮无残差/Event/加权。

先完整读本地PROTOCOL.md、RUN_STATUS.json、UNCHANGED_COMPONENTS.json、launch_evidence/CONTRACT.json、GPU_PREFLIGHT、LOCAL_LAUNCH_ACCEPTANCE与最新恢复证据。SSH root@jq1.9gpu.com:11920已有授权，凭据仅内存/getpass，不保存文件日志。两条本地tmp/ssh_bridge.py会话仅为连接方式，不是运行所需后台服务。

- W=/root/mlr_ifa/experiment_workspaces/SC_Coreset_GOAL_seed42_20261018
- O=/root/mlr_ifa/experiments/SC_Coreset_GOAL_seed42_20261018
- P=/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed42_20261017
- 父源=/root/mlr_ifa/experiment_workspaces/Coreset_paper_GOAL_three_seed_20261017/seed42/source
- source W/env.sh 后，使用/data/mlr_ifa/env/bin/python。

用户先前允许GOAL新产物系统盘，原数据与基线只读。真实准备检查source/input/features/四任务绑定PASS。5个性质测试（含穷举取整与30组独立暴力交换对照）、全4759真实窗口、所有删除半径/配额/最小交换及有序S0与父一致PASS；GPU RTX3090 PyTorch1.13.0+cu116，恰好4隔离原生optimizer及边界恢复PASS，0环境/0策略rollout。不能重复预检。Task9两个padding起点歧义但所需demo/真实target151/length152/末端action唯一，已经验证，不擅自去重。

SC保留原距离/全任务标定/原K-Center，以时间四分位+真实终点和末端夹爪符号的联合组约束，每组floor/ceil，合法配额首先min L1交换M，继而min总取整误差，最后固定ID。先补欠额最远点，再删超额组中删除后全局半径最小点，FP64/1e-12原平局规则；在被移出样本原槽位替换，其他顺序不动。四任务M=70/38/38/39，所需联合构成均满足；Task6正命令54.7094→41.4830%（全体41.4243%）、终点2.8056→8.0160%（全体8.0241%），覆盖半径0.278769→0.337656。只是选择器预检，不是闭环收益。

正式仅sc_coreset一组，从官方GOAL权重空buffer重新seed42开始；复用父stateTrue特征/原modal合同，不重新导出。任务6–9各100完整epoch/batch10，共400epoch/76100step，配额499/445/452/985。原BC+.1meanIFA、gamma.7、warmup500逐步线性、原concat/shuffle/sampler0、原模型/输入/动作全部不变。IFA按本轮实际buffer计算并与父记录比较，不冻结原配对。Task6首次入库前实际模型张量与父Coreset必须相等，初始模型/RNG摘要相等；失败停止，不强行改seed凑一致。

W/pipeline.py后台依次torch.distributed.run -m coreset_exp.train sc_coreset、单独进程coreset_exp.evaluate sc_coreset、sc_exp.audit。四ep99最终CP；原0–19初态每cell20、10lane/env42/43、600策略步、5零动作、每stage globalRNG42、argmax分量Normal。总200正式记录20vectorattempt，无680/9000/新初态/独立轨迹RNG。GPU预检状态丢弃，正式单独进程重置种子。

每次先source W/env.sh运行W/monitor_snapshot.py；看进程、GPU磁盘、日志时间、RUN_STATUS、commands、pipeline/train/evaluate/audit.exit、epochs/recovery/models/storage/正式trials/attempt、冻结源码及输入。防止重复启动。正在正常运行则保存轻量快照及实测速率ETA后结束，不持续守候。

可恢复工程阻塞须先确认进程退出与合同一致，保留原失败exit/log。只从本run完整epoch恢复模型optimizer/scheduler/grads/modes/全部RNG/cursor，使用显式train sc_coreset --resume；原pipeline已启动保护禁止盲重启，另记恢复编排及真实退出。epoch99存储先提交清单再幂等复制，不能重抽/重复入库/计步。不能悄改冻结source/合同或重贴标签。评估部分记录存在时主动拒绝重启；正常失败不重试，不能保证原globalRNG与环境顺序则BLOCKED并报告。只读分析工程故障可外置留证修复，不重跑环境。

pipeline.exit=0且无残留后运行W/final_process_audit.py。必须400epoch/76100step/4完整加载CP/Task6张量与初始化RNG/200records20attempt/输入和真实buffer载荷/SC联合表及M/全矩阵精确AUC FWT NBT OldFinal FinalAll/成本/退出验收全部PASS。训练完不能标COMPLETED。数据构成保证不等于效益，负结果照常验收。

W/package_light.py仅打包允许元数据。bindings原清单含真实末端动作，严禁下载发布原bindings；工具生成binding_metadata副本去除last_action，仅保留真实身份/时间/group/哈希。禁止任何权重、payload、latent、语言数组或archive发布。按manifest下载核验，再独立从全部200episode复算、比较父Coreset及Random全任务，不只报告Task6。

若所有工程与双指标门槛通过，按用户授权补seed43/44：分别使用原GOAL对应父源/原feature和对应对照；独立工作区、空buffer、官方起点、新合同，仅替换seed/路径与基线门槛，不变SC机制/参数/预算/标定42/采样器0。当前runner只seed42，扩展需明确参数化prepare/audit/协议、核验各自输入，不直接复用42绑定，也不重训旧Random/Coreset。若42未通过则报告停止，不追加seed或调参。

报告披露探索性单seed→事后条件扩展、原global动作RNG未逐轨迹隔离、官方pretrain state元数据unknown、调度/ep99补全、IFA随buffer间接变化、覆盖半径取舍与离散表不能保证连续分布。每轮检查/实验后按AGENTS只提交本轮代码报告轻量证据到publish_mlr_branch的MLR，不force push；核验远端SHA并本地留receipt。常规2小时监控、近结束可30分钟不建重复；无变化无可行动问题保持安静。所有授权任务完成后删除当前自动化，不能恢复旧ZR-v2监控。
