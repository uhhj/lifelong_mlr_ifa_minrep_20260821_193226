# GOAL Random / Coreset 三seed监控交接

用户明确授权LIBERO-GOAL Random MLR+IFA与Coreset MLR+IFA，seeds42/43/44，所有新产物允许系统盘。先完整读PROTOCOL.md、spec.yaml、DERIVATION.json、RUN_STATUS.json与最新恢复证据。源码基点a721807f2f7fa7f9833dcc85c592d78e9accd00b，沿用已验收OBJECT共同实现补全，GOAL gamma=.7。旧state_v2是有效历史结果但与本轮RNG/IFA/导出实现不完全等价，不当配对Random；本轮三seed均两组新跑。旧实验只读，无Event/ZR/其他新模块。

SSH已有授权root@jq1.9gpu.com:11920，密码只内存/getpass，禁止文件日志。本地tmp/ssh_bridge.py可连接。Q=/root/mlr_ifa/experiment_workspaces/Coreset_paper_GOAL_three_seed_20261017，W=Q/seed{s}，S=W/source，O=/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed{s}_20261017。前置Qprev=/root/mlr_ifa/experiment_workspaces/ZR_coreset_seeds40_41_20261017；admission.py已排队等其queue.exit0、无前置残留与GPU空闲，不能中断前置或重复启动GOAL。前置失败则BLOCKED，先留证修复前置，不绕过保护。

Q/supervise_queue.sh依次执行run_queue.sh：wait_predecessor→seed42 configure/suite_audit/4调用GPU预检/supervise→43同流程→44同流程→aggregate；wrapper在queue退出后运行final_queue_audit.py并记录wrapper.exit。18软件测试每seed已安排启动前执行，队列只验证exit0不重跑；真实GPU预检仅在GPU释放后每seed一次，最多4optimizer调用/1单零动作接口/0策略rollout。CPU资源与配置审计不是GPU预检通过。全部三入口布尔stateTrue、615完整官方权重/12状态参数及GMM语义要通过。

训练每arm4任务6–9×100完整epoch/batch10，从官方GOAL权重独立空buffer，同seed共享新stateTrue特征，跨任务继承网络。BC+.1平均IFA/gamma.7，warmup500逐step线性、原concat/shuffle/sampler0、0.5ceil存储。Random与Coreset唯一组间变量入库成员；visual/action距离/校准42/参考256/标定4096/FP64等冻结不调。数据实际4759窗口[997,889,904,1969]，配额[499,445,452,985]，各epoch batch[100,139,185,337]，每arm76100step。总2400epoch456600step24ep99CP1200episodes120vectorattempt。不能套OBJECT105500。

同seed初始化模型/RNG及Task6首次存储前实际模型tensor完全一致。每seed旧与新输入、source/config/modal/feature/buffer/checkpoint合同一致。评估原协议初态0–19、20/cell、10lane/env[s,s+1]、5零动作/600步、原argmax分量Normal采样，global动作RNG每stage重置，非逐轨迹独立，不增加680/9000或重复次数。正常失败不重试。

运行/data/mlr_ifa/env/bin/python Q/monitor_snapshot.py保存真实snapshot，看Q/queue.pid/exit、wrapper.exit、commands，各O/RUN_STATUS/pipeline/supervisor/epochs/models/storage/evaluation/trials和attempts、源/config/modal哈希、GPU/磁盘/log时间。正常运行保存轻量snapshot和按实际export/train/eval速率ETA即结束，不守候。初始待前置期间GOAL训练与闭环均0，不报成已训练。ORCHESTRATION_SHA256.json冻结root控制代码及逐seedSOURCE_SHA256、软件测试证据；不得改代码重贴标签。

工程恢复先确认进程已停、合同输入一致并保存原始失败，仅从本seed/arm完整epoch恢复模型optimizer/scheduler/grads/modes/全部Python NumPy CPU CUDA RNG/cursor；入库按预提交清单幂等补齐，不能重抽/重复入库/计步。部分评估拒绝盲重启，保留全部attempt与global RNG顺序，无法保持则BLOCKED。不能盲重启queue/整个pipeline，不能覆盖exit/log。只读汇总工程错可留证外置修复，不重跑环境。原准入前语法草稿、首个shell python命令不存在、Windows manifest分隔符问题均是0正式更新/0rollout的准备期工程错误；原证据保留，不是正式实验失败。

完成：各seed8CP实际加载/400episode/40attempt/800epoch152200step、Task6tensor、所有配额/载荷/原输入、矩阵与精确指标审计PASS，pipeline/supervisor0；全部queue.exit0和wrapper.exit0、无残留、FINAL_QUEUE_ACCEPTANCE PASS。训练完成不算COMPLETED。package_evidence.py打包轻量文件，不含weights/latent/demos/buffer payload；下载并按PACKAGE_MANIFEST逐hash验证，本地独立从1200条复算三seed全部矩阵、精确AUC/FWT/NBT、均值样本SD(ddof1)与C−R，检查120attempt，返回命令与退出码/成本/所有任务/限制。

披露n3探索性、3090对论文A100、pretrain state unknown、调度/ep99补全、global动作RNG未隔离与IFA间接作用；旧GOAL不用于配对。负结果正常验收。每2小时检查，近结束可30分钟不新建重复任务；有实质进展/阶段转换/恢复/失败/完成/需用户行动才通知。完整交付后删除GOAL自动化，不自动扩展seed/套件，不push，不上传下载权重数据，不申请新付费资源。
