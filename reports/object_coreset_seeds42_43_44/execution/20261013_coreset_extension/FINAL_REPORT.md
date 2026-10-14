# OBJECT Random vs Greedy Coreset：seed42/43/44完整报告

COMPLETED。新增seed43/44均完成，2026-10-14 02:25:37（北京时间）队列及三seed汇总退出0。2026-10-14本次远端哈希/退出核查与本地独立复算PASS。三seed共2400 epoch、633000 optimizer steps、24个ep99最终checkpoint、1200条正式闭环；其中新增两seed1600 epoch、422000步、16个CP、800条记录。定时任务object-coreset-seeds43-44已删除，不启动下一轮。

## 逐seed指标

单位为%，C−R为百分点；NBT按遗忘量定义，越低越好。AUC为任务均衡的阶段平均，不是10个单元直接平均。

| Seed | 方法 | AUC | FWT | NBT |
|---|---|---:|---:|---:|
| 42 | random | 66.145833 | 66.250000 | -0.555556 |
| 42 | coreset | 68.125000 | 65.000000 | -5.833333 |
| 42 | C−R | +1.979167 | -1.250000 | -5.277778 |
| 43 | random | 78.750000 | 82.500000 | 6.666667 |
| 43 | coreset | 94.375000 | 88.750000 | -10.000000 |
| 43 | C−R | +15.625000 | +6.250000 | -16.666667 |
| 44 | random | 84.166667 | 82.500000 | -3.333333 |
| 44 | coreset | 86.250000 | 86.250000 | 0.000000 |
| 44 | C−R | +2.083333 | +3.750000 | +3.333333 |

## 三seed描述性汇总

样本标准差采用ddof=1；总体标准差ddof=0同时保存在LOCAL_INDEPENDENT_ACCEPTANCE.json与THREE_SEED_SUMMARY.json。配对差先在同seed计算，再统计三seed。

| 方法 | AUC均值 ± SD | FWT均值 ± SD | NBT均值 ± SD |
|---|---:|---:|---:|
| random | 76.354167 ± 9.246222 | 77.083333 ± 9.381942 | 0.925926 ± 5.161985 |
| coreset | 82.916667 ± 13.438711 | 80.000000 ± 13.050383 | -5.277778 ± 5.023095 |
| paired_C_minus_R | 6.562500 ± 7.848528 | 2.916667 ± 3.818813 | -6.203704 ± 10.032099 |

三个seed的AUC均为正差，但提升幅度分别只有1.979167、15.625、2.083333个百分点，平均提升主要受seed43贡献影响。seed44的NBT反而增加3.333333个百分点；最终任务6由Random的50%降至Coreset的40%。不能将总体均值改善表述为所有seed、所有任务均改善。

## 全部阶段—任务成功率矩阵

每格为Random / Coreset（%），每个单元20次闭环；—表示未评估未来任务。

| Seed | 阶段 | 任务6 | 任务7 | 任务8 | 任务9 |
|---|---|---:|---:|---:|---:|
| 42 | 6 | 30 / 30 | — | — | — |
| 42 | 7 | 45 / 70 | 95 / 85 | — | — |
| 42 | 8 | 10 / 20 | 100 / 95 | 95 / 100 | — |
| 42 | 9 | 10 / 30 | 100 / 90 | 100 / 100 | 45 / 45 |
| 43 | 6 | 60 / 60 | — | — | — |
| 43 | 7 | 55 / 95 | 95 / 95 | — | — |
| 43 | 8 | 40 / 85 | 75 / 100 | 95 / 100 | — |
| 43 | 9 | 55 / 90 | 85 / 90 | 100 / 100 | 80 / 100 |
| 44 | 6 | 50 / 50 | — | — | — |
| 44 | 7 | 65 / 70 | 95 / 100 | — | — |
| 44 | 8 | 65 / 40 | 95 / 100 | 95 / 100 | — |
| 44 | 9 | 50 / 40 | 85 / 100 | 100 / 100 | 90 / 95 |

## 真实执行、退出码与尝试账目

实际队列入口：

```bash
nohup bash /data/mlr_ifa/experiment_workspaces/Coreset_paper_OBJECT_extension_20261013/run_queue.sh > /data/mlr_ifa/experiment_workspaces/Coreset_paper_OBJECT_extension_20261013/queue.log 2>&1 < /dev/null &
```

队列按43→44运行；每个seed依次configure、validate_extension.py、18项software、GPU preflight、supervise。supervise中的正式命令依次为export→train random→train coreset→evaluate random→evaluate coreset→summarize→acceptance。torchrun端口43为38316、44为38317；每个命令的完整参数及开始/结束时间已保存在final_evidence/remote/queue/commands.log和seed43/commands.log、seed44/commands.log。全部正式阶段、supervisor和queue最终退出0。aggregate_three_seeds.py退出0。

每个新seed14项seed/config/source检查、18软件测试、GPU模态/参数加载/GMM一致性/恢复检查PASS；隔离GPU各4次optimizer调用和一次零动作环境接口步骤，不计入正式训练或策略闭环。三个seed各400正式记录，每seed40个10-lane向量尝试；总120组、1200条，无技术异常或正式重试。软件合成测试不是实验结果。

每个seed两组任务6实际张量一致、初始RNG摘要一致；24个最终checkpoint在各run独立验收时完整加载，每个100个下游参数键。当前远端最终checkpoint哈希与训练完成清单一致。当前冻结源码/config/input/feature/modal核查PASS；本地下载402个轻量工件逐文件hash一致，seed42此前归档中的132个可比较工件逐字节未变。未下载checkpoint。

本地独立命令：

```powershell
python execution/20261013_coreset_extension/final_evidence/verify_local.py
```

退出0；独立复算1200条记录、全部矩阵与精确分数、环境seed顺序、尝试successes与episode对应、同seed同初态配对、800epoch/seed、三seed均值和标准差。仅CPU只读分析，不创建optimizer或执行环境。

## 配额、覆盖与时间成本

三seed六组每任务均保留781/740/780/660个原始窗口；新seed各重新导出5921个state=True窗口，同seed两组共用。存储清单保存原文件SHA256、选择顺序、Coreset距离尺度/参考子集/半径轨迹以及阶段buffer哈希，原payload验证由各run汇总通过。Coreset独立标定/参考seed42固定，训练/Random入库/评估种子为42/43/44，环境seed分别[s,s+1]。

| Seed | Random训练epoch合计秒 | Coreset训练epoch合计秒 | Random评估墙钟 | Coreset评估墙钟 |
|---|---:|---:|---|---|
| 42 | 2799.383 | 3116.623 | 76分18秒 | 78分39秒 |
| 43 | 2627.676 | 2661.399 | 83分02秒 | 58分49秒 |
| 44 | 2705.697 | 2564.896 | 81分11秒 | 58分43秒 |

新增43/44队列从10月13日17:56:55至10月14日02:25:37，墙钟8小时28分42秒，包含配置、预检、导出、两组训练、评估及审计。epoch累计秒不包含全部加载、存储与审计开销，不能用它替代总计算成本。

## 当前服务器情况与解释边界

本次第一次SSH握手失败，随后重连成功；没有重启实验。当前nvidia-smi退出9，报告无法连接驱动，因此旧monitor_snapshot.py在GPU查询时失败。这发生在已记录成功结束之后；没有证据可确定驱动何时或为何不可用。本次只读哈希、进程和本地复算均通过，未将当前驱动状态当作新的GPU测试通过，也未操作驱动或启动资源。此前各run实际GPU验收和正式退出证据保留，当前情况详见FINAL_PROCESS_ACCEPTANCE.json。

扩展43/44是在看到seed42结果后授权，属于探索性扩展，不能包装成预先独立的确认试验；n=3描述性均值和标准差不能单独支持统计显著性。state=True按论文描述启用，官方预训练cfg.policy.use_state缺失仍为unknown/DISCLOSED_GAP。seed42来自官方启动脚本，正文未列出42/43/44具体数值。RTX3090不同于论文A100；warmup500、逐step线性调度和固定epoch99为已披露工程补全。Coreset改变buffer后IFA对可间接变化，因此结果不能独立归因于几何覆盖。旧state=False E0不可作本轮同协议Random替代。

## 产物位置

本地：本目录FINAL_REPORT.md；final_evidence/LOCAL_INDEPENDENT_ACCEPTANCE.json；final_evidence/remote/queue/THREE_SEED_SUMMARY.json；final_evidence/remote/seed42、seed43、seed44保存各阶段日志、记录、attempts、manifest、配置与验收。修改diff位于seed43/source_vs_seed42.patch和seed44/source_vs_seed42.patch；seed42原实现diff仍在../20261013_coreset_object/source.patch。

远端：/data/mlr_ifa/experiments/Coreset_paper_OBJECT_seed{42,43,44}_20261013，权重在各random/models与coreset/models；队列/总汇总位于/data/mlr_ifa/experiment_workspaces/Coreset_paper_OBJECT_extension_20261013。

轻量包SHA256：791b325fbcfabbdb9c7707f8966abf8059a36273fff7ade2f4a24f53ba7dd896。未push、未上传/下载权重、未扩展额外seed或套件。
