# ZR-v2 seed43：已完成校准，等待闭环结果

**PARTIAL，证据时间：2026-10-18 15:45:31 CST。不能标记整个实验完成，也不能据此启动其他 seed。**

已实际绑定 seed43 第二次完整重跑 `Seed43_Random_Coreset_repeat_20261016_v1/coreset` 的四阶段模型、缓存、Coreset 清单与逐 episode 记录，四个真实 checkpoint 哈希与父记录一致。父200条记录本地精确复算通过：AUC 151/192、FWT 7/10、NBT −61/360、OldFinal 13/15、FinalAll 63/80。没有重训或追加评估父基线。

| 项目 | 已核实结果 |
|---|---|
| 附件合成软件测试 | 25项，通过；不冒充GPU实验 |
| 真实语言门准入 | FP64双向门；batch1/10/32、全部原缓存语言变体通过 |
| 真实GPU集成 | 恰好4次隔离optimizer调用，0环境步骤；通过 |
| 模型路径 | 零残差分布一致；Complete get_action与latent路径history1/4/8一致 |
| 梯度与恢复 | 缺席分支grad=None且Adam不动；冻结公共/T0不变；恢复更新完全一致 |
| 正式校准 | 40/40 epoch，10,550/10,550步，105,240次真实末端标签曝光 |
| 结构检查 | 16次均通过，包括Task7/8/9合法旧buffer全量检查 |
| 入库 | 父Coreset成员/文件名/载荷/数量完全复用，写入本轮独立buffer |
| 阶段快照 | 四个calibration epoch9快照已保存，父权重只读引用 |
| 退出 | calibrate.exit=0；独立evaluate进程已启动；pipeline尚未结束 |
| 正式闭环 | 此快照0/200，结果尚不可用 |

新增校准的epoch计算时间合计185.90秒；含输入核验、结构诊断、保存和入库，校准进程约4分钟。速度快来自只反向更新小型UV且公共网络/T0冻结，**不是缩减epoch或样本**。训练曝光与逐batch身份均记录。闭环仍需真实视觉/环境执行，历史同口径参考约50–90分钟；新评估尚无完成单元，因此这只是暂估，不是实测完成时间。

额外T0为18,932,224个参数，参数及buffer合计75,735,064字节。batch10、history8、仅时序及策略头的CUDA计时：父路径4.36ms，ZR-v2路径9.29ms，增加4.92ms；此计时不含空间前端和环境，不代表完整控制延迟。四UV合计16,608个可训练标量。实际正式校准峰值CUDA张量分配约216MB，不含CUDA上下文等驱动占用。

一次真实Task7分支更新对Task6均值的直接最大变化为0，修正量变化约6.29e−24；这是结构隔离证据，不能替代闭环收益证据。后续自身任务UV重新校准允许自身输出改变。

## 后续固定判定

后台按 `calibrate → evaluate → audit` 执行；最终需完整200条记录/20向量attempt、四快照严格重载、输入哈希、指标复算和独立进程退出验收通过。工程完成后再用精确Fraction判定：

- AUC > 151/192；
- FWT ≥ 7/10；
- NBT < −61/360；
- OldFinal ≥ 13/15。

用户已授权四项和工程门全部通过后扩展其他seed；本轮按现有五seed范围理解为40、41、42、44，各自另绑定对应Coreset+IFA父资产，不重训基线。未全部通过就报告取舍并停止，不改变门槛。

本轮新增校准计算和固定T0推理开销，不能声称等计算预算优势。repeat1历史GPU RTX4090、本轮RTX3090；原全局动作随机流未逐轨迹隔离；预训练state模态元数据仍unknown。这些限制保留，不据单seed声称显著性或一般因果结论。

## 执行与产物

实际入口：`admission.py`、附件`test_reference.py`、`python -m zr2_exp.integration`、`python -m zr2_exp.cost_and_isolation`、`python -m zr2_exp.freeze`、后台`pipeline.py`。详细进程启动命令见MONITOR_HANDOFF.md，正式子命令与退出码见轻量证据commands.jsonl、calibrate.exit。

服务器源码：`/root/mlr_ifa/experiment_workspaces/ZRv2_OBJECT_seed43_20261018/source`。
服务器结果：`/root/mlr_ifa/experiments/ZRv2_OBJECT_seed43_20261018`。

本地及发布内容只有代码、协议、哈希、预算/样本身份元数据和父episode结果；没有checkpoint、权重、演示、latent、buffer载荷或语言向量数组。`verify_light.py`已核验79份远程轻量文件与完整父200条记录；这只是轻量证据核验，不是新增200条闭环的验收。

初始磁盘估算按可能保存完整模型及缓冲峰值预留，实际实现只保存残差与父模型引用，新增存储较保守估算更小。用户已批准系统盘，本轮继续使用隔离系统盘目录，不移动原资产。
