**Fixed-kappa confirmation v2：结果产生前冻结的确认协议**

主比较仅为原MLR与同批聚合软约束κ=0.5。新seed固定45/46/47；历史42/43/44只在新seed主分析之后单列。既有独立参考、分任务联合约束、功能子空间和其他κ不进入本轮。

训练固定LIBERO_OBJECT，task6→7→8→9，每任务100 epochs，batch10，num_workers0，原GMM BC、原样本均值与replay逻辑、past_data_rate=0.5。AdamW(lr=1e-4, betas=0.9/0.999, eps=1e-8, weight_decay=0.1)，batch-step linear scheduler，每任务500 optimizer steps warmup，原grad_clip=100及原任务开始时optimizer/scheduler初始化方式不变。使用官方OBJECT共同预训练checkpoint及每seed独立生成、两arm共享的原潜特征缓存。前端保留use_language=True、use_state=False；不添加机器人state token。特征生成仍使用原save_feature.py的种子、数据顺序和增强逻辑。

Soft调用既有fixed_kappa核心：对同一次前向的原混合BC梯度g和同批旧样本参考r，仅在reference有效且g·r<0时写入g−0.5(g·r/||r||²)r。之后执行原clip、AdamW和scheduler。不重构损失、不再追加旧梯度。

新增代码均位于confirmation_experiment；原训练、评估、fixed-kappa及其他历史文件不改写。包装器仅负责记录、保存和完整性等待。启动前冻结依赖源码、配置、官方checkpoint、原始HDF5及缓存哈希。共享历史依赖源码的副本仅作重现与审计，不作为新算法。

**执行顺序与task6闸门**

预检查与smoke只使用历史seed42资产，不用于新方法选择。新seed使用检查覆盖仓库路径、结构化JSON/CSV seed、文本seed设置及git历史标题；发现既往同类选择/调参使用则失败，不替换seed。

正式顺序固定为45→46→47，每个seed先MLR再soft。为在任何task7更新前核对task6，MLR进程完成task6后等待，soft进程再完成task6后等待。比较逐batch BC序列、task6 ep99参数、optimizer/scheduler/RNG哈希。两arm的原生完整test.py评估进程分别先完成第一格task6/ep99的20 trials，在该原生调用返回处等待；两个结果必须一致。

以上检查全部通过才释放MLR的task7→9训练，完成后让其同一评估进程继续原test.py剩余调用；随后释放soft的task7→9训练和同一评估进程。等待不调用模型、不抽随机数、不额外评估、不重置训练或评估RNG。评估任务调用顺序、每格20 trials、GMM动作采样、success判定及ep99均保持原样；每个矩阵单元只记录一次。闸门只因完整性错误失败，绝不因成绩高低停止。

正式训练与评估不可按成绩重跑、换seed、换epoch、增减trials或提前终止。不设best checkpoint。原ep70/75/80/85/90/95及ep99保存规则保留，主评估只用ep99。

**逐步事件：不影响优化决策**

Soft每batch流式写events.jsonl：seed/task/epoch/batch位置、全局与任务内optimizer step前后、lr、新旧样本数和实际权重a/b、task来源组成及stage plan中的记录索引、混合/新/旧BC、原梯度和参考范数、内积、余弦、原生trigger、requested修正范数、实际写入修正范数/比例、clip前后梯度范数和必要计时。原生门只使用原有g·r及reference floor，不读取新增派生量。

n·r=(g·r−b||r||²)/a；||n||²=(||g||²−2b g·r+b²||r||²)/a²；z=1−g·r/(b||r||²)。只在a>0、b>0、参考有效时给出；余弦的范数抵消情况显式标记，不取绝对值掩盖误差。这些是已有FP32梯度统计的代数派生值，不增加backward；不是独立直接测量的n梯度。z≤0为无新旧负冲突，0<z≤1为被回放正项抵消的负冲突，z>1为回放后仍存在的强冲突。

实际写入修正范数通过临时保留原梯度并在原生修正后求差获得；不保存逐步完整梯度。日志与快照的额外耗时单列，带诊断代码的耗时不称为纯算法开销。

**机制快照：固定上限48个，仅soft**

每seed×task7/8×phase最多4个，共3×2×2×4=48个，不补齐缺失槽位，不按loss/成功率挑选：

| phase | 首个trigger的三个独立epoch区间（含左不含右） | 非trigger对照 |
|---|---|---|
| 30–69 | [30,43)、[43,56)、[56,70)各首个trigger | 本phase首个0<z≤1且原生trigger=False的batch |
| 70–99 | [70,80)、[80,90)、[90,100)各首个trigger | 本phase首个0<z≤1且原生trigger=False的batch |

要求从快照边界起同任务至少剩20次更新，以便后续另行开展1/5/20步复核。快照为原生observe之前、尚未forward/zero_grad/update的状态：完整model/AdamW/scheduler及计数、Python/NumPy/torch CPU/CUDA RNG、模块training mode、task/epoch/batch、当前实际batch、原始记录索引、当前epoch sampler序列、stage dataset/buffer有序记录与内容哈希、未来epoch确定性采样定义、config/code hash。保存参数时尚未更新，可能被forward修改的buffer用提前捕获的值恢复到pre-forward边界。禁止快照采集修改主序列；真实GPU parity必须覆盖快照保存及恢复后下一步一致性。

task6仅另外保存完整性哈希，不额外保存整套大状态。机制快照本轮只采集，不运行反事实续训、不反向影响正式实验。

**分析顺序与决策**

Primary confirmation仅使用45/46/47。原定义AUC为各任务列学习后各阶段成功率均值的均值；FWT为对角线均值，NBT保留历史定义。输出每seed配对ΔAUC/ΔFWT/ΔNBT/Δfinal success，以及各任务首次/最终成功率、列AUC、peak-to-final forgetting与worst-old-task final success。之后单列历史42/43/44，最后才给6seed探索性描述。绝不以6seed均值改写新seed结论，不声称小样本显著性。

为使“很小/系统下降/单seed主导”不由结果出来后临时决定，本轮采用保守的操作化门槛（单位均为成功率百分点）：

- 新seed平均ΔAUC≤0，或FWT/final success平均下降≥2.5且至少2/3 seed下降：NOT_SUPPORTED。
- SUPPORTED需平均ΔAUC≥1.0、至少2/3 seed为正、最差seed的ΔAUC≥−2.5、单seed占全部正AUC增益比例≤80%、平均ΔFWT和Δfinal均≥−2.5，且没有旧任务三个seed都下降且平均额外下降≥5。
- 其余INCONCLUSIVE。旧任务一致退化、结果异质或单seed主导不得被NBT或历史混合均值掩盖。

这些是预先冻结的研究决策标准，不是统计显著性阈值；最终报告同时显示全部原始数值，让读者能独立判断。只在SUPPORTED后才允许另行启动预存状态的反事实机制复核。

事件统计按seed/current task/epoch段00–09、10–29、30–69、70–99，报告所有batch、有效reference、trigger数率、z分布与0<z≤1/z>1比例、trigger内z、cosine、范数和修正比例。trigger率不等于全部负冲突率，更不等于实际闭环遗忘率。

**启动条件**

必须完成单元测试、κ0/0.5/1端点、真实GPU事件记录开关loss/gradient/parameters/optimizer/scheduler/RNG逐位一致、真实快照恢复一致性、两arm torchrun smoke、前端逐项一致、seed审计、输入/缓存哈希、空间检查、只使用本轮输出目录和git实现提交。PRECHECK_REPORT.md记录每项证据，任一失败不启动6次正式训练。正式运行另有task6闸门。队列保存PID、STARTED.json、status.json及启动日志。
