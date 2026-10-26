# S：OBJECT seed41、42、43有限完整序列

用户2026-10-26授权用S方法跑OBJECT seed41、42、43。本轮只迁移已验收S方法，不依据GOAL或OBJECT测试结果改损失、权重、选样、模型、预算。各seed独立从本suite官方OBJECT初始化，完整Task6→7→8→9，每阶段100epoch，每seed400epoch105500主更新；总1200epoch316500主更新。原OBJECT人口1562/2261/3081/3620；合法标签和旧合法窗口数以各seed自己的原RA/ALIGN输入实际预检冻结，不能复制GOAL或其他seed人口mask。

方法沿用已验收CCBP-v4 engineering02 S及S GOAL42/43：原RA成员、ALIGN next-action标签与独立mask、BC+0.1IFA、C持续专家cost+0.1margin路由不变；额外0.1保持项、分组W2、固定任务获得时自身教师、每主步4个均衡旧任务锚槽。FP32/batch10/sampler0/原AdamW/scheduler/clip/读出/GMM不改。尺度来自本任务合法训练动作RMS floor0.1。Task6没有旧任务辅助项。部署为argmax分量Normal采样，每步重观测，不能用专家标签在推理筛分量。

每任务获得时进行10回合训练采集，初态20–29，只保留最终成功轨迹候选；前8轨迹各13、后2各12窗口确定性reservoir，最多128/task、512/seed。不补采、不重试失败、不重新分配缺额；空任务仍占固定旧任务均衡分母。教师来自本任务获得时ep99固定CP，不滚动；实际1–8帧历史，使用native temporal_encode。采集在独立子进程，父完整model/optimizer/scheduler/grads/modes/requires/allRNG/step前后必须精确。Task9采集虽然无后续训练用途仍照原协议执行和记账。

按train41→train42→train43→evaluate41→evaluate42→evaluate43串行派发。每seed40训练采集、200正式episode/20vector/10cell；总120训练采集、600正式/60vector/30cell。各stage评所有已见任务，每cell20初态0–19，600动作上限+5zero，首次成功终止。任务采集和正式初态字节SHA不交叠；这些初态曾用于历史开发，不称独立确认集。所有seed均完整执行，不按中途正负结果筛选。

随机数沿原S公式仅代入seed：训练/初始化control_seed(seed)；采集Normal RandomState(20271025+seed*1000+task*100+index)，envseed=seed+100；正式Normal RandomState(20261024+seed*1000+task*100+index)，envseed=seed+group；reservoir RandomState(90100+seed+task*100+index)。锚槽RandomState(20261025+stage*1000000+step*4+slot)不含实验seed，原样保留。

每seed自身GPU准入4+6+6+6=22隔离优化器调用，三seed共66。关保持=原C完整下一步精确，Task6同卡原ALIGN更新精确，真实保存恢复与全部状态/梯度测量开关精确，不能套GOAL准入。各seed1–8短历史额外8autograd/17策略调用，共24/51；正式额外梯度autograd216，准入54+24=78另列。实际硬件只记录；官方初始化及非CUDA RNG严格，历史CUDA RNG/Task6逐位差异记录而不因换卡直接拒绝。

新root CCBP_V4_S_OBJECT41_42_43_20261026_v1，各seed独立合同；全/data新增保守6GiB峰值按稳定唯一inode抵扣，system普通保留1GiB/data root保留2GiB。不额外付费、不删原始/最终/活跃资产、不降预算。只有读历史输入，不重新准备RA、重导出features或重跑旧基线。故障先确认进程停、合同、完整恢复和原失败/attempt再版本化修复；不覆盖冻结源、不盲重启部分正式评估、不重试正常负结果。

全部自然controller.exit0且无自身进程后实际最终验收12CPU strict CP、3完整恢复、12固定缓存源教师重放和抽样、原RA/ALIGN输入与全epoch账目、120训练及600正式初态真实字节/独立噪声SHA、成本标量复算。只下载代码、metadata、hash、标量和outcomes，禁止下载/发布CP、特征、原动作状态、demo、缓存、RNG或凭据。最终外置wrapper静态检查不算PASS，运行错误留证外置修，不重训重评。

报告三seed完整矩阵/AUC/FWT/NBT低好/OldFinal/FinalAll、全部负单元、同seed跨阶段配对SF/FS、离线与闭环解释及成本。与原MLR+IFA、RA、RA+ALIGN历史OBJECT同seed结果对照，但不是同动作噪声或等交互实验；没有C/O OBJECT三seed，不声称成功筛选独立因果或显著。不存在从GOAL复制的事后C门；固定完成三seed后停止，无自动扩seed、调权、消融或阈值扫描。

监控按整seed训练含采集或全部评估预计结束点调同一监控，不内部阶段频繁检查。健康无变化安静。每次检查/审计按根AGENTS经publish.py精确审核轻量类型大小secret/hash，非force推MLR核remoteSHA与localreceipt；完整验收和分析发布后删除监控。凭据仅已授权Windows Credential Manager/内存通道，不明文写入文件参数日志。
