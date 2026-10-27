# C 完整序列有限扩展：GOAL42/43、OBJECT41/42/43

用户于2026-10-27授权补齐C方案GOAL剩余seed42/43及OBJECT seed41/42/43，实验协议保持。既有GOAL41 C独立验收结果只读保留，不重训。本次唯一5个run，各自本suite、本seed官方初始化，Task6→7→8→9每阶段100完整epoch、固定ep99；先五组训练完，再按同顺序正式评估。不会增加S训练采集、成功筛选或保持缓存。

## 不变的科学协议

直接派生已验收v3工程02的C路径：原RA成员、人口、顺序、载荷、ALIGN下一步标签和有效mask、BC+0.1IFA、时序modality0读出、5分量7D GMM、FP32/batch10/sampler0、AdamW 1e-4/wd0.1、500 warmup、clip100、IFA gamma0.7不变。各seed父manifest和官方初始化独立核验，不能复制GOAL标签到OBJECT。

旧合法窗口的辅助项为0.1×mean(c_exec+0.1×margin_route)，c含归一化均值误差和方差，原三组尺度按全部合法旧训练动作RMS、floor0.1冻结。路由目标为argmin专家cost，完全并列按原C首编号；margin0.1。没有教师预算、V2混合评分、方向锚、固定行为保持、S采集或额外网络。为严格复用原执行流，保留原C的滚动教师缓存、tau/w准备及诊断，但C优化目标不使用教师值。学生辅助eval保留梯度，原模式和全RNG恢复；不改BC dropout流。

GOAL每seed400epoch76100主更新；OBJECT每seed400epoch105500主更新。合计2000epoch468700主更新，0训练环境采集，1000正式回合。各阶段重建原优化器scheduler、组内随机流连续；不选测试最优CP，不减预算，不调参，不追加seed或消融。

每run自己的GPU准入：Task6 4隔离更新(保存/加载下一步重放及原ALIGN对照)，Task7–9每阶段5次(另含状态化梯度测量开关和原C更新精确对照)，共19/run、95次。正式梯度诊断每run48 autograd、准入12，合计240/60，独立计费核算；0环境。每阶段检查长度1–8与部署argmax Normal语义。官方初始化及非CUDA RNG精确，当前卡原ALIGN更新与完整恢复严格；历史跨卡最终tensor仅记录，不因差异重试科学实验。

## 评估、验收及解释边界

每run完整三角矩阵10cell，每cell初态0–19，10环境×两组，600步和5zero，部署argmax分量Normal逐步重新观察。逐轨迹噪声沿原种子化协议20261024+seed*1000+task*100+index，env_seed=seed+group；每run200回合20vector，总1000/100/50。同suite/seed与既有S可核初态、noiseSHA后配对，但S多了训练交互和保持项，不称等交互或独立因果。旧原MLR/RA/ALIGN依既有历史口径作为外部参考。开发seed和旧初态不冒称新确认集，不以结果决定重试。

所有run和controller自然exit0、无自身进程后才运行外置CPU最终验收：20 strict完整CP、5完整最终恢复及history/全部RNG/grads/modes/requires、15滚动教师源CP/cache和身份顺序、原RA成员及全部mask、1000初态实际文件字节与噪声、唯一attempt/事件/分片全集、旧合法窗口分量cost和路由独立复算。只输出轻量误差/logit/hash，不导出action/state/feature/模型/RNG。报告GOAL和OBJECT三seed矩阵/AUC/FWT/NBT低好/OldFinal/FinalAll、全部负单元及C对S等已完成方法的配对SF/FS、离线闭环不一致、计算与交互成本。GOAL41沿用已验收C，跨卡历史差异记录。无新增探索成功门或扩展自动授权。

## 工程与监控

新root CCBP_C_GOAL42_43_OBJECT41_42_43_20261027_v1，独立输出全部/data，保守新增峰值6GiB含原子保存；按唯一稳定inode抵扣新增，system普通留1GiB/data root留2GiB。旧完成实验不恢复，旧最终/原始/活跃资产不删，无新增付费。失败先确认停止、冻结hash、完整epoch恢复和原失败证据，再版本化工程修；不盲重启部分正式评估、不重复完成epoch或正常负结果。

按当前整run训练结束或全部评估结束估下一次同一监控，不为内部Task切换短轮询；使用实际吞吐，保存/准备/诊断另计；无自身吞吐时标注可比参考。健康无变化静默。每轮检查→本地审计→保存→经本目录publish.py调用publish_mlr_branch精确审核轻量文件，非force推MLR并核remoteSHA/localreceipt。全部独立验收、分析及发布完成才删除监控。凭据仅内存或用户授权Windows Credential Manager，禁止明文参数/日志/文件；.session_control不发布。
