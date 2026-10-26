# CCBP-v4 S：GOAL seed42、43固定协议扩展

用户2026-10-26明确授权“用S跑一下GOAL的seed42、43，协议保持一致”。这项授权覆盖seed41原“不自动扩seed”的限制，仅新增42、43的S。两seed均完整执行，不因第一个seed结果提前筛除第二个，不重跑已完成seed41或任何基线。

以已实际验收的seed41 engineering02 S为唯一方法实现：官方GOAL初始化，各自Task6→7→8→9、每阶段100完整epoch、400epoch76100主更新/seed，总800epoch152200更新。FP32、batch10、sampler0、原AdamW/scheduler/clip、BC+0.1IFA、C专家误差+0.1margin路由以及额外0.1保持项不变；每seed使用自身原RA成员、ALIGN标签、mask及人口，不把seed41的成员或旧窗口合法数复制给新seed。实际人数与SHA由预检冻结。

S每任务获得后恰好10次初态20–29训练采集，只保留最终成功轨迹候选；每轨迹前8条13窗口、后2条12，确定性reservoir，最多128/task、512/seed。不补采、不重试失败、不重新分配缺额，空旧任务仍占固定均衡分母。教师为本任务获得时自身ep99实际执行Normal，缓存后永久冻结。实际1–8帧历史，使用已修复native temporal_encode保持路径；每主步4个旧任务均衡锚槽，采样规则/尺度/分组W2/argmax与margin并列规则全同seed41。新任务专家监督不使用在线伪标签。

两seed主训练及采集全部结束后各200正式episode/20vector/10cell，共400正式和80训练采集。每阶段评所有已见任务，各20初态0–19，600动作上限+5zero，argmax分量Normal、每步重新观测，首次success终止。训练采集包含终末Task9，其缓存无后续训练用途，成本照计。

种子按原公式自然替换：训练/初始化control_seed(seed)；训练采集噪声RandomState(20271025+seed*1000+task*100+index)，envseed=seed+100；正式噪声RandomState(20261024+seed*1000+task*100+index)，envseed=seed+group。reservoir为RandomState(90100+seed+task*100+index)。锚槽采样原公式20261025+stage*1000000+step*4+slot不含实验seed，保持不改。合成准入的固定随机种子不改。正式初态与训练初态字节SHA必须不交叠；这些初态已用于历史开发，不声称全新确认集。

每seed每阶段自身GPU准入4+6+6+6=22隔离optimizer调用，两seed44；Task6自身短历史1–8验收8额外autograd/17策略调用；两seed正式144额外梯度autograd、原准入36+短历史16=52另列。关闭保持项与原C完整下一步更新精确、Task6与同卡原ALIGN更新精确、所有恢复字段和状态化测量开关精确。无同seed第二个新arm，因此不伪造新arm间Task6相等；官方初始化及非CUDA RNG严格，历史CUDA RNG/Task6张量相等只记录，沿用用户忽略硬件差异要求。

训练采集独立子进程，父model/optimizer/scheduler/grads/modes/requires/allRNG/step前后精确。每完整epoch完整恢复文件；固定ep99模型，不按测试选checkpoint。失败先停态/合同/hash/完整恢复态/attempt账本留证后版本化修复；无盲目重启训练、采集或部分评估，无科学负结果重试，不改冻结源码。

新root CCBP_V4_S_GOAL42_43_20261026_v1，seed42/seed43独立子目录和合同，顶层单controller严格顺序train42→train43→evaluate42→evaluate43。新产物全/data，总保守4GiB峰值按唯一稳定inode抵扣，system普通留1GiB、data root留2GiB。不付费、不删旧最终/原始/活跃资产、不降预算。

最终自然exit0且无自身进程后，实际8CPU strict CP/2完整恢复/8不可变缓存源教师重放与抽样、80训练及400正式初态真实字节与独立noise重生成、RA身份mask与152200更新全账目审计。仅下载代码/metadata/hash/成本标量/episode outcomes，不下载CP/特征/动作/状态/缓存/RNG。外置最终包装尚未运行前不得宣称PASS。

最终与同seed原MLR+IFA、RA、RA+ALIGN历史结果及已验收S41一起报告完整三角矩阵、AUC/FWT/NBT低好/OldFinal/FinalAll、所有负单元、早期7_6和晚期9_8/9_9。没有C42/43、O42/43新实验，不伪造同seedC门或O-S消融；历史动作噪声不配对，逐初态SF/FS若报告必须标为描述而非随机流受控因果。原seed41对C的未过门结论保留。额外交互和辅助计算另列，不能将相对原基线收益全部归因于成功筛选。

这次是用户指定的有限重复探索，不重设有利阈值，不扫描参数，不自动扩展新seed。无论正负完成两seed后停止。每次检查按整组含采集或全部评估预计结束点调同一监控，健康无变化安静，不内部任务短轮询。每轮经本目录publish.py精确轻量审核后非force推MLR并核remoteSHA/localreceipt；完全验收分析发布后删除监控。
