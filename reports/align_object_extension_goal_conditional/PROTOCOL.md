# ALIGN OBJECT 42/43 → 条件 GOAL 41/42/43，统一协议

用户2026-10-22明确授权：启动ALIGN OBJECT seed42、43；与已完成seed41合并，若OBJECT三seed等权平均AUC高于同seed原RA平均AUC，则自动启动GOAL seed41、42、43。新授权覆盖此前不扩seed/GOAL的限制，仅限本有限队列。TAC、EFTR、COVR及LIBERO50仍保持完成或用户取消，不恢复。

唯一门槛：三个完整OBJECT ALIGN结果均工程验收PASS后，用原始200episode矩阵精确Fraction重算，mean(AUC_ALIGN[41,42,43]) > mean(AUC_RA[41,42,43])。不增加2/3seed条件，不以NBT/FWT/OldFinal作门，不四舍五入判定。等于或低于则停止；工程失败属于BLOCKED，不能当性能失败跳分支。已完成ALIGN41原包/服务器输出/receipt只读复用，不重训重评。该seed已被看到，故后续为探索性条件实验，不宣称事前独立验证或显著性。

顺序OBJECT42→OBJECT43→精确门槛→GOAL41→GOAL42→GOAL43；GPU串行、有限入口、无自动科学重试。门槛经服务器独立原始记录复算后可自动继续，最终交付仍需轻量下载和本地独立验收，不将服务器PASS冒充本地验收。

统一算法：原八步(8,4,512)特征、Temporal Transformer、原五模式七维部署GMM、FP32、batch10；主BC目标a[u+1]，合法mask与已验收ALIGN41一致：时间身份唯一、八个真实未填充帧、u+1<len(actions)。原DataLoader人口/路径排序concatshuffle/DistributedSampler seed0、所有主步和IFA保留；无效BC仅屏蔽对应NLL并按有效数均值，全无效时可微零BC，IFA照常。无效果回归、对比、辅助头、候选更新或旧任务新增保护。物理效果图和物体专用状态解析不用于ALIGN；新标签程序只核真实8x7动作窗口及观测时序、导出同索引下一动作，不能因GOAL缺少物体效果支持而阻塞ALIGN。

各suite与自己的RA基线对齐，不跨suite强行改超参：OBJECT gamma=.3、官方object_multitask_model_ep99、各阶段人口1562/2261/3081/3620，每epoch157/227/309/362步，每seed400epoch105500主步；RA有序配额781/740/780/660。GOAL gamma=.7、原官方goal_multitask_model_ep96，各阶段人口997/1388/1848/3365，每epoch100/139/185/337步，每seed400epoch76100主步；RA有序配额499/445/452/985。共享BC+.1meanIFA、原AdamW、warmup500和逐主步线性scheduler、原stateTrue等配置均克隆同seed父RA。标定42、Random硬锚点/原RA最少交换算法及其有序成员逐hash复现；初始化tensor与全部RNG、实际IFA必须同父。因BC目标改变，Task6最终tensor允许与RA不同，记录实际差异。

三seed按各自s控制global训练RNG，env[s,s+1]；每个stage重置原global动作RNG=s，0–19原初态、10lane、600策略步/5zero、argmax分量Normal保持。每seed4ep99严格100键、200正式记录/20完成vector/10cell、全部阶段/监督/管线有效exit0无残留。正式正常失败不重试；部分globalRNG评估不盲重启。

原源只读。新版本复用已验收ALIGN41的model/observe/preflight/pipeline/final_process原文件；只参数化suite/预算、数据标签及工程入口。每新seed先独立零训练全窗口/父RA成员/冻结hash/完整初态输入审计，再自己的5隔离GPU主调用、0辅助0环境，关闭精确RA/合法mask/空mask/完整恢复通过后正式重建。CPU测试不替代GPU准入。OBJECT41旧零训练元数据用于核新BC标签器与旧版本逐样本next_action SHA完全一致。

存储：新输出统一/data专用新路径，复用父只读feature软链接，buffer按原逐字节复制，不改路径排序算法。保留所有旧资产/原失败/终止证据，不下载任何payload，不降精度、batch或预算，不新付费资源。起步与每阶段资源记录/data root f_bfree并留2GiB，system只普通空闲；先核未来五run产物规划和实测余量。工程故障先停进程核合同留证，再版本化恢复；仅完整epoch恢复模型optimizer/scheduler/grads/modes/requires_grad/全部RNG/cursor，存储幂等，不自动盲重启dispatcher。

完成各seed后服务器严格CP/成员载荷/IFA/初始化/逐步mask/原始正式episode/attempt与退出独立验收；三seed等权均值及样本SD(ddof1)、同seed差及所有负结果完整报告。最终本地独立复算相同矩阵与门槛。只读汇总错误可外置修复，不重跑模型或环境。

每轮进度/审计按AGENTS向publish_mlr_branch/MLR精确review/stage、非force推送并核remoteSHA/localreceipt。仅代码报告轻量hash及episode结果允许发布，禁模型/示范/rawactions/states/latent/buffer/RNG二进制及未审查归档。整体PARTIAL直至本轮所选分支全部本地验收发布；30分钟监控健康无变化安静，仅阶段变化/恢复失败/完成/需行动通知。
