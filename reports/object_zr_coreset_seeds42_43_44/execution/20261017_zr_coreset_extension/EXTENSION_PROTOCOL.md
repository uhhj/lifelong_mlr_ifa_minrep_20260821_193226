# 已授权的seed43/44零响应残差扩展

报告补充（用户2026-10-17最新要求）：seed43主比较改用第二次完整重跑repeat1的Coreset＋IFA；首次保留辅助表。此变更是事后报告口径调整，详见reporting_baseline_revision/REPORT.md。下述训练来源/缓存/IFA/初始化绑定与冻结合同不变，原验收及自动汇总文件不覆盖；最终外置只读复算同时提供两种口径。seed42、44及历史五seed报告不变。

用户在seed42完成后明确要求继续seed43和44，保持本轮Coreset＋IFA＋指令零响应残差方案不变，复用各自首次完成Coreset＋IFA基线，不重训/重评基线。该授权覆盖原seed42文档的单seed范围限制。其他科学规则原样继承。

每seed新增一组：官方六任务权重、独立空buffer、任务6→9、各100完整epoch、batch10、105500更新、4个ep99最终模型，原协议200闭环/20向量组；合计新增800epoch/211000更新/8最终模型/400闭环。预检每seed最多4次隔离optimizer调用（含一次同更新恢复重演，有效3步），0环境。已完成seed42只读，不重复原型和seed42预检。

训练/数据集打乱/评估全局RNG使用对应43/44，环境seed为[s,s+1]。原DistributedSampler默认seed0、Coreset标定seed42及残差独立局部初始化42+task保持冻结，不把所有42机械替换成实验seed。正式模型、损失BC+0.1meanIFA、gamma0.3、warmup500逐step线性、精度、门向量、rank8、旧分支冻结保留z梯度、所有通道执行均不变。附加模型参数量不变。

复用对应seed的已导出state=True空间缓存（每seed5921窗口）和对应父Coreset入库清单；逐成员/顺序/名字/载荷/配额一致。引用官方OBJECT权重和同一原型门，仅复用门及结构实现，不加载原型训练UV。新旧特征/模态合同分别保持原字节。初始化公共tensor与RNG必须等于对应父INITIAL_STATE。IFA对与reference必须等于对应父；seed42恢复仅针对整数/字符串元数据键的修复在新源码冻结前落实，不改变GPU参考tensor字典。

模型和BC/IFA/评估内核继承已验收seed42 zrg_exp；父libero/coreset基础源码按对应seed原合同逐文件核验。仅参数化seed/路径/基线校验及元数据键修复。规范指令、门FP64、贡献计算、所有数值阈值、诊断频次和全量旧buffer扫描不变；违反即BLOCKED，不调阈值。

每seed完整闭环后相对于自身主基线计算AUC、FWT、有符号NBT、OldFinal、FinalAll，主基线值从通过哈希核验的该seed200条原记录确定，不能沿用seed42数值。四项筛选门槛是相对各自父组，三seed最终汇总报告每seed及均值、样本SD(ddof1)和配对差，不按结果删除seed或启动更多组。原seed42 spec中的数值主基线表仅为历史参考，以本扩展PARENT_BINDING.json为各seed正式比较值。

用户另明确授权新checkpoint、恢复文件、新buffer与日志放系统盘/root/mlr_ifa专用目录；/data旧父基线/缓存/示范/权重均只读。数据盘现约0.95GiB，系统盘约32.7GiB；不得删除有效实验资产。

顺序seed43再seed44，后台队列只运行未启动步骤，失败即停并保留日志/退出。各seed有独立代码/合同/输入hash/输出。恢复仅从本seed完整epoch状态恢复模型optimizer/scheduler/grads/modes/allRNG/cursor、原子存储事务，不重复入库或计步。部分闭环不得盲重跑，原全局动作RNG未隔离，正常失败不重试。需要改变协议则BLOCKED。

验收每seed400epoch/105500步/4严格加载CP/200条记录/20attempts/44结构检查及buffer、IFA、source、输入hash/过程退出；队列完成后汇总三seed。seed42所有旧结果不变。下载轻量证据，不下载权重和载荷，不push，不启动新付费资源。监控沿用每两小时、近结束可缩短，完成删除定时任务。扩展是在看到seed42结果后授权，必须披露探索性、原全局动作RNG、参数量变化、官方pretrain state未知、父调度补全及共享遗忘等限制。
