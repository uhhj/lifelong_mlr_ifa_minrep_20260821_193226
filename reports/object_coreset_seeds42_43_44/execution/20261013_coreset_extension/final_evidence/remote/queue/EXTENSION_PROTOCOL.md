# 用户授权的 seed43/44 扩展

2026-10-13用户在seed42完整报告之后明确要求“将后面两个seed也一起跑了”。这覆盖seed42协议中的“不扩展43/44”，其他科学设定不变。已有seed42及其他实验均保持只读，不重训seed42，不用旧state=False历史E0替代配对Random。

每个新增seed分别运行Random和Coreset，官方OBJECT权重起步、独立空buffer；任务6—9各100完整epoch、batch10、每组105500步；state=True、平均IFA0.1、gamma0.3、warmup500逐optimizer-step线性调度、原拼接shuffle与ceil(0.5N)存储配额。固定epoch99，每组200正式闭环。新增共1600epoch、422000步、16最终checkpoint、800正式记录。合并seed42后为2400epoch、633000步、24最终checkpoint、1200正式记录。

seed43与44各从对应种子重新导出新特征，同seed两组共享字节一致的新导出文件。训练全局RNG、导出RNG、数据集构建、Random入库、评估全局RNG按对应seed；评估环境两组seed分别为[s,s+1]。不改变原DistributedSampler的默认seed=0及set_epoch方式。Coreset距离标定/参考子集固定独立seed42，属于既有冻结选择器定义，不改成新超参数。模态、模型、损失、距离、参考数、校准对、精度、模块模式、预算不变。

唯一必要软件变更：把原硬编码训练seed42改为显式CORESET_SEED=43/44，更新路径、端口、种子校验及报告元数据。libero模型、IFA/优化器基础实现、greedy_kcenter和eval_metric字节保持相同。通过完整resolved config归一化对比，除seed与新路径外必须与seed42相同。各seed有新合同与代码hash，禁止把旧特征或checkpoint改标签复用。

同一RTX3090顺序队列：seed43 configure→兼容性审计→18软件测试→真实GPU预检→正式export/train/eval/验收；全部成功后seed44同流程。每个新seed隔离预检最多4次optimizer调用和一次单零动作环境接口检查，不新增策略闭环smoke。正式运行重置RNG，从不接续smoke训练状态。

队列失败立即停；不盲重启queue/pipeline。保留原日志/退出码，先确认进程已退出、合同输入一致，再恢复未完成阶段。训练只从该seed该arm完整epoch checkpoint恢复全部模型/optimizer/scheduler/grads/模式/RNG/cursor。存储按已提交清单幂等复制，不重抽/重复追加。部分评估必须验证原全局RNG连续性后技术恢复；正常失败不重试。不悄改冻结代码或重贴合同。

所有新组完成后只读复算三seed的每任务矩阵、精确指标和C−R，报告逐seed、均值、样本标准差ddof=1，并附总体标准差ddof=0避免混淆。扩展是在看到seed42后决定，属于探索性扩展，不能包装成事先独立确认试验或仅凭三个seed宣称显著性。官方预训练模态元数据unknown、RTX3090对论文A100、调度与ep99规则未公开、IFA随buffer间接变化等边界保留。

用户之前对本实验的定时监控/可恢复堵塞处理/完成后取消授权持续适用。新增扩展监控通常每2小时，临近结束可30分钟；最终验收并下载轻量证据后取消。不push、不上传或下载权重、不启动新付费资源、不扩展其他seed/套件/超参数。
