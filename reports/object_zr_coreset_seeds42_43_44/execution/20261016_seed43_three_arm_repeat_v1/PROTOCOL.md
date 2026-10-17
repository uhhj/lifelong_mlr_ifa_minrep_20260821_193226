# seed43 Random / Coreset / Event+Coreset完整重复运行 v1

用户2026-10-16明确选择“完整重跑：重新训练三组并评估”。此授权覆盖旧实验“不重训基线”约束，仅新增本次三组seed43重复。旧结果、代码、缓存、权重只读保留，不根据结果筛选或替换旧seed43，不新增seed、阈值、超参数或独立RNG评估方案。

- 三组均从相同官方OBJECT预训练权重、各自空buffer开始，网络跨任务继承。复用原seed43已验明的5921个state=True特征；保留FEATURES与MODAL_CONTRACT原始字节和来源合同哈希。重新训练不接续旧模型或旧optimizer。
- 每组任务6→9各100完整epoch、batch10，105500optimizer step，4个epoch99最终checkpoint；合计1200epoch/316500step/12checkpoint。
- 原BC+0.1平均IFA，gamma0.3，逐step线性调度/warmup500、模型、dtype、sampler默认seed0、Coreset标定seed42、存储配额781/740/780/660全部不变。训练/评估seed43；Event沿用冻结candidate_intervals_v1与rho=1/2，仅改变入库成员。
- 原20初态0–19、每阶段已见任务、10lane×2、env seeds43/44、600步、原argmax GMM分量后Normal采样、阶段全局RNG。每组200新episode，总600。不是此前取消的9000条复评，也不追加680口径。
- 新Random和新Coreset共用一份训练实现，新Event沿用原Event实现，仅将配对父运行指向本次新Coreset。组内起始RNG/模型和Task6实际张量必须一致；不强求与旧3090运行逐位一致，不按闭环成败重试。
- 当前服务器GPU为RTX4090，旧运行RTX3090；作为硬件变化披露，不能将前后差异全部解释为采样随机性。无新付费资源启动。
- GPU预检每个实现4次隔离optimizer调用及1次单零动作接口检查（共8+2），不计正式训练，不追加策略rollout。软件测试RC18项，Event28项及其父18项。正式进程重新初始化RNG，不接续预检状态。
- 数据、正式buffer/权重/日志位于/data数据盘。原特征只读符号链接复用，隔离预检smoke恢复文件放系统盘专用目录，避免占用正式产物空间；不删除任何旧产物。
- 不创建新的定时任务、不push、不下载或上传权重与演示。后台有限队列依次RC训练/评估/验收，再EC训练/评估/验收，最后新旧完整结果对照。原seed43仍作为首次运行报告，此次明确标记repeat1，不作为新的独立seed统计。
- 工程失败保留命令/日志/退出码并停止。仅从本次本组完整epoch恢复模型/optimizer/scheduler/grads/modes/全部RNG和cursor；存储按预提交清单幂等恢复。部分评估不得盲重启，正常失败不重试。
- 最终验收1200epoch/316500step/12可加载最终CP/600记录、三组矩阵/精确AUC/FWT/NBT、入库和IFA、输入hash保护、进程退出证据；负结果同样完整报告。训练完成而评估未完成不能COMPLETED。
