# GOAL seed41: Random / Coreset / RA-Coreset

用户明确授权补齐三组；不存在可复用的 GOAL seed41 基线。此扩展在看到42–44结果后决定，属于探索性扩展，不替换旧结果。

先从官方GOAL预训练权重导出本seed的state=True特征，两组原基线共用；Random、Coreset各自空buffer独立起步并跨任务连续继承。基线完成验收后，RA复用本seed对应特征、身份与原Coreset/Random名单，只改变当前任务入库成员。旧42–44与其他实验全部只读。

每组任务6–9各100完整epoch，batch10，共76100步、4个ep99最终checkpoint、200条原协议闭环及20个vector attempt；三组1200epoch/228300步/12CP/600记录/60attempt。BC+0.1平均IFA、gamma0.7、warmup500逐optimizer-step线性、原路径排序concat/shuffle、sampler0不变。state=True三入口一致，原pretrain模态元数据未知如实披露。

RA沿用半配额原Random硬锚点、原视觉动作距离及最小损害交换；标定42，训练/导出/随机入库/评估seed41，配额499/445/452/985，锚点249/222/226/492。不引入SC/Event/残差/新损失/稳定槽位。IFA按本组真实buffer计算并报告实际变化。

原初态0–19、20/cell、10lane、环境seed[41,42]、600步、5零动作、每stage重置global动作RNG为41、argmax分量后Normal。没有独立轨迹RNG，不增加680/9000或重评旧基线。

新产物在已获授权的系统盘专用目录。有限队列：baseline软件18/configure/真实资源审计/4次隔离GPU更新恢复及1零动作接口/导出/两组训练评估严格加载验收，然后RA软件8/真实身份及全删除代价离线验收/freeze/4次隔离GPU更新恢复0环境/正式训练评估验收，最后三组复算及退出验收。已通过步骤不重复，软件不冒充GPU证据。

故障先确认进程退出、保存原失败并核对冻结合同；只从本arm完整epoch恢复模型、优化器、scheduler、grads、模式、全部RNG和cursor，入库按预提交名单幂等补齐；不重抽、不重复计步。部分global RNG评估禁止盲重启，正常失败不重试；无法保持顺序和预算则BLOCKED。不改冻结科学源码或重贴合同。

最终独立复算所有任务矩阵、精确AUC/FWT/NBT/OldFinal/FinalAll、RA-C/RA-R、配额锚点/原载荷哈希、Task6实际tensor与初始化RNG、全部尝试和成本。指标高低不影响固定三组预算与工程验收。所有退出0及无残留才COMPLETED。只发布代码、报告和轻量证据到MLR，不上传下载权重、缓存载荷或凭据；不启动付费资源、不自动扩展。
