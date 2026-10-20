# LIBERO-50 尚未启动：资产缺口与实现选择

用户授权自动获取数据并自主选择推荐实现，保持既有实验协议。以下选择在看到任何LIBERO-50结果前记录，不能将其称为作者完整代码复现。

1. 数据：官方LIBERO镜像固定revision，50厨房任务与固定LOTUS划分逐名匹配。HDF5存储无损重排，所有数组字节与属性相同；原上游SHA与新打包SHA并存。
2. 权重：截至本次官方仓库/发布页/公开模型目录检查，只找到OBJECT与GOAL权重。不得跨套件冒用。准备从相同OpenAI CLIP ViT-B/16基座、25个预训练任务各50示范复现共享预训练，固定seed42，100epoch、batch10、AdamW1e-4，最终ep99。三个下游seed从同一份新权重起步，与此前共用预训练的做法一致。权重名称必须明确`reproduced`。
3. 预训练范围：CLIP基座冻结，仅其rank8/alpha8的q/v LoRA可训练；新建视觉投影、语言FiLM、状态编码与状态FiLM、时序网络及GMM头训练。论文附录明确CLIP LoRA/时序/GMM，但未穷尽新建模块的优化细节，故此处作为明确的工程补全披露。
4. 预训练按原GroupedTaskDataset的25任务交错索引与DistributedSampler(seed0)完整遍历。没有闭环挑选checkpoint。固定FP32；任何内存方案不得悄改batch、优化步数或精度。
5. 下游：5阶段每阶段5任务、每任务10示范，100完整epoch、batch10，BC+.1平均IFA，gamma.1、500步warmup、每优化步线性调度。原RA每任务ceil(.5*N)配额与半配额随机锚点、距离及标定seed42不变。每阶段5任务全部当前数据与旧buffer拼接，原排序后Random(seed)打乱，DistributedSampler seed0。
6. 多新任务IFA补全：在当前5任务和旧实际记忆上建立共同相似度表，沿用语言/视觉top-half交集。对每个当前任务使用包含其自身的选中对，旧样本仍只参加BC；对当前batch有效样本平均其配对项。单新任务时退化为原实现。当前任务之间的对必须记录，不凭闭环表现改变。
7. 正式评估：5个ep99阶段模型，仅评估已学的新增任务，不含25个基础任务；5+10+15+20+25=75单元，每单元20原初态，共1500条/seed、150个vector，三seed4500条。原stage global RNG、10lane、env seed[s,s+1]、600步、5零动作、argmax分量后Normal不变。
8. 指标先按每阶段五个任务平均形成5×5三角表，沿用原精确有理数指标公式；同时公布5×25任务表，不能把不同任务聚合掩盖。
9. 不额外训练Random/Coreset对照，不使用论文均值冒充配对基线。所有预训练成本、准入步骤和下游成本分开记账。新增预训练可能显著延长总时间，须从真实GPU准入吞吐估计，当前不提供虚假完成时间。

参考：[论文](https://arxiv.org/html/2603.10929v2)、[作者仓库](https://github.com/yfqi/lifelong_mlr_ifa)、[LOTUS](https://github.com/UT-Austin-RPL/Lotus)。固定任务划分、原始代码与数据SHA见PARTITION.json及资产清单。
