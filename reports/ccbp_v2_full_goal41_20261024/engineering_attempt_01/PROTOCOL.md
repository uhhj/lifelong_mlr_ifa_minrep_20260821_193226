# CCBP-v2：GOAL seed41 四组完整序列

用户明确替换此前共同Stage8的Task9分叉，要求四组各自从头训练完整GOAL序列。本项目的完整增量序列为Task6→7→8→9；从官方GOAL multitask ep96初始化，不是随机初始化视觉模型，也不加载已训练ALIGN Stage8或其他实验的下游权重。四组B/C/N/V各自独立训练全部400epoch。此前分叉9epoch后工程失败记录保留，不恢复、不当科学负结果。

## 训练人口与预算

原GOAL RA-Coreset的成员、样本顺序、payload、ALIGN下一步动作标签与mask、IFA选择及语言参考保持不变。只读复用冻结的各阶段输入manifest和features/labels，不复制payload，不暴露未来任务样本。逻辑回放从空开始：Task6无旧数据，Task7仅6，Task8仅6/7，Task9仅6/7/8。旧成员由原RA选样确定，四组共享相同人口以隔离辅助目标。并不从旧策略权重续训。

|阶段|人口|合法BC|屏蔽|合法旧窗口|每轮主步|100轮主步|
|---|---:|---:|---:|---:|---:|---:|
|6|997|917|80|0|100|10000|
|7|1388|1286|102|477|139|13900|
|8|1848|1723|125|899|185|18500|
|9|3365|3218|147|1329|337|33700|

每组400epoch76100主更新，四组1600epoch304400主更新。105500是OBJECT预算，不能用于GOAL。FP32、batch10、sampler0、seed41、原AdamW1e-4 wd.1、各阶段重置优化器和500warmup原scheduler、clip100、Temporal modality0、5分量7D GMM、BC+0.1当前任务IFA/gamma.7保持。整个组内随机流连续，阶段之间不人为重置为41。每组初始tensor及全部RNG须与原官方初始化证据精确相等；Task6最终tensor应与原ALIGN Task6精确相等。原始RA和ALIGN的历史评估只作外部参考。

## 辅助方法与教师

Task6无旧窗口，辅助项为0。Task7–9在该阶段所有合法旧窗口上加辅助，不仅Task8。每组只使用自己上一阶段ep99模型，eval/no_grad冻结并整阶段固定，cache逐身份、manifest、CP、标签SHA绑定，不读取其他组教师。教师生成必须保存/恢复全RNG，避免改变BC流。

尺度延续原算法：该阶段合法旧训练窗口的专家动作分组RMS并floor0.1，分组平移3/旋转3/夹爪1，四组同输入得到相同尺度。不能把旧Task9的数值尺度用于更早阶段。每个组/阶段tau=该教师执行分量专家cost的float64中位数（floor1e-8），w=tau/(tau+cT)，一次冻结，不用测试调整。

B：原执行Normal分组W2教师蒸馏与最接近教师分量margin0.1路由。C：原执行分量专家cost与最小cost margin0.1路由。N：rho0，J=c及新版代价差softplus路由。V：rho1，J=(c+w*d)/(1+w)。c为包含方差的分组归一化期望专家平方误差，d为对教师实际argmax执行Normal的分组W2平方。执行项J[argmax logits]；路由sum stopgrad(Jm-minJ)*softplus(zm-zbest)/(5-1)，同分数按最高logit选reference，完全并列遵循上传原型。总loss=BC+.1IFA+.1mean(exec+.1route)。不加新网络、不扫描margin/epsilon/rho，不为特定失败窗口加权。推理无专家、无tau/w参与，原argmax分量Normal采样和逐步重观测不变。

## 准入、诊断与恢复

各组Task6自身准入3次序列化/下一步重放更新加1次原ALIGN更新对照，必须精确。每组Task7–9以实际本组阶段教师再做3次恢复更新、1次有既存梯度的测量开关更新对照；B/C另1次原loss实现对照。合计70隔离optimizer更新，0环境；每次回到正式初始全状态，不能挪用准入权重。每次真实长度1–8和argmax Normal部署检查。旧分叉准入不是本实验PASS。

此前诊断断言把optimizer.zero_grad()留下的零tensor当作必须None，是工程错误；新版记录诊断前p.grad状态，autograd.grad后逐tensor精确比较并核RNG，不改变BC/辅助损失。真实准入必须在已经执行过更新的状态下验证测量开关精确，不只测新建优化器。

各组Task7–9的epoch0/9/49/99首个合法旧batch记录BC、加权IFA、加权exec、加权route的梯度L2和非零tensor数。正式额外192次autograd，准入额外48次，单独计耗时。每完整epoch原子保存model、optimizer、scheduler、grads、modes、requires、allRNG、stage/next_epoch、累计步数及history、teacher/cache/label/input绑定。停止/故障只核全状态后版本化完整epoch恢复，不盲重启controller或部分正式评估。

## 全矩阵闭环与停止标准

每组Stage6评Task6、Stage7评Task6/7、Stage8评Task6/7/8、Stage9评Task6/7/8/9，每cell20初态0–19。四组800episodes80vectors40cells、16实际CP。四组同task/index以及不同stage使用相同600×7float32 Normal噪声（seed20261024+41000+task*100+index）；每组10env、envseed41/42、600步与5zero不变。沿用前次已探索初态和noise，属于探索，不冒充确认实验或历史globalRNG逐样本等价。

全部四组完成才比较AUC、FWT、NBT（注明低好）、OldFinal、FinalAll、各阶段各任务成功率、所有负单元和配对SF/FS。延续候选v2主要依据：最终V Task8>C且V Task9>=C；Task6/7副作用单独报告，任一低于C不能声称其他旧技能无损。完整序列整体保持还须报告旧任务平均与全矩阵，不能用一个均值掩盖Task9下降。不强求每任务第一，不自动扩seed/组/权重。V对N只说明锚的增量，未加uniform不能独立声称一致度权重有效；若只改善离线代理而闭环未改善则停止该版本。

## 存储、发布、调度

经用户明确授权，精确删除三份废用中间恢复副本，删除前hash/size、删除后空间和旧最终资产留存证据均保留。旧最终模型、最终恢复、原始数据、报告不删。新组B/N恢复在/root，C/V在/data；Task6–8共12模型CP除B6在/root外均在/data；Stage9最终CP与不可再更新的恢复文件硬链接。完整恢复字段不缩减，每volume按唯一inode统计，原子临时文件算峰值。root新增1.07GiB/data新增1.88GiB边界（含teacher/cache/标量与保存峰值），system普通留1GiB/data root留2GiB。容量不够先停并报告，不降精度/预算或付费。

新目录execution/20261024_ccbp_v2_full及新root CCBP_V2_FULL_GOAL41_20261024_v1，不覆盖旧冻结源码。发布仅代码/报告/metadata/hash/scalars/outcomes，按AGENTS精确审核非force推MLR核remoteSHA/receipt。正式前预登记，再自己的真实准入。预计时间先以GOAL阶段预算和可比速度提供量级，实际按整组训练或整段评估完成点安排同一监控，不高频内部阶段轮询。全部最终独立验收及发布后删除监控。
