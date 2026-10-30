# 仅C匹配已有完整S优化器调用预算：GOAL/OBJECT seed41/42/43

2026-10-30用户已选择“保留完整S预算、增加C”“保留各自方法，仅匹配优化器预算”，随后明确“S不用重跑，只跑C”。本协议覆盖原DRAFT的12个新run方案，仅重新训练6个C完整序列并新增1200正式回合。已有S训练、240采集、1200正式回合及独立验收只读复用，不重训/重评S；旧机制永久取消。

|套件|Task6 C调用|Task7 C调用|Task8 C调用|Task9 C调用|每seed C生产调用（等于已有S）|
|---|---:|---:|---:|---:|---:|
|GOAL|10000|27800|37000|67400|142200|
|OBJECT|15700|45400|61800|72400|195300|

C在Task6为100轮，Task7–9各200轮，700轮/seed；对应已有S四阶段各100轮，后续阶段C/Q两个候选均计费（包括丢弃候选）。各阶段生产调用相等，C所有生产更新保留。新6C合计4200轮、1012500接受/物理训练调用，与已有6S的1012500物理调用匹配。不得将S544800接受更新误写成物理调用数。

各suite/seed官方初始化、RA成员/顺序/标签/mask、batch10、模型、原C BC+0.1IFA+原expert/route目标、AdamW1e-4/wd0.1、clip100、Task6→9保持。原C的teacher/tau/w准备仅保留原执行流，C目标不用teacher值；无S采集、成功保持缓存或保持损失。训练数据配置与已有S共同RA绑定一致，不宣称完全相同训练信息或等交互。

按预算适配每阶段学习率调度终点：Task6总步数100×batches，后续200×batches。保留原linear warmup/decay函数和500 warmup；显式覆盖a.total_steps和cfg.train.n_epochs后再建立scheduler，实际核0/500/中点/末步/终点，防止后100轮零学习率空转。C sampler每阶段epoch0–99或0–199、seed0，不把额外数据遍历说成与S接受轨迹相同。初始模型、全RNG和硬件与实际已有S逐项比对；Task6最终tensor也与该S实际CP精确比对，失败保留原证据，不自动重跑。

每阶段自己的隔离准入：Task6 4 optimizer；后续原C5调用，加两次10步完整重放和两次2步保存加载后续重放24调用，共29。全部恢复初始全state，无生产更新或环境回合；每seed91，与已有S91相等。六C隔离546，计划新optimizer总1013046（既有六S本轮自身生产+隔离同数）。这些验证检查实际预算扩展路径，不用无效调用填数。C准入12 autograd/seed、方向诊断48/seed分别记账；诊断epoch0/19/99/199；任何额外工程调用须独立披露，不能混入生产或抹去旧成本。已有OBJECT的3914工程验证及历史中断UNKNOWN仍归原账，不能声称所有历史总成本相等。

复用已通过逐位验证的CPU复制优化，每个新C阶段起点再实际比对原capture全部state；只去掉CUDA→CPU后的重复CPU clone，CPU来源仍clone。原C损失/updates/evaluate源码逐字复用（仅运行配置/预算和验证外置修改），源hash及diff保存。准入未实际通过前不能称PASS。

独立root `C_MATCH_FULL_S_BUDGET_GOAL_OBJECT_20261030_v1`。固定队列GOAL41/42/43、OBJECT41/42/43六C训练，随后同顺序六C正式评估，绝不调用S入口。每阶段固定最终epoch99/199，无测试选择。正式各stage全部已见task初态0–19，每cell20回合，600步5zero、原配对Normal噪声和env_seed；每C200、共1200，新旧C/S配对合计2400已有/新增outcomes。不得重试/补采/加seed/扫参。

数据目录保守新增峰值18GiB并留2GiB，system普通可用留1GiB；旧资产不删除。只部署一次，无盲恢复入口。失败先核原进程、冻结hash、完整epoch恢复、原失败/候选/环境attempt及成本；不重复已完成轮或回合。

全部自然exit0且无自身进程后独立实际CPU24阶段CP、6完整recovery、18滚动teacher缓存重放、全部RA/标签/初态字节/noise与1200正式回合验收。与已接受S逐一核同suite/seed数据、官方初态、任务顺序、正式初态/noise/envseed与阶段预算；比较两套件各自三seed矩阵/AUC/FWT/NBT低好/OldFinal/FinalAll，全部负单元及配对SF/FS。等optimizer调用不是等FLOPs/墙钟/交互；S额外240训练采集保留，开发seed及旧测试初态不能宣称新独立确认集或单机制因果。

每次检查和最终接受按根AGENTS审查轻量文件nonforce发布MLR并核remoteSHA和本地receipt，禁CP/状态/缓存/RNG/凭据。按整seed或全部评估实测余时调整同一新监控；健康进展静默，完成/新失败/需操作才通知。六C独立验收、与已有S解释及发布后才删除监控。


## 2026-10-30 16:20 用户调度修订（覆盖此前全训练后统一评估顺序）

GOAL41/42/43训练→GOAL41/42/43正式评估→OBJECT41/42/43训练→OBJECT41/42/43正式评估。S只读，所有科学预算与配置不变。不打断现有GOAL42或重跑完成部分。工程目录engineering_suite_order_20261030_v1实现自然训练边界交接，保持原科学冻结源；工程完成标记为该目录controller.exit，原root/controller.exit不伪造。详细安全证据、使用方式与当前ARMED/完成状态以MONITOR_HANDOFF及实际快照为准。
