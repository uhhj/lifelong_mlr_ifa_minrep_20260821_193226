# 下一轮：MLR+IFA + ALIGN，Random示范记忆

用户2026-11-09明确授权：当前固定q=1重复运行之后，OBJECT、GOAL各seed41/42/43，共6个run。已有MLR+IFA结果直接复用，不重训不重评。当前q1训练不中断、不修改冻结调度器。此文件记录已授权队列；不是已部署或自动启动成功证明。

## 唯一科学变化

以各seed已接受的MLR+IFA Random配置为父源，保留其有序Random成员、配额、原始特征、初始化、任务顺序、模型、BC+0.1 IFA、IFA任务对选择、优化器和调度；只加入历史ALIGN的下一动作目标及合法mask。不要从当前C/Q/S复制训练目标，因为它们含额外执行专家监督E。

明确不含RA-Coreset、执行专家监督E、成功轨迹保持R、支持度q或候选门控G；没有训练期环境采集、成功缓存或teacher保持。只训练新配置，不改变已有Random、RA、RA+ALIGN、C/Q/S结果。

ALIGN遵循已接受实现：同轨迹时间身份唯一，窗口内8帧真实未填充，且窗口末端u的下一动作u+1存在，目标为a[u+1]。不合法窗口仅屏蔽BC，按有效窗口数取均值；全无效BC为可微零，IFA照常。保留原人口、排序、shuffle/sampler和更新数，不能丢弃非法窗口或把当前论文文字中的简化mask另当新算法。对应源码bc_labels.py、masked_bc及observe的ALIGN分支已做源绑定；不能启用同文件内TAC/EFTR等其他分支。

Random历史成员与样本SHA逐阶段绑定，不能用RA buffer改名；也不重新随机抽一套后把差值全归于ALIGN。若原Random成员记录缺失，先从对应远端冻结缓存/manifest只读核实。只有确认同一成员与顺序后才构建自己的训练manifest和标签。新模型从官方初态开始，不能加载已训练的旧Random末轮或RA权重接续。Task6监督改变，因此不要求其最终tensor等于原Random。

## 固定预算与评估

每任务Task6–9各100轮，batch10，FP32，AdamW lr1e-4、wd0.1、warmup500、逐更新线性调度及各阶段重置规则与父源一致。OBJECT gamma0.3、官方object ep99；GOAL gamma0.7、官方goal ep96。保持论文模态vision/language/state=True。

|套件|run数|完整轮|单候选生产optimizer调用|训练采集|正式回合|
|---|---:|---:|---:|---:|---:|
|OBJECT|3|1200|316500|0|600|
|GOAL|3|1200|228300|0|600|
|合计|6|2400|544800|0|1200|

OBJECT每run阶段batch为157/227/309/362；GOAL为100/139/185/337，须实际manifest人口确认。本轮每stage计划5次自身隔离optimizer准入（原更新/禁用ALIGN等价2次、ALIGN两步及加载重放3次），24stage合计120次，另账；目前0次，不能套未来PASS。不得混入544800次生产更新或未经记录重试。复用历史ALIGN准入设计的关闭等价、合法mask、空mask、保存恢复，并要求新Random数据自己的实测证据；旧RA数据上的PASS不是新Random准入。

正式使用四阶段三角10格、各20个初态0–19、ep99、600策略步上限/5zero、10lane、argmax GMM分量Normal。为与MLR+IFA/历史RA+ALIGN对照，沿用历史按stage重置seed的global动作RNG和原评估执行口径；不能静默替换成当前q1的逐轨迹固定噪声评估器。实际初态、env seed、噪声方案、模型SHA、attempt全部记账。跨硬件或历史未记录逐轨迹噪声的差异如实保留，不补评旧基线来追求相同结果。

## 接续与资源

当前q1两套件训练正式自然退出并完成独立验收、比较和发布后才启动本轮，不与q1训练/评估重叠。采用独立版本目录，独立目录execution/20261109_mlr_ifa_align_baseline；远端root为MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261109_v1，已只读核不存在，部署时再次核唯一性。最多两个训练并行，独立c10d localhost:0和UUID；所有正式环境互斥。OBJECT三训练后600串行正式，再GOAL同序；不以OBJECT成绩决定GOAL是否运行。

当前q1已冻结的协调器只包含q1六run。本授权已持久保存，ALIGN源适配、输入审计、文本冻结和单次部署入口已准备完成。当前未部署，未安装后台等待器，不能称自动接续已启动。本轮结束前可准备本地轻量代码与协议；不修改当前冻结controller。接续时需要新的实际容量预检，旧的8.15GiB空闲已包含q1未来占用，不能重复当ALIGN可用空间。只传轻量源码/标签元数据；不向本地下载模型或大备份，不删除旧模型、结果或付费扩容。

## 分析与接受

保留所有六run与所有负单元。与既有MLR+IFA(Random)、RA、RA+ALIGN组成2×2：原监督/ALIGN × Random/RA。先分别报告新配置对Random及RA+ALIGN的差值，再描述交互；不预设ALIGN或RA在两套件必然提高，不按结果挑seed/版本。OBJECT43比较父源仍为历史表中已绑定的repeat1，其既有选择历史不改写。

报告全部AUC/FWT/NBT/OldFinal/NewFinal/FinalAll、三seed均值±样本SD、十单元矩阵和逐回合SF/FS。新旧初态匹配不等于历史逐轨迹随机流完全一致；硬件、前端预训练来源和开发集重复使用边界继续保留。实际完成、CPU模型与恢复验收、完整记录预算、独立指标复算和发布全部通过后才接受。
