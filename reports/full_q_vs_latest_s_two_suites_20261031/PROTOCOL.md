# Q 对最新完整 S：GOAL / OBJECT 各 seed41、42、43

用户2026-10-31明确授权六个Q，已有完整S直接复用，不重训不重评。经澄清选择各任务100轮，比较有无门控，不匹配物理optimizer预算。

Q从各suite/seed官方初始化开始，Task6→9各100epoch；Task6无保持，Task7–9保留最新S的逐任务q加权保持，关闭双候选门控，所有生产更新只执行Q。q=该任务自身成功缓存贡献的独立采集轨迹数/10；保持系数0.1，回归和route均乘q，原4槽分母，不归一化、不补采。原RA数据、合法mask、主batch顺序、FP32模型、AdamW、1e-4学习率、weight decay .1、warmup500、100轮调度终点及梯度裁剪100不变。

每seed每阶段仍采集初态20–29十次，600步上限/5zero、配对Normal噪声、成功轨迹quota128；Task9也采集。各Q用自己的缓存，不能复制S缓存；缓存差异属于完整训练路径的后果，不能称共同中间状态下的单步门控因果实验。正式初态0–19，各stage所有已见任务，共200回合/seed，原配对噪声/环境seed/600步/5zero不变。S既有1200回合仅只读配对。

GOAL每seed76100接受且物理调用；OBJECT105500。六Q2400epoch、544800生产调用，240训练采集、1200正式。Task6自身4隔离optimizer/8额外autograd，7–9各29/11（沿用已有S准入并实际测Q恢复、q1原S、lambda0原C、10步原C重放、测量状态不变与短历史）；总546隔离/246准入autograd。训练方向诊断每stage在ep0/9/49/99固定batch最多216额外autograd；固定训练探针只测量，不决定Q更新。没有生产C候选、没有门控决策，也不能把准入C调用混入生产。

GOAL三seed训练→GOAL三seed正式评估→OBJECT三seed训练→OBJECT三seed正式评估。每阶段独立准入；实际初始化/RNG/hardware与对应已接受S绑定，Task6全tensor精确对比S。复用已验真的CPU复制优化，每stage实际COPY_EQ对原完整状态比较。计数器独立记录物理AdamW.step，恢复不能抹去已耗调用。

新独立root FULL_Q_NO_GATE_GOAL_OBJECT_20261031_v1；不运行旧入口，不覆盖旧结果，不恢复已取消机制。逐epoch完整保存，任何失败先核进程、合同、已保存轮、完整状态及原失败成本；禁止盲重启部分轮或环境attempt。无负结果重试/选优/扩seed/扫参。容量按8GiB新输出峰值预留、root普通1GiB、data2GiB，不删除旧资产或付费。

最终自然controller.exit0且无自身进程后，真实CPU验收24阶段CP、6完整恢复、24成功缓存teacher重放、1200实际初态/noise、全部逐轮账目与每stage预算；再与已有S逐回合配对，解释新旧任务、SF/FS、全部负单元及成本边界。Q/S同采集次数和接受更新数，不同缓存内容、物理调用、FLOPs或墙钟；既有开发seed非独立确认集。未实际验收前均PARTIAL。

每轮按根AGENTS发布审查后的code/report/hash/scalars/outcomes到MLR，禁止模型/CP/缓存/状态/RNG/凭据。六Q验收、S比较解释及发布全部完成后关闭本轮新监控；旧监控不恢复。

## 实际工程版本补记

活动root已版本化为FULL_Q_NO_GATE_GOAL_OBJECT_20261031_v2。v1在首次optimizer/环境之前因只读调度检查属性错误退出，实际零成本证据在engineering_startup_v2/capacity_readonly_20261031_095052；v1全部失败和科学冻结文件保留。v2仅修正检查表达式，科学协议与预算完全相同。源码见engineering_startup_v2六case；根六case为v1历史，不得部署。
