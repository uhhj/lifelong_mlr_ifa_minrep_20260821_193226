# DEBR已完成实验清单（2026-11-06）

本次根据本地既有接受汇总和状态文件整理，不运行训练、环境或远程验收，不改变原结果。最终方法指最新支持度校准与候选门控S，论文名DEBR；早期未门控S单独区分。

## 论文主要证据

|类别|已完成配置|用途|
|---|---|---|
|主基线与逐步加入组件|MLR+IFA、RA、RA+ALIGN、最新S；GOAL/OBJECT各seed41/42/43|完整方案效果及阶段性改进|
|普通强C对照|普通C与最新S；两套件三seed|C保留RA+ALIGN与旧窗口执行专家cost/margin路由，不含成功历史保持|
|等优化器预算C|增加C的生产optimizer调用至S对应调用数；两套件三seed|检验额外更新次数能否解释收益；不称等FLOPs、墙钟或交互|
|完整去门控Q|成功历史与支持度加权保持保留，关闭候选选择；两套件三seed|评估门控方案效果；实际优化器预算不同|
|保持系数敏感性|lambda_R=0.05/0.1/0.2；两套件三seed|两个新增档共12次完整训练，0.1引用既有6次；权重影响及保持/学习取舍|
|行为与资源分析|阶段任务矩阵、AUC/FWT/NBT/OldFinal/FinalAll、配对SF/FS、离线执行cost/路由/保持及候选统计、计算和缓存账目|解释闭环效果及实际代价；分析不重复计为独立训练实验|

上表前五项全序列正式协议每配置200回合。各方法数据重用仅引用一次，不将重复引用计算为新实验。历史基线不保证与后续S逐轨迹噪声相同。GOAL/OBJECT是同一LIBERO基准的两个任务套件。

## 主对比AUC，三seed均值（%）

|方法|GOAL|OBJECT|
|---|---:|---:|
|MLR+IFA|76.35|68.30|
|RA|81.98|73.23|
|RA+ALIGN|81.11|80.17|
|普通C|82.50|85.00|
|等优化器预算C|83.72|80.90|
|Q无门控|83.51|84.62|
|最终S/DEBR|87.26|86.49|

S的FinalAll两个套件均85%；OldFinal为GOAL83.89%、OBJECT81.11%。AUC不能代替逐任务负单元或NBT：GOAL NBT相对原MLR+IFA并非改善。

## 已完成的前期设计探索

- OBJECT seed40–44：Random、普通Coreset、Event-Coreset，15次首次运行3000正式回合；重复运行不并入首次均值。
- GOAL seed41共同Stage8分叉：A/B/C/D，教师蒸馏、持续专家cost及预算可接受集合路线，160正式回合；不是全序列AUC。
- GOAL seed41完整v2 B/C/N/V：教师锚与新路由对比，800正式回合。
- GOAL seed41完整v3 C/R/G：margin回退及纠错方向锚，600正式回合。
- GOAL seed41完整D/O/S：示范历史/全自主轨迹历史/成功自主轨迹历史，600正式及120训练采集；旧S不是最终门控S。
- RA/ALIGN共同窗口离线与配对轨迹诊断：状态COMPLETE_DIAGNOSTIC_ACCEPTED_NO_PROTECTION_TRAINING。
- 有限语义机制筛查：状态COMPLETE_SCREEN_STOPPED_NO_ADMITTED_MECHANISM；不能写成已成功证明因果干预机制。

## 命名和归因范围

已有RA与原基线对比不等于最终DEBR独立去RA对照。D/O/S是早期单seed设计对照，不是最终方法两套件三seed的同缓存纯筛选消融。最终支持度单独关闭、执行路由单独关闭等没有在本次接受汇总中列为完成的全序列独立消融。新确认分布、实机或独立重跑外部方法不在本清单已完成范围。

OBJECT敏感性状态PASS_COMPLETE_LOCAL_ACCEPTANCE，2026-11-03接受并已发布关闭；GOAL敏感性状态COMPLETE_ACCEPTED_REPORTED_PUBLISHED，2026-11-04接受发布。旧嵌套中间状态不覆盖最终接受状态。所有来源字节与SHA绑定在EVIDENCE_REVIEW.json中。
