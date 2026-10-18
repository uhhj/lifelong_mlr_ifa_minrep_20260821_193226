# SC-v2 GOAL seed42最终验收与比较

**工程实验COMPLETED；效果门槛未通过；不扩展seed43、44。** 服务器于2026-10-18 22:11 CST完成退出验收，本地已独立复算。原Coreset和SC-v1均使用已有同seed闭环证据，没有重训或追加评估。

## 完整指标

单位为百分比；差值单位为百分点。NBT采用已冻结公式，越低表示该指标下的净遗忘越小，但可以被任务间改善抵消。

|方法|AUC|FWT|NBT|最终旧任务均值OldFinal|最终全部任务均值FinalAll|
|---|---:|---:|---:|---:|---:|
|Random（参考）|80.9375|82.50|4.4444|76.6667|80.00|
|原Coreset|78.7500|81.25|5.5556|68.3333|73.75|
|SC-v1|73.4375|76.25|7.2222|70.0000|70.00|
|SC-v2|69.0625|70.00|0.0000|70.0000|65.00|
|SC-v2 − 原Coreset|−9.6875|−11.25|−5.5556|+1.6667|−8.75|
|SC-v2 − SC-v1|−4.3750|−6.25|−7.2222|0.0000|−5.00|

精确值保存在LOCAL_FINAL_ACCEPTANCE.json：SC-v2 AUC=221/320，FWT=7/10，NBT=0，OldFinal=7/10，FinalAll=13/20。

固定推进条件逐项判定：

- AUC严格高于原Coreset：221/320 > 63/80 不成立。
- 最终旧任务均值不降低：7/10 >= 41/60 成立。
- 工程完整验收：PASS。

双指标必须同时通过，因此NO_EXTENSION。未新建43/44训练或评估，SC-v2定时监控已删除。

## 全部阶段和任务

每个单元20条真实闭环，成功率每5个百分点对应一条episode。表中所有10个有效单元均完整，没有挑选任务或checkpoint。

|阶段|任务|Random|原Coreset|SC-v1|SC-v2|
|---|---|---:|---:|---:|---:|
|6|6|65|65|65|65|
|7|6|70|65|70|65|
|7|7|100|100|100|100|
|8|6|60|65|60|55|
|8|7|100|100|100|100|
|8|8|75|70|70|65|
|9|6|70|45|60|30|
|9|7|100|100|100|100|
|9|8|60|60|50|80|
|9|9|90|90|70|50|

SC-v2最终Task8达到80%，比原Coreset高20pp、比SC-v1高30pp；但Task6只有30%，分别低15pp、30pp；Task9只有50%，分别低40pp、20pp。Task7保持100%。最终旧任务均值70%掩盖了Task6与Task8此消彼长，不能解读为各旧任务均受到保护。

AUC变化的逐任务贡献（各任务在AUC中权重1/4）：

|对照|Task6|Task7|Task8|Task9|合计|
|---|---:|---:|---:|---:|---:|
|SC-v2 − 原Coreset|−1.5625|0|+1.8750|−10.0000|−9.6875|
|SC-v2 − SC-v1|−2.5000|0|+3.1250|−5.0000|−4.3750|

本轮整体下降主要由Task9贡献。Task9本任务训练结束后才执行Task9入库，后面没有训练阶段，因此不能将Task9下降归因于Task9最后那6条入库交换。可能影响Task9训练的是此前Task6–8记忆成员引发的训练轨迹变化；本轮没有隔离这些变化的具体机制。

NBT=0也不是“没有遗忘”：Task6由初学65%降至最终30%，其后续阶段平均遗忘为15pp；Task8由初学65%提高至最终80%，对应−15pp；Task7为0，三者平均恰好抵消。SC-v2较低的初学Task8成功率也改变了NBT参照点。必须与FWT、AUC和全部任务矩阵共同解读。

## 工程验收

- 400个唯一task/epoch组合；每任务100完整epoch；总76,100 optimizer step；没有提前停止或追加训练。
- 4个epoch_index=99最终checkpoint在服务器严格加载，每个100个下游参数键；记录权重SHA256、训练合同及模态合同。没有下载或发布权重。
- 200条正式记录，20次向量开始与20次向量完成，10次环境构建与关闭，0技术异常、0技术重试；每个单元恰好20个原初态，attempt与逐episode结果完全一致。
- 原Coreset、Random、SC-v1和SC-v2对应episode初态哈希、模态合同一致；评估规则一致，但没有独立隔离每条轨迹的实际动作随机流。
- Task6首次入库前模型张量一致，max_abs_diff全部为0；初始化状态与父Coreset相同。
- 冻结源码、原输入及载荷保护通过；90个最终轻量包文件逐一核对bytes/SHA256。本地从逐episode记录用Fraction复算全部指标，与服务器AUDIT完全一致。
- train/evaluate/audit/pipeline/supervisor退出码全部0；PROCESS_EXIT_ACCEPTANCE=PASS，实际检查无残留训练或评估进程，GPU空闲。

实际入库顺序与离线一致，全部2,381个buffer文件源端哈希验证通过：

|任务|配额|替换数|不同监督时刻，前→后|最大半径守线|平均距离守线|
|---|---:|---:|---:|---|---|
|6|499|5|495→495|PASS|PASS|
|7|445|2|426→426|PASS|PASS|
|8|452|4|449→449|PASS|PASS|
|9|985|6|981→981|PASS|PASS|

原示范及终点支持保留，构成偏差E下降。Task6–8实际影响后续训练的不同成员仅11条。IFA选中任务对与原Coreset相同，但视觉相似度和实际记忆输入有变化，不能说IFA实际损失路径完全相同。

本地第一版复算对全部浮点中间值使用了逐字值相等，因Task8原mean的1.39e-17差异退出1；已保存LOCAL_RECOMPUTE_DIAGNOSTIC.json，按冻结的原1e-12容差修正只读比较后退出0。所有身份、计数及指标仍精确比较；未更改服务器源码、合同、输入、阈值或结果，没有重跑环境。

## 实际命令、时间与产物

以下由final_evidence/commands.jsonl与timings.jsonl记录，Python为/data/mlr_ifa/env/bin/python：

```sh
python -m torch.distributed.run --standalone --nproc_per_node=1 -m coreset_exp.train sc_coreset
python -u -m coreset_exp.evaluate sc_coreset
python -u -m sc_exp.audit
```

三条退出均0。训练2,328.32秒（38.81分钟），评估4,317.91秒（71.97分钟），审计16.17秒；三阶段合计111.04分钟，不含先前离线/预检与本地核查。

本轮检查执行monitor_snapshot.py、package_light.py，均退出0；没有重新执行训练或闭环入口。本地命令：

```powershell
python -X utf8 execution/20261018_scv2_goal_seed42/prepare_comparison_evidence.py
python -X utf8 execution/20261018_scv2_goal_seed42/recompute_final.py
```

远端产物：/root/mlr_ifa/experiments/SCv2_Coreset_GOAL_seed42_20261018。
本地：execution/20261018_scv2_goal_seed42/final_evidence、baseline_evidence、scv1_evidence、LOCAL_FINAL_ACCEPTANCE.json、final_matrix.csv。

## 解释边界

这是单seed、每单元20初态的探索性筛选；不能作显著性或普遍因果断言。几何覆盖和离散构成约束按预定方式实现了，但未转化为整体性能收益；本轮证据不支持继续扩seed。仍披露原global动作RNG未逐轨迹隔离、官方预训练state配置元数据unknown、warmup/调度及ep99规则补全、IFA输入间接变化。没有因负结果选择其他checkpoint、改门槛或追加评估。
