# A1-OBJECT-v1

Status: OBJECT_COMPLETE_SCREEN_NOT_MET
Corrected own-reference IFA baseline. Only raw IFA features are projected; actions use the original features.
Historical baseline was trained on RTX4090; this run uses the recorded current environment. No baseline retraining or outcome-matching reruns.

|Seed|A1 AUC|Corrected IFA AUC|Delta|FWT|NBT|
|---|---:|---:|---:|---:|---:|
|42|67.083333333|72.916666667|-5.833333333|67.500000000|-0.833333333|

A positive single-seed delta is a screening signal, not evidence of significance or shared-skill decomposition.
No E1/P1/G1/R1/A2/A3, no replay reweighting, no baseline/pretraining rerun. No auxiliary evaluation.
See per-seed raw rollout records, RESULT, MECHANISM and VERIFICATION files.

## 最终独立核验与完成记录

2026-10-11 北京时间 17:05 完成全部获准运行。34 项独立检查全部通过，包括 400 个唯一阶段—epoch、105,500 条连续更新记录、4 个 ep99 检查点与评估加载身份、200 条唯一主评估记录、阶段缓存提交、完整末阶段恢复点、原生采样、冻结源码与配置、增量参考范围、旧基向量保持不变及指标独立复算。

工程与完成性 PASS 不等于研究收益 PASS。本轮结果为 OBJECT_COMPLETE_SCREEN_NOT_MET；不追加 seed43/44，不进入 GOAL/LIBERO-50，不启动其他实验。

|主指标|正参考修正 IFA|A1|差值（百分点）|
|---|---:|---:|---:|
|AUC ↑|72.916667|67.083333|−5.833333|
|FWT ↑|75.000000|67.500000|−7.500000|
|NBT ↓|3.888889|−0.833333|−4.722222|

NBT 改善不能替代预注册的 AUC 筛选门，也不能掩盖初学表现下降。

### 实测成功率矩阵

行表示完成的训练阶段，列表示被评估任务；每个有效单元 20 次执行，单位为百分比。

|A1 检查点|Task6|Task7|Task8|Task9|
|---|---:|---:|---:|---:|
|Task6 ep99|30|—|—|—|
|Task7 ep99|30|100|—|—|
|Task8 ep99|20|95|85|—|
|Task9 ep99|10|100|100|55|

|Task9 ep99 对比|Task6|Task7|Task8|Task9|
|---|---:|---:|---:|---:|
|正参考修正 IFA|25|65|90|80|
|A1|10|100|100|55|
|差值|−15|+35|+10|−25|

按原 AUC 的任务列平均分解，Task6、Task9 分别贡献 −3.750、−6.250 点，Task7、Task8 分别贡献 +2.916667、+1.250 点，总计 −5.833333 点。这是指标分解，不是因果归因。

### 机制与资源记录

四阶段实际参考秩为 7、8、9、10。Task6 和 Task9 原配对规则没有对当前任务施加 IFA。Task7/8 投影范数与原范数的平均比为 0.3827/0.3959，平均投影能量比例为 0.1491/0.1582；小投影事件比例为零。投影 IFA 的任务对—batch 平均 hinge 激活比例为 3.406%/2.049%。这些统计说明投影实际接入，不能单独解释闭环变化的因果机制。

训练子进程耗时 5,986.69 秒（99.78 分钟）；200 次主评估合计 4,441.55 秒（74.03 分钟）。训练和评估合计约 2 小时 54 分钟。训练测得峰值 PyTorch allocated GPU memory 为 831,919,616 字节；不是整个 CUDA 进程的显存占用。逐阶段直接投影 GPU 计时合计约 4.84 秒，不包含日志、验证、状态捕获或其他训练成本；没有匹配的基线计时，不能据此声称总体加速。最终阶段 U32 为 20,480 字节，U64/U32 与参考表张量合计 81,920 字节，其他阶段副本及序列化元数据另计。

### 解释边界

只完成一个 seed；历史基线为 RTX4090，本轮为 RTX3090。无 A1 干预的 Task6 初学成功率已有 35%→30% 的差异，不能将全部轨迹差异都直接归因于投影。按附件规定不重跑来匹配历史结果，不调权重、秩或门限。当前结果足以判定预注册筛选未通过，不足以断言所有参考子空间方法均无效；也不支持“共享知识分解已证明”。

原始执行记录见 PRIMARY_TRIALS.csv；完整机器核验见 INDEPENDENT_FINAL_AUDIT.json；机制、成本与原始来源见 MECHANISM_seed42.json 和 EXECUTION_COST.json。
