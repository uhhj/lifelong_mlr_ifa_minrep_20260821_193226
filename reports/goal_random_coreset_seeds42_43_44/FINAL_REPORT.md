# LIBERO-GOAL 三 seed Random / Coreset＋IFA 最终报告

状态：**COMPLETED**。本地独立验收时间：2026-10-18T15:02:07+08:00。

三 seed 的 Coreset AUC 均低于 Random，平均差 −3.020833 pp；本次 GOAL 结果没有复现 OBJECT 上的平均提升。负结果不影响协议验收通过。没有追加实验、挑选 checkpoint 或修改超参数。

共 2,400 个完整 epoch、456,600 次优化器更新、24 个 ep99 最终 checkpoint、1,200 条正式闭环记录、120 次十路 vector attempt，零技术异常。六组各 400 epoch / 76,100 step / 200 episodes。服务器独立进程实际加载全部 24 个 checkpoint，每个完整加载 100 个下游键；每 seed 两组 Task6 实际张量和初始化 RNG 一致。queue、wrapper、三个 seed 的 pipeline / supervisor 均退出 0，最终快照无残留实验进程。

## 指标（百分数；差值为百分点）

|Seed|方法|AUC↑|FWT↑|NBT↓|
|---|---|---:|---:|---:|
|42|random|80.937500|82.500000|4.444444|
|42|coreset|78.750000|81.250000|5.555556|
|42|C−R|-2.187500|-1.250000|+1.111111|
|43|random|79.062500|76.250000|-6.666667|
|43|coreset|78.437500|83.750000|11.666667|
|43|C−R|-0.625000|+7.500000|+18.333333|
|44|random|79.062500|80.000000|1.666667|
|44|coreset|72.812500|78.750000|10.555556|
|44|C−R|-6.250000|-1.250000|+8.888889|

三 seed 均值 ± 样本标准差（ddof=1，按 seed 统计，未把 episode 当独立训练重复）：

|方法|AUC↑|FWT↑|NBT↓|
|---|---:|---:|---:|
|random|79.687500 ± 1.082532|79.583333 ± 3.145764|-0.185185 ± 5.782406|
|coreset|76.666667 ± 3.341461|81.250000 ± 2.500000|9.259259 ± 3.255258|
|paired_C_minus_R|-3.020833 ± 2.903617|1.666667 ± 5.051815|9.444444 ± 8.624541|

定义：r(s,t) 为阶段 s 后任务 t 的 20 次成功率。AUC = ¼∑t mean(s≥t) r(s,t)；FWT = ¼∑t r(t,t)；NBT = ⅓∑t=6..8 mean(s>t)[r(t,t)−r(s,t)]。NBT 越小越好，可为负。精确有理数保存在 FINAL_RECOMPUTED.json。

## 全部成功率矩阵

行是训练完成阶段，列是被评估任务；每个有效单元固定 20 次。任务6：奶油奶酪放入碗；7：打开炉灶；8：碗放到盘子上；9：酒瓶放到架子上。

|Seed|方法|阶段|Task6|Task7|Task8|Task9|
|---|---|---|---:|---:|---:|---:|
|42|random|6|65|—|—|—|
|42|random|7|70|100|—|—|
|42|random|8|60|100|75|—|
|42|random|9|70|100|60|90|
|42|coreset|6|65|—|—|—|
|42|coreset|7|65|100|—|—|
|42|coreset|8|65|100|70|—|
|42|coreset|9|45|100|60|90|
|43|random|6|80|—|—|—|
|43|random|7|80|100|—|—|
|43|random|8|80|100|55|—|
|43|random|9|95|100|70|70|
|43|coreset|6|80|—|—|—|
|43|coreset|7|55|100|—|—|
|43|coreset|8|70|100|70|—|
|43|coreset|9|70|100|50|85|
|44|random|6|60|—|—|—|
|44|random|7|60|100|—|—|
|44|random|8|55|100|65|—|
|44|random|9|50|100|65|95|
|44|coreset|6|60|—|—|—|
|44|coreset|7|35|100|—|—|
|44|coreset|8|35|100|70|—|
|44|coreset|9|15|100|70|85|

## 差异集中在哪里

Task6 是三个 seed 的共同退步点：最终阶段 Coreset 比 Random 分别低 25、25、35 pp。Task7 六组所有已见阶段均为 100%；Task8、9 的收益方向不一致。seed43 的新任务学习表现（FWT）更高，却伴随更差的旧任务保持（NBT），所以 FWT 提升不能代替持续学习总体改善。

以下是各任务对总 AUC 差的精确分解，已包含总指标的 ¼ 权重，四列相加等于该 seed 的总 AUC 差；它描述差异位置，不证明机制。

|Seed|Task6|Task7|Task8|Task9|
|---|---:|---:|---:|---:|
|42|-1.562500|+0.000000|-0.625000|+0.000000|
|43|-3.750000|+0.000000|-0.625000|+3.750000|
|44|-5.000000|+0.000000|+1.250000|-2.500000|

不能仅凭这些结果认定具体操作片段缺失或距离设计错误；这需要另行授权的只读覆盖/轨迹审计。此次不追加训练或评估。

## 覆盖、配额和成本

每任务完整窗口数 997/889/904/1969，入库条目 499/445/452/985；两组预算相同。24 份 storage 清单的成员唯一、合同一致、阶段累计 buffer 哈希连续通过；服务器已逐项核对原 feature 与入库载荷哈希。Coreset 的 visual/action 距离设置、标定/参考 ID、动作尺度和完整覆盖半径曲线均在各组 storage/task*.json。Random 没有另算同一距离覆盖半径，不能声称已有同口径覆盖改善比较。未下载任何真实载荷。

|Seed|方法|训练调用分钟|epoch累计分钟|环境构建至关闭累计分钟|Coreset选样秒（四任务）|
|---|---|---:|---:|---:|---:|
|42|random|42.89|36.06|70.71|0.000|
|42|coreset|40.40|33.99|67.36|11.009|
|43|random|37.80|32.13|67.80|0.000|
|43|coreset|38.45|32.86|74.81|8.473|
|44|random|38.52|33.36|71.02|0.000|
|44|coreset|39.81|34.11|71.90|9.830|

成本来自真实计时，环境列含构建/关闭而不包括两 cell 间全部模型准备开销；选样秒已包含在训练调用中，不能再相加。完整开始/结束时间和命令见原 commands.log。

## 执行与验收证据

远端队列：`/root/mlr_ifa/experiment_workspaces/Coreset_paper_GOAL_three_seed_20261017`；产物：`/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed{42,43,44}_20261017`。原始源码 commit `a721807f2f7fa7f9833dcc85c592d78e9accd00b`，执行补丁/冻结源码/配置/模态/官方权重与输入哈希随证据保存。

真实队列和六组执行命令及退出码原文如下（摘录 START/EXIT）：

```text
queue\commands.log
2026-10-17T23:17:34+08:00 START wait_predecessor /data/mlr_ifa/env/bin/python -u /root/mlr_ifa/experiment_workspaces/Coreset_paper_GOAL_three_seed_20261017/admission.py 
2026-10-18T03:19:43+08:00 EXIT wait_predecessor 0
2026-10-18T03:19:44+08:00 START configure_42 python -u -m coreset_exp.configure 
2026-10-18T03:20:28+08:00 EXIT configure_42 0
2026-10-18T03:20:28+08:00 START suite_audit_42 python -u /root/mlr_ifa/experiment_workspaces/Coreset_paper_GOAL_three_seed_20261017/audit_suite.py 
2026-10-18T03:20:43+08:00 EXIT suite_audit_42 0
2026-10-18T03:20:43+08:00 START preflight_42 torchrun --nproc_per_node=1 --master_port=39342 -m coreset_exp.preflight 
2026-10-18T03:21:24+08:00 EXIT preflight_42 0
2026-10-18T03:21:24+08:00 START supervise_42 bash /root/mlr_ifa/experiment_workspaces/Coreset_paper_GOAL_three_seed_20261017/seed42/source/coreset_exp/supervise.sh 
2026-10-18T07:12:50+08:00 EXIT supervise_42 0
2026-10-18T07:12:51+08:00 START configure_43 python -u -m coreset_exp.configure 
2026-10-18T07:13:09+08:00 EXIT configure_43 0
2026-10-18T07:13:09+08:00 START suite_audit_43 python -u /root/mlr_ifa/experiment_workspaces/Coreset_paper_GOAL_three_seed_20261017/audit_suite.py 
2026-10-18T07:13:25+08:00 EXIT suite_audit_43 0
2026-10-18T07:13:25+08:00 START preflight_43 torchrun --nproc_per_node=1 --master_port=39343 -m coreset_exp.preflight 
2026-10-18T07:14:01+08:00 EXIT preflight_43 0
2026-10-18T07:14:01+08:00 START supervise_43 bash /root/mlr_ifa/experiment_workspaces/Coreset_paper_GOAL_three_seed_20261017/seed43/source/coreset_exp/supervise.sh 
2026-10-18T11:03:13+08:00 EXIT supervise_43 0
2026-10-18T11:03:14+08:00 START configure_44 python -u -m coreset_exp.configure 
2026-10-18T11:03:32+08:00 EXIT configure_44 0
2026-10-18T11:03:32+08:00 START suite_audit_44 python -u /root/mlr_ifa/experiment_workspaces/Coreset_paper_GOAL_three_seed_20261017/audit_suite.py 
2026-10-18T11:03:49+08:00 EXIT suite_audit_44 0
2026-10-18T11:03:49+08:00 START preflight_44 torchrun --nproc_per_node=1 --master_port=39344 -m coreset_exp.preflight 
2026-10-18T11:04:25+08:00 EXIT preflight_44 0
2026-10-18T11:04:25+08:00 START supervise_44 bash /root/mlr_ifa/experiment_workspaces/Coreset_paper_GOAL_three_seed_20261017/seed44/source/coreset_exp/supervise.sh 
2026-10-18T14:56:40+08:00 EXIT supervise_44 0
2026-10-18T14:56:40+08:00 START aggregate python -u /root/mlr_ifa/experiment_workspaces/Coreset_paper_GOAL_three_seed_20261017/aggregate_three_seeds.py 
2026-10-18T14:56:40+08:00 EXIT aggregate 0
seed42\commands.log
2026-10-18T03:21:25+08:00 START export torchrun --nproc_per_node=1 --master_port=39342 -m coreset_exp.export 
2026-10-18T03:27:57+08:00 EXIT export 0
2026-10-18T03:27:57+08:00 START train_random torchrun --nproc_per_node=1 --master_port=39342 -m coreset_exp.train random 
2026-10-18T04:11:05+08:00 EXIT train_random 0
2026-10-18T04:11:05+08:00 START train_coreset torchrun --nproc_per_node=1 --master_port=39342 -m coreset_exp.train coreset 
2026-10-18T04:51:44+08:00 EXIT train_coreset 0
2026-10-18T04:51:44+08:00 START evaluate_random python -u -m coreset_exp.evaluate random 
2026-10-18T06:03:16+08:00 EXIT evaluate_random 0
2026-10-18T06:03:16+08:00 START evaluate_coreset python -u -m coreset_exp.evaluate coreset 
2026-10-18T07:11:26+08:00 EXIT evaluate_coreset 0
2026-10-18T07:11:26+08:00 START summarize python -u -m coreset_exp.summarize 
2026-10-18T07:12:39+08:00 EXIT summarize 0
seed43\commands.log
2026-10-18T07:14:02+08:00 START export torchrun --nproc_per_node=1 --master_port=39343 -m coreset_exp.export 
2026-10-18T07:20:19+08:00 EXIT export 0
2026-10-18T07:20:19+08:00 START train_random torchrun --nproc_per_node=1 --master_port=39343 -m coreset_exp.train random 
2026-10-18T07:58:22+08:00 EXIT train_random 0
2026-10-18T07:58:22+08:00 START train_coreset torchrun --nproc_per_node=1 --master_port=39343 -m coreset_exp.train coreset 
2026-10-18T08:37:05+08:00 EXIT train_coreset 0
2026-10-18T08:37:05+08:00 START evaluate_random python -u -m coreset_exp.evaluate random 
2026-10-18T09:45:39+08:00 EXIT evaluate_random 0
2026-10-18T09:45:39+08:00 START evaluate_coreset python -u -m coreset_exp.evaluate coreset 
2026-10-18T11:01:37+08:00 EXIT evaluate_coreset 0
2026-10-18T11:01:37+08:00 START summarize python -u -m coreset_exp.summarize 
2026-10-18T11:03:01+08:00 EXIT summarize 0
seed44\commands.log
2026-10-18T11:04:26+08:00 START export torchrun --nproc_per_node=1 --master_port=39344 -m coreset_exp.export 
2026-10-18T11:11:02+08:00 EXIT export 0
2026-10-18T11:11:02+08:00 START train_random torchrun --nproc_per_node=1 --master_port=39344 -m coreset_exp.train random 
2026-10-18T11:49:46+08:00 EXIT train_random 0
2026-10-18T11:49:46+08:00 START train_coreset torchrun --nproc_per_node=1 --master_port=39344 -m coreset_exp.train coreset 
2026-10-18T12:29:49+08:00 EXIT train_coreset 0
2026-10-18T12:29:49+08:00 START evaluate_random python -u -m coreset_exp.evaluate random 
2026-10-18T13:41:48+08:00 EXIT evaluate_random 0
2026-10-18T13:41:48+08:00 START evaluate_coreset python -u -m coreset_exp.evaluate coreset 
2026-10-18T14:54:48+08:00 EXIT evaluate_coreset 0
2026-10-18T14:54:48+08:00 START summarize python -u -m coreset_exp.summarize 
2026-10-18T14:56:28+08:00 EXIT summarize 0
```

最终自动执行 `final_queue_audit.py`，FINAL_QUEUE_ACCEPTANCE=PASS，queue.exit=0，wrapper.exit=0。手动只读打包命令 `/data/mlr_ifa/env/bin/python /root/mlr_ifa/experiment_workspaces/Coreset_paper_GOAL_three_seed_20261017/package_evidence.py` 退出0：941文件、24,799,387字节，下载后逐项核验。

本地及发布目录独立复算命令（退出0）：

```sh
python -X utf8 recompute_final.py
```

脚本仅用标准库：复算 1,200 条真实结果、120 次尝试、所有矩阵/精确指标/样本SD，检查 epoch/step/入库清单/配置/源码/加载凭证与包内 941 文件哈希，并检查输入前后未变。不导入训练框架、不加载权重、不调用环境。服务器已核验大载荷；本地依据下载的哈希和原验收凭证核查，不冒称本地重新加载权重。

首次预检和历史失败材料若在包内，按原文件保留；不把已解决的历史材料误写为正式失败，也不覆盖日志。此次正式三 seed 队列无需训练或评估恢复。

## 解释边界

- 三 seed 跨套件探索性比较；GOAL 三 seed 在本轮结果产生前指定。这里只作描述性结论，不宣称统计显著或普遍因果。
- 与论文明确参数对齐，但官方预训练 `cfg.policy.use_state` 缺失仍记 unknown；本轮实际三模态为 True，状态参数完整加载。
- warmup500、逐更新调度调用、ep99最终权重等公开细节缺口已明确补全，不能声称精确恢复作者训练轨迹；seed42来自官方脚本，正文未列具体seed。
- RTX3090 与论文 A100 不同；旧 GOAL state_v2 的 RNG/IFA/导出细节不完全等价，仅保留历史参考。
- 评估全局动作 RNG 每阶段重置，原 argmax GMM 分量后 Normal 采样；未逐轨迹隔离。初态、seed、配置对应，但两模型消耗随机数的顺序仍可能分岔。
- Coreset 改变 buffer 成员后 IFA 配对可能间接变化，结果属于完整 Coreset＋IFA 方案，不能隔离为单纯覆盖因果。

本轮完成后停止并取消 GOAL 进度定时检查。只发布代码、报告、配置、哈希、逐 episode 结果及技术尝试；不发布权重、演示、latent 或 buffer 载荷。
