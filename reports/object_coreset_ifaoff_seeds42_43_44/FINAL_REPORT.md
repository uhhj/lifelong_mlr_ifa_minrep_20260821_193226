# LIBERO-OBJECT：三seed IFA关闭对照最终报告

状态：**COMPLETED**。六组共2400个epoch、633000步、24个最终epoch99 checkpoint、1200条新正式闭环记录，已通过服务器终验和本地逐记录独立复算。旧IFA开启1200条仅作只读对照。

实验完成后已删除自动化`ifa-seed`，不再启动下一轮检查或实验。

## 真实执行与验收

队列于2026-10-14 12:53:41 CST开始，2026-10-15 00:36:47完成，queue.exit=0；三个seed的pipeline与独立acceptance均退出0。最后一组评估结束后未追加训练或rollout。

实际主要命令（各seed及arm由既有队列设置，完整命令/退出码保存在归档内commands.log）：

```sh
/data/mlr_ifa/env/bin/python /data/mlr_ifa/experiment_workspaces/Coreset_validation_20261014/original_protocol/launch.py
/data/mlr_ifa/env/bin/torchrun --standalone --nnodes=1 --nproc_per_node=1 -m coreset_exp.train random
/data/mlr_ifa/env/bin/torchrun --standalone --nnodes=1 --nproc_per_node=1 -m coreset_exp.train coreset
/data/mlr_ifa/env/bin/python -u -m coreset_exp.evaluate random
/data/mlr_ifa/env/bin/python -u -m coreset_exp.evaluate coreset
/data/mlr_ifa/env/bin/python /data/mlr_ifa/experiment_workspaces/Coreset_validation_20261014/original_protocol/final_audit.py
```

final_audit.py退出0；本地`python -X utf8 execution/20261014_ifaoff_original_protocol/final_evidence/verify_local.py`退出0。18项软件测试、纯spawn及4次隔离GPU optimizer调用/seed已在正式运行前通过。未重复预检。24个checkpoint已有独立实际加载验收，并在最终归档前再次核验字节哈希。

每个seed任务6模型实际张量及初始化RNG跨arm、与相应IFA开启起点一致；相应arm的buffer成员、顺序、载荷核验一致。冻结source/config/input/feature/modal/orchestration及原对照输入保护检查通过。新正式1200条对应120个完成的10-lane vector attempts，零技术异常、零额外重试；60次环境构造均完成并关闭。

早前SSH握手失败属于监控连接问题：服务器恢复访问后查明队列早已正常完成，未重启实验。重连时nvidia-smi无法连接驱动；本轮最终检查仅作只读哈希/记录验收，GPU训练和模型加载证据来自此前实际完成的进程，不声称本次又通过GPU测试。

## IFA关闭结果

指标以百分数表示，AUC/FWT越高越好，NBT越低越好；差值为百分点。

|seed|方法|AUC|FWT|NBT|
|---|---|---:|---:|---:|
|42|random|59.791667|63.750000|7.777778|
|42|coreset|80.000000|78.750000|-2.222222|
|43|random|81.562500|83.750000|4.444444|
|43|coreset|85.104167|81.250000|-6.666667|
|44|random|85.104167|87.500000|5.000000|
|44|coreset|80.833333|77.500000|-6.388889|

|条件/方法|AUC均值±样本SD|FWT均值±样本SD|NBT均值±样本SD|
|---|---:|---:|---:|
|on_random|76.354167 ± 9.246222|77.083333 ± 9.381942|0.925926 ± 5.161985|
|on_coreset|82.916667 ± 13.438711|80.000000 ± 13.050383|-5.277778 ± 5.023095|
|off_random|75.486111 ± 13.706660|78.333333 ± 12.767961|5.740741 ± 1.785861|
|off_coreset|81.979167 ± 2.738217|79.166667 ± 1.909407|-5.092593 ± 2.489691|
|on_C_minus_R_pp|6.562500 ± 7.848528|2.916667 ± 3.818813|-6.203704 ± 10.032099|
|off_C_minus_R_pp|6.493056 ± 12.503616|0.833333 ± 12.829004|-10.833333 ± 0.734931|
|interaction_off_minus_on_pp|-0.069444 ± 16.103888|-2.083333 ± 16.072751|-4.629630 ± 10.139206|

SD采用ddof=1；C−R使用同seed配对差值计算SD。交互=(C−R)IFA关闭−(C−R)IFA开启。

|seed|IFA开启 AUC C−R|IFA关闭 AUC C−R|交互差值|
|---|---:|---:|---:|
|42|1.979167|20.208333|18.229167|
|43|15.625000|3.541667|-12.083333|
|44|2.083333|-4.270833|-6.354167|

## 全部任务矩阵

行是训练完成阶段6–9；列为任务6–9。每个有效单元20条，未见任务记为—。

### seed42 / IFA on / random

|阶段|Task6|Task7|Task8|Task9|
|---|---:|---:|---:|---:|
|6|30|—|—|—|
|7|45|95|—|—|
|8|10|100|95|—|
|9|10|100|100|45|

### seed42 / IFA on / coreset

|阶段|Task6|Task7|Task8|Task9|
|---|---:|---:|---:|---:|
|6|30|—|—|—|
|7|70|85|—|—|
|8|20|95|100|—|
|9|30|90|100|45|

### seed42 / IFA off / random

|阶段|Task6|Task7|Task8|Task9|
|---|---:|---:|---:|---:|
|6|30|—|—|—|
|7|20|100|—|—|
|8|15|95|100|—|
|9|15|95|95|25|

### seed42 / IFA off / coreset

|阶段|Task6|Task7|Task8|Task9|
|---|---:|---:|---:|---:|
|6|30|—|—|—|
|7|50|100|—|—|
|8|25|100|100|—|
|9|35|100|100|85|

### seed43 / IFA on / random

|阶段|Task6|Task7|Task8|Task9|
|---|---:|---:|---:|---:|
|6|60|—|—|—|
|7|55|95|—|—|
|8|40|75|95|—|
|9|55|85|100|80|

### seed43 / IFA on / coreset

|阶段|Task6|Task7|Task8|Task9|
|---|---:|---:|---:|---:|
|6|60|—|—|—|
|7|95|95|—|—|
|8|85|100|100|—|
|9|90|90|100|100|

### seed43 / IFA off / random

|阶段|Task6|Task7|Task8|Task9|
|---|---:|---:|---:|---:|
|6|60|—|—|—|
|7|55|95|—|—|
|8|50|100|95|—|
|9|50|90|90|85|

### seed43 / IFA off / coreset

|阶段|Task6|Task7|Task8|Task9|
|---|---:|---:|---:|---:|
|6|60|—|—|—|
|7|95|85|—|—|
|8|85|65|100|—|
|9|75|95|100|80|

### seed44 / IFA on / random

|阶段|Task6|Task7|Task8|Task9|
|---|---:|---:|---:|---:|
|6|50|—|—|—|
|7|65|95|—|—|
|8|65|95|95|—|
|9|50|85|100|90|

### seed44 / IFA on / coreset

|阶段|Task6|Task7|Task8|Task9|
|---|---:|---:|---:|---:|
|6|50|—|—|—|
|7|70|100|—|—|
|8|40|100|100|—|
|9|40|100|100|95|

### seed44 / IFA off / random

|阶段|Task6|Task7|Task8|Task9|
|---|---:|---:|---:|---:|
|6|50|—|—|—|
|7|55|100|—|—|
|8|40|95|100|—|
|9|40|95|95|100|

### seed44 / IFA off / coreset

|阶段|Task6|Task7|Task8|Task9|
|---|---:|---:|---:|---:|
|6|50|—|—|—|
|7|95|90|—|—|
|8|60|95|90|—|
|9|45|80|95|80|

## 解释与边界

关闭IFA时Coreset的AUC在seed42、43提升，seed44下降；三seed平均仍为正，但变异很大。这个结果说明在本协议和这些seed中，Coreset的收益可以在没有IFA梯度项时出现；不能将原收益全部归因于IFA配对改变，也不能宣称它完全独立于IFA。

seed44关闭IFA后，Coreset最终任务6/7/8/9为45/80/95/80%，Random为40/95/95/100%。Coreset虽然Task6略高，但Task7、9更低，首次学习成绩FWT也低10个百分点；更低的NBT不能单独视为总体更好，因为它相对于各自刚学完的成绩计算。

仅3个探索性训练seed；seed43/44最初扩展是在看到seed42后授权。初态0–19与旧对照相同，非独立新测试集；每cell仅20条，成功率分辨率5个百分点。沿用原全局动作RNG，未隔离每条轨迹随机流，保留轨迹依赖与执行噪声，不作统计显著性或普遍因果结论。论文预训练模态元数据unknown、论文A100与本轮RTX3090、学习率调度/warmup工程补全等边界不消失。已取消9000条复评保持取消且未混入结果。

## 耗时与产物

队列墙钟耗时11.7183小时（训练、评估、预检与队列内验收）；各命令耗时见COMMAND_COSTS.json。

远端汇总：`/data/mlr_ifa/analyses/Coreset_IFAoff_original_protocol_20261014`；checkpoint保留在`/data/mlr_ifa/experiments/Coreset_IFAoff_OBJECT_seed{42,43,44}_20261014/{random,coreset}`。本地仅下载轻量证据，未下载checkpoint、未推送远端。

本目录包括FINAL_ACCEPTANCE.json、COMPARISON.json、LOCAL_FINAL_RECOMPUTATION.json、POST_EXIT_INTEGRITY.json、LIGHT_FILE_HASHES.json和light_evidence.tar.gz。归档内每个文件已按服务器清单核验SHA256，全部12张矩阵、精确三指标及均值/样本SD/交互已本地独立复算。
