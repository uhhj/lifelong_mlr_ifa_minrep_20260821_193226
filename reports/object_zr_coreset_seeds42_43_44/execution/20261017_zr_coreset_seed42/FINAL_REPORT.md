# Seed42 指令零响应残差单组实验最终报告

状态：COMPLETED。服务器于2026-10-17 17:45:33（北京时间）完成退出验收；本地独立复算PASS。17:47复查无残留训练或评估进程，GPU利用率0%。

本轮仅新增Coreset＋IFA＋指令零响应残差一组，复用首次seed42 Coreset＋IFA已有200条基线；未重训或追加基线评估。

## 完整成功率矩阵

各单元为基线 → 本轮，单位%；每个单元各20条记录。

|完成阶段|Task6|Task7|Task8|Task9|
|---|---:|---:|---:|---:|
|6|30 → 35|—|—|—|
|7|70 → 65|85 → 75|—|—|
|8|20 → 45|95 → 100|100 → 100|—|
|9|30 → 55|90 → 100|100 → 100|45 → 75|

## 指标独立复算

|指标|基线%|本轮%|差值（百分点）|本轮精确分数|
|---|---:|---:|---:|---|
|AUC|68.125000|79.166667|+11.041667|19/24|
|FWT|65.000000|71.250000|+6.250000|57/80|
|NBT|-5.833333|-15.000000|-9.166667|-3/20|
|OldFinal|73.333333|85.000000|+11.666667|17/20|
|FinalAll|66.250000|82.500000|+16.250000|33/40|

预定四项推进门槛全部满足：AUC严格升高、FWT不降低、按合同定义NBT更低、OldFinal不降低。只说明本seed通过预定筛选，不自动扩seed。

提升主要见Task6后期（阶段8、9各+25百分点），Task9初学/最终+30百分点；Task7最终+10百分点。也存在下降：阶段7的Task6下降5百分点、Task7下降10百分点。Task8初学与最终均为100%，没有额外收益空间。NBT的负值表示后续成绩超过本任务刚学完的锚点，不应简单理解为通常定义的负迁移。

## 工程与结构验收

400/400 epoch、105500/105500有效优化步、4个ep99最终checkpoint严格加载PASS；新200/200 episode和20/20 vector attempts，零技术异常、无正常失败重试。无额外rollout，隔离集成总计4次optimizer调用、0环境。

44次零响应结构检查全部通过；阶段7/8/9最终全量旧buffer分别781/1521/2301窗口。全程旧门最大2.7248437085153343e-7，新增通道累计贡献最大9.466288730664019e-8，均低于1e-6；选中GMM分量一致，旧U/V与门保持不变。新增分支U均非零。该结论是同一共享模型上的通道隔离，不意味着共享模型训练不能改变旧策略。

6136项原输入哈希均无变化，冻结源码、配置、原合同、外置恢复适配器均一致；Coreset成员/顺序/载荷/配额及IFA配对/reference与父组一致。正式恢复前的整数/字符串审计键误比较退出1已保留，实际未发生优化更新；外置适配器仅规范元数据键，原科学源码和合同未改。

真实执行命令与退出码：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 /data/mlr_ifa/experiment_workspaces/ZR_coreset_OBJECT_seed42_20261017/engineering_recovery/train_metadata_adapter.py # exit 0
python -u -m zrg_exp.evaluate # exit 0
python -u -m zrg_exp.finalize # exit 0
python execution/20261017_zr_coreset_seed42/independent_local_acceptance.py # local exit 0
```

原pipeline.exit=1、原supervisor.exit=1继续保留；恢复resume.exit=0、supervisor.exit=0，PROCESS_EXIT_ACCEPTANCE=PASS。

## 时间与资源

训练程序墙钟54.80分钟，其中epoch计算48.54分钟；闭环评估73.23分钟。恢复流程从15:35:44至17:45:33约129分49秒。结构诊断累计10.78秒。

真实10-lane get_action总调用9096次，加权平均423.74毫秒/批调用（包含原完整观测前端，非单轨迹独立时延）；Task9 cell均值423.11毫秒、cell内p95约427.98毫秒。没有等参数/同轨迹无残差时延对照，不能把全部时延差归因于残差。

新增残差参数18784个FP32，75136字节；门参数16384字节；buffer载荷197344659字节，与父组相同。四个最终模型共346557352字节，仅保留服务器。训练PyTorch峰值allocated约541.95MiB（并非整卡进程总显存）。

## 解释边界

这是单个探索性seed，不支持统计显著性或跨seed稳定性结论。额外18784参数及共享优化轨迹变化混在模块效应中，尚不能证明零响应设计优于等参数普通残差。原动作全局RNG未逐轨迹隔离，相同seed/初态不等于全过程噪声配对。官方预训练state元数据仍unknown；本轮明确state=True，沿用父实现的调度补全与RTX3090环境。没有重跑基线、改配额、按结果选择checkpoint、扩seed或push。

## 产物

服务器产物：/data/mlr_ifa/experiments/ZR_coreset_OBJECT_seed42_20261017
服务器隔离源码与恢复证据：/data/mlr_ifa/experiment_workspaces/ZR_coreset_OBJECT_seed42_20261017
本地final_evidence/：136份轻量记录、训练与结构日志、成员清单、基线/新组逐episode记录、代码及失败证据；final_acceptance/：最终严格加载、汇总及退出验收。未下载checkpoint/latent/buffer载荷或示范。
本地LOCAL_FINAL_ACCEPTANCE.json与independent_local_acceptance.py为独立复算输出与脚本；两份tar.gz快照保留下载时原样，未覆写早期日志。

完整验收后按授权删除seed42定时任务，不启动下一轮。
