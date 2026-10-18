# SC-v2 GOAL seed42进度检查

**PARTIAL_EVALUATING；证据时间2026-10-18 21:56:27 CST。**

训练400/400 epoch、76100/76100步，train.exit=0，四个ep99 checkpoint已保存且哈希匹配。评估170/200条（85%）、17/20个向量完成，0技术异常；当前stage9/task8/group1。GPU RTX3090运行正常，pipeline与evaluate进程在运行，评估/审计/pipeline/supervisor退出码尚无。尚未最终验收，不标COMPLETED。

## 已有结果

下表是成功次数/已完成次数，未完成的单元不能作完整成功率或最终指标使用。

|阶段|Task6|Task7|Task8|Task9|
|---|---:|---:|---:|---:|
|6|13/20|—|—|—|
|7|13/20|20/20|—|—|
|8|11/20|20/20|13/20|—|
|9|6/20|20/20|8/10（未完）|未开始|

Task6在最终阶段仅6/20，较刚学完13/20减少7次成功，提示这次温和入库修复仍未消除Task6遗忘；这是本次闭环结果的描述，不能归因到具体交换窗口。原评估动作全局随机流未逐轨迹隔离。整体AUC与最终旧任务均值待完整记录复算，暂不判定双指标门槛。

## 一致性

本轮重新只读核对全部163个冻结源文件、196个合同文件、2381个实际buffer载荷哈希及源feature哈希。Task6–9实际有序成员与离线/预检清单完全一致；替换数5/2/4/6，独立监督时刻495/426/449/981不减少，原示范/终点支持保留，最大覆盖与平均最近距离守住原S0底线。没有发现配置或入库漂移。Task6模型张量一致性证据及初始化状态比较通过；IFA选中任务对仍与原Coreset一致，但实际视觉相似度存在变化。

## 时间估计

按已完成600步向量平均281.44秒、当前向量已执行时间计算，21:56时剩余评估约13.94分钟，预计22:10左右结束；包含最终加载、汇总及退出验收，完成时间仍估计为**今晚22:10–22:30 CST**。不持续守候，原定监控继续负责最终验收及条件扩展。

## 命令与证据

远端（退出0）：

```sh
source /root/mlr_ifa/experiment_workspaces/SCv2_Coreset_GOAL_seed42_20261018/env.sh
/data/mlr_ifa/env/bin/python /tmp/check_scv2_20261018_2155.py
```

本地独立复算（退出0）：

```powershell
python -X utf8 execution/20261018_scv2_goal_seed42/progress_20261018_2155/recompute.py --offline execution/20261018_scv2_offline/evidence
```

MANIFEST.json校验下载文件；LOCAL_CHECK.json复算有序ID、离线约束、400epoch、episode/attempt对应关系及ETA。未重启进程，未调用模型、optimizer或环境，未追加训练或闭环。只有完整工程验收通过且AUC>78.75%、OldFinal>=68.33333333333333%才扩展43/44，当前未启动扩展。发布仅代码、报告、哈希和episode结果，不含权重与数据载荷。
