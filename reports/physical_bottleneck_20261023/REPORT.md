# 物理瓶颈定位：PARTIAL

观察：2026-10-23 15:18 CST。用户授权按同单位诊断→读出→空间patch有限对照，A通过后才进入B。旧A失败完整保留，所有旧队列不恢复。

D0实际完成：50个原native选中checkpoint，train/validation/test统一毫米误差，690批前向，0优化/环境/服务器写入。same_unit_readonly_20261023_150704及LOCAL_AUDIT核对全部SHA、分折身份/数量和原测试复现PASS；新训练集指标是实际前向观测，本地核对不冒充第二次模型重执行。

P3各折指标的等权平均（不是合并样本分位数）：

|任务|分区|物体－末端 median/P90 mm|物体－目标 median/P90 mm|
|---|---|---|---|
|6|训练|5.14/9.68|6.91/12.20|
|6|测试|11.12/30.16|21.43/54.12|
|7|训练|4.76/9.34|6.90/12.00|
|7|测试|14.21/27.61|24.45/46.31|

这确认当前表示/读出能在已见演示上拟合较准确位置，未见演示存在明显泛化差距。它不证明patch一定能解决，也不排除进一步优化收益；不能把重叠训练折视为独立样本。

新v1合同c26f09ff41b912935a85e6eb92e2eeda62ff76f8fc1e0ccae8c1d5cd3d75c8dc，root=/data/mlr_ifa/experiment_workspaces/PHYSICAL_BOTTLENECK_OBJECT41_20261023_v1，controller120819已实际启动。readout_admission exit0，P0/M/Q各3次隔离优化与完整恢复核对，正式从原独立初始化开始。15:18快照readout正常、GPU worker120854、冻结源/原合同hash无变化；尚未完成新A，不判门、不启动B。

有限50fits各200epochs：P0/M预算50/100/200、M/Q仅改变帧内融合；G/S仅改变相机输入为重复全局token或真实patch网格，架构/批量/FP32/训练预算相同。先D1全完成，再导出patch及自身GPU准入，再D2。200epochs实际预算与原A不同，单独报告。原50结果重现检查保留。10/25mm、信息改善和动态门槛不放宽。

这些十demo外层结果已在上一轮分析使用，新有限对照明确是探索性；不能宣称新独立确认。B需完整A本地独立验收通过后实施及准入，不能用placeholder启动。容量起步data root8.738GB/system普通6.645GB；新产物规划≤3.3GiB，分别保留2/1GiB，不删除旧资产或新增付费。


## User termination 2026-10-23 15:31 CST
Confirmed controller120819 and worker120854 share the dedicated session. SIGTERM stopped both; no remaining PBD_ROOT process. All checkpoint/log/prediction/recovery assets retained, 139 checkpoint SHA recorded. No source overwrite or experimental acceptance. Replaced by the new user-specified A1/A2/A3 plan. Automation b deleted.
