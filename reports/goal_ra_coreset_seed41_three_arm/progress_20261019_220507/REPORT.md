# GOAL seed41 进度核查：RA评估160/200

状态 **PARTIAL**，证据时间 **2026-10-19T22:06:07.766596+08:00**。三组训练已完成1200完整epoch、228300步。Random和Coreset各200条闭环与8个最终checkpoint严格加载已通过；RA已完成160/200条、16个vector，另1个vector进行中，技术异常0。合计560/600条。服务器评估进程与GPU正常，queue/wrapper尚未退出，本次没有恢复或重启。

冻结源、配置、输入与父基线保护哈希通过；Task6实际模型张量和初始化RNG与父基线一致。Task6–9实际入库数量499/445/452/985，硬锚点249/222/226/492，交换119/107/112/244；所有有序成员与offline逐项一致，全部源/目标载荷哈希通过。四个RA ep99 checkpoint文件哈希与训练凭证一致，但其严格加载及最终退出验收仍待既定流水线完成。

当前RA完成单元如下；缺失单元不填充、不用于计算全程指标。

| 阶段 | 评估任务 | 成功/正式记录 |
|---:|---:|---:|
| 6 | 6 | 14/20 |
| 7 | 6 | 12/20 |
| 7 | 7 | 20/20 |
| 8 | 6 | 7/20 |
| 8 | 7 | 20/20 |
| 8 | 8 | 15/20 |
| 9 | 6 | 7/20 |
| 9 | 7 | 20/20 |

基线全程指标保持：Random AUC69.0625、FWT71.25、NBT2.777778、OldFinal66.666667、FinalAll66.25；Coreset AUC71.875、FWT78.75、NBT12.222222、OldFinal66.666667、FinalAll67.5。完整基线逐episode复算与矩阵见progress_20261019_203008，当前BASELINE_EVIDENCE_REFERENCE.json记录其哈希与发布版本。RA全程指标暂不计算。

按本组16个已完成vector实测速率和父基线速率，剩余4vector加约5分钟验收预计还需18–19分钟；考虑任务差异，预计 **22:20–22:35 CST**。这是吞吐估计，不是完成保证。保持30分钟定时检查，本轮保存证据后结束，不持续守候。

真实命令及退出码：

- `python execution/20261019_ra_coreset_goal_seed41/check_progress.py`：0。
- `python execution/20261019_ra_coreset_goal_seed41/collect_ra_progress.py progress_20261019_220507`：0。
- `python execution/20261019_ra_coreset_goal_seed41/audit_ra_evaluation.py progress_20261019_220507`：0，PASS_PARTIAL_RA_EVALUATING。

远端只读命令与逐文件SHA见COMMAND、RA_READONLY_COMMAND和RA_FROZEN_INPUT_AUDIT证据。本次未训练、未调用环境、未加载模型或改动冻结源码。最终仍须600条/60vector、12CP严格加载、全部矩阵与精确指标、queue/wrapper退出0及FINAL_QUEUE_ACCEPTANCE和本地独立验收。仅发布本轮代码、报告与轻量证据，不含权重、动作载荷、latent、buffer或凭据。

事后探索性扩seed、global轨迹动作RNG未隔离、文件排序/IFA间接变化、pretrain state元数据未知及调度补全的解释边界保持不变。
