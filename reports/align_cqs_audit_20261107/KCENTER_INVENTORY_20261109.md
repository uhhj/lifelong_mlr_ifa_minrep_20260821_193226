# 已有 K-Center 对照清点

2026-11-09，本地只读核对；0新增训练/前向/环境调用。结论：已有纯 Greedy K-Center 的闭环性能对照，也有与 RA 的选样几何对照；不能说项目没有 K-Center 对比。

## 已有完整训练和正式闭环

历史方法名 `coreset` 对应 visual/action Greedy K-Center，使用视觉与动作各0.5的冻结距离，保留半数窗口，不含 RA 的保护随机锚点与后续交换。Random 与 Coreset 两组都保留 IFA。

| 历史组 | seed | Random AUC | K-Center AUC |
|---|---|---:|---:|
| OBJECT 三seed扩展 | 42/43/44 | 76.354167 | 82.916667 |
| GOAL 三seed | 42/43/44 | 79.687500 | 76.666667 |

两套件各组均有完整训练及1200条正式回合记录（两种方法×三个seed×200），相应最终报告保存验收来源、逐seed和任务矩阵。这是已完成的历史证据，不能改称未做。OBJECT另外还有五seed40–44首轮汇总和seed43重复版本；不得为当前对比混用版本或挑seed。此处列42/43/44只是标明上述两份历史三seed报告的原口径，不替换当前主表。

来源：
- `execution/20261013_coreset_object/ADJUSTED_PROTOCOL.md`
- `execution/20261013_coreset_extension/EXTENSION_PROTOCOL.md`、`FINAL_REPORT.md`
- `execution/20261017_coreset_goal_three_seed/launch_evidence/queue/PROTOCOL.md`、`FINAL_REPORT.md`
- `execution/20261017_five_seed_summary/RESULTS.json`

## 已有 Random / K-Center / RA 选样比较

`execution/20261107_ra_selection_visualization/DISTANCE_REPORT.md` 已完成 OBJECT/GOAL、seed41/42/43、Task6–9 共24个案例的原始距离复算：同候选总体、同配额、原始冻结距离。包含最近邻距离ECDF、覆盖半径、轨迹分配和辅助t-SNE展示。

RA相对K-Center的平均最近邻距离在OBJECT/GOAL分别增加0.79%/0.61%，最大覆盖半径两套件平均均增加4.71%；RA保留全部指定随机锚点。该证据说明锚点保留与几何覆盖的权衡，不能将其当作新的闭环性能比较，也不能宣称RA所有几何指标胜过K-Center。

## 当前范围边界

已有K-Center闭环是早期MLR+IFA训练目标下的选样对照，不等于“在最新完整S中只把RA替换成K-Center，其余ALIGN/执行专家监督/保持/门控全相同”的最终目标对照。此前审稿评估所说同最终目标下的纯K-Center对照应明确限定为后一种，不能让人误以为此前完全没有做过K-Center。

当前RA/ALIGN主表常用41/42/43，不能直接把本表42/43/44的均值相减冒充同seed配对比较。正式编表应从已冻结版本中明确匹配seed和协议。本次没有重组或替换既有结果。
