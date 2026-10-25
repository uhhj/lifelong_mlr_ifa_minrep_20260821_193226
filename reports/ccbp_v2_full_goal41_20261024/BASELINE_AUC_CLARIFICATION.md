# V的AUC与哪一条基线比较

2026-10-25：针对用户“V不是高于基线的AUC了吗”的核对。只读已有本地最终报告，无服务器命令、前向、更新或环境。

|比较对象|GOAL AUC|V=82.50%的数值差|
|---|---:|---:|
|历史RA-Coreset seed41|78.7500%|+3.7500pp|
|历史ALIGN seed41|83.4375%|−0.9375pp|
|本轮B固定蒸馏|76.2500%|+6.2500pp|
|本轮C专家成本|83.1250%|−0.6250pp|
|本轮N新路由无锚|78.7500%|+3.7500pp|

如果“基线”指RA-Coreset seed41，V的AUC数值确实更高，这个正向结果不能因预登记条件未过而抹去。此前简短结论强调V未优于C的Task8/9，未同时列出RA基线，容易让人误解为V低于所有基线；这里明确更正表达。若指ALIGN seed41，V的AUC略低。

历史RA/ALIGN采用globalRNG评估，本轮采用配对逐轨迹Normal噪声，不能把数值差当作同协议配对因果提升。历史GOAL三seed均值RA81.979167%、ALIGN81.111111%，也不能与当前单seed82.50%当作同样本确认比较。本轮四组完整序列评估相互配对，可直接作描述性比较。

“高于RA历史AUC”与“未通过本轮主要探索条件”并不矛盾。预登记条件是V最终Task8>C且Task9>=C；实际80%/80%对85%/90%，未过。V和C的FinalAll同为76.25%，V的OldFinal为75.00%高于C的71.666667%，说明存在保持与新任务表现的取舍，而非V全面无效。停止自动扩展是执行既定协议，不是否定所有正向信号。

来源：

- execution/20261022_align_object_extension_goal_conditional/final_GOAL_41_20261023_052225/REPORT.md
- execution/20261022_align_object_extension_goal_conditional/aggregate_GOAL_20261023_0925/REPORT.md
- execution/20261024_ccbp_v2_full/final_readonly_20261025_054702/LOCAL_FINAL_AUDIT.json

来源SHA和精确差值见BASELINE_AUC_CLARIFICATION_AUDIT.json。未修改冻结协议、最终验收或科学门槛，不恢复已删除监控。
