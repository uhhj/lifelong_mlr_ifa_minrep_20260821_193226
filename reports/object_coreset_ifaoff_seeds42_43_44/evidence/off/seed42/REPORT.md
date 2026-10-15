# Seed42 Random vs Coreset

仅一个训练seed；按论文描述启用状态模态，官方checkpoint原始模态配置缺失时仍记未知。

| 方法 | AUC (%) | FWT (%) | NBT (%) |
|---|---:|---:|---:|
| random | 59.791667 | 63.750000 | 7.777778 |
| coreset | 80.000000 | 78.750000 | -2.222222 |

400正式闭环；每组400epoch、105500步、4个ep99最终checkpoint。
状态：数值与输入审计通过，仍需独立进程退出和权重加载验收。
限制：单seed不能判断统计显著性；RTX3090与论文A100不同；预训练模态元数据、warmup/调度及最终checkpoint选择细节未公开；IFA随buffer变化，不能独立归因于几何覆盖。
不扩展seed，不更新旧实验指标，不执行后续调参。

最终验收：COMPLETED；pipeline.exit=0，无训练/评估残留进程，8个最终checkpoint实际张量完整加载通过。
