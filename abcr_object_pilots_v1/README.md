# OBJECT 前置实验 P1–P3

依据用户本轮四项方案，全部改用 OBJECT；用户随后明确暂不执行 P4。
只跑 CPU 数据/特征导出和统计，不训练、不运行 QP。

本轮使用独立五轨迹/旧任务诊断池，包含基础任务；这不是正式 MLR 的实际缓存。
当前任务固定十轨迹，原始 HDF5 全时间步。原生 CompletePolicy flags 来自
`libero/configs/policy/complete_policy.yaml`，而非缺少这些字段的预训练 checkpoint cfg。
同图重建 pre-FiLM 与 full-H，关闭随机增强，严格加载 checkpoint 权重。

原始观测在 action[t] 执行之后：全部70轨迹实测 obs[t] 的关节与 states[t+1]
逐位一致。未来动作从 t+1 起，物体状态同样对齐到 t+1，末尾不填充。
这与历史 B 按其旧协议用同下标动作不同；历史结果不覆盖。

`common.PROTOCOL` 固定标准化、beta网格、主beta、分位数、覆盖率分母、P2混杂对照、
P3共同完整窗口集、轨迹统计与解释边界。分位数“造出两端”不算物理真实性认证。

顺序运行：prepare → encode --mode language → encode --mode vision → analyze → report。
各命令接受 `--out`；prepare/encode 另需 `--repo`。使用原 `.migration/activate.sh` 环境，
本独立源码路径叠加到 PYTHONPATH，HF离线，CUDA_VISIBLE_DEVICES为空，BLAS/torch2线程。
每轨迹及每阶段有输入/代码/输出SHA绑定，可验证后续做；不要并行启动第二实例。
`python -m unittest abcr_object_pilots_v1.tests` 为CPU小型契约测试。

大规模中间数组保留服务器，不提交 Git；报告、逐轨迹/检索统计和代码提交研究分支。
`formal_training_resumed=false`。
