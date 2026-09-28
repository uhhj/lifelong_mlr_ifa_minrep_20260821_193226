# E1：已有闭环证据的同初态配对分析

本目录发布已完成的回顾性描述分析。输入固定为发布提交 `eb15698ae8ee356f6098b9777e5c7dad21450b0e` 中的200条正式闭环记录；实际Git版本和11份输入的Git内容哈希已核验。该输入提交不随此次分析发布提交变化。

附带15项合成软件测试通过；真实分析退出码0，验收PASS。472份轻量run文件前后哈希不变。两种视角各120条配对，共240行，对应180个去重跨阶段配对，均复用原200个episode。没有新增训练、环境执行、checkpoint加载或输入修补。

## 结果与文件

- [完整执行及结果说明](results/OPERATOR_REPORT.md)：真实命令、退出码、12行完整统计、任务6/7/8/9结果历史和解释边界。
- [附件脚本原始报告](results/REPORT.md)
- [12行配对汇总](results/paired_summary.csv)：SS/SF/FS/FF、保留/丢失率、结果翻转、净变化与SS子集步数变化。
- [240行配对明细](results/paired_transitions.csv)、[80行初态历史](results/initial_state_histories.csv)、[历史聚合](results/history_summary.json)
- [NBT分解](results/nbt_decomposition.json)、[原指标复算](results/metrics_recomputed.json)、[成功率矩阵复算](results/success_matrix_recomputed.csv)
- [分析验收与输入哈希](results/audit_acceptance.json)、[实际来源](source_provenance.json)、[执行命令](execution_commands.jsonl)、[只读验证](final_read_only_verification.json)
- [未修改的附件分析工具](tools/MLR_E1_Paired_Audit/analyze_paired_closed_loop.py)、[软件测试](tools/MLR_E1_Paired_Audit/test_paired_closed_loop.py)、[本次测试日志](logs/software_tests.log)
- [本次发布文件哈希](publication_manifest.json)

按任务内、任务间两级平均，观测成功→失败贡献为7.222222个百分点，失败→成功贡献为6.111111个百分点，两者相减为原NBT的1.111111个百分点。任务6从刚学完到最终阶段SF=2、FS=3；任务8为SF=3、FS=2。任务7的20个初态三次观察均成功；任务9只有一次观察，其遗忘、恢复和后续稳定性为NA。

这些是单次执行的观测结果历史。相同初态与seed不保证逐episode随机动作流相同；没有额外对照或行为事件，不能据此证明回放伤害、定位动作阶段或生成训练采样权重。没有自动选择下一轮超参数或启动新实验。

## 发布与原分析记录的区别

24份原分析文件逐字节复制，历史日志和报告中的本地路径、`pushed_remote=false`及“不push”表述保持原样，表示当时只读分析的执行范围。用户随后另行授权本次远端发布；本次操作只发布这些工具和结果，没有再次执行真实分析、训练或评估。

原始 `execute_audit.py`、`summarize_histories.py` 是本地执行过程的留档，包含当时目录布局。若以后需要再次进行纯软件分析，应在新的输出目录调用附件标准库脚本，不运行旧训练driver。此目录不包含模型权重、演示或latent数据。
