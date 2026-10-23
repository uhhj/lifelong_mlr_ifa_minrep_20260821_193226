# 容量诊断实施与工程恢复

只修改两层 Temporal 的 dim_feedforward：512→128。输入512→128投影、d_model128、4heads、2层、P0 frame FFN512、空间394token、三个查询、Head、优化器、FP32、数据/划分/种子/预算均保持。readout_base.py由原始LightReadout直接参数化；models.py仅传递新参数。预期参数量由自身GPU模型核数：P0 483335，G/S各400775。

原512配置的构造顺序、完整state_dict、CPU/CUDA随机数状态与同输入输出均在新准入实际逐元素一致，每组2个未训练前向，共6个，单独计数；不拟合旧基线。FFN128每组3隔离优化器调用，共9个，实际通过精确恢复、最新视觉梯度和长度1–8有限性。P0视觉梯度0，G/S非零。准入使用受控重复全局值验证G/S接口，正式S使用已冻结真实Patch；不将准入等同物理状态可恢复性。

首版root READOUT_CAPACITY_OBJECT41_20261023_v1在准入报告写出时发生输出临时变量a覆盖argparse namespace的AttributeError。snapshot_20261023_173025保留原日志/失败和hash，0正式epoch/更新，无GPU。源码与失败前已执行的9隔离优化器调用、6一致性前向单独记成本，不称首版正式准入PASS。

prepare_engineering02.py保存首版本地源到engineering_attempt_01，只将两个输出临时变量改名。部署新root READOUT_CAPACITY_OBJECT41_20261023_v1_engineering02前核首版exit1、全部冻结hash、无带首版PBD_ROOT进程、无EVENTS和恢复CP、GPU空闲、锁可取得；原远程源和失败未覆盖。科学模型/train/audit完全不变，没有正式训练可恢复，未重试科学负结果。新合同绑定父预算、v2、原输入和失败源/日志hash。

build_capacity.py和prepare_engineering02.py仅为一次性推导记录，部署后禁止重跑。ACTIVE_ROOT.json为唯一当前root。旧预算与v2已验收只读，不重复检查/下载/拟合；新小模型不适用50前缀相等断言，禁止运行旧audit_prefix_progress.py。

最终路径已实现但仅静态检查，未宣称新run最终PASS：自然退出且无自身进程后，remote.py final_scalars_readonly.py实际CPU strict 90CP/验证best/完整30轨迹/6000epoch49800更新/分区/父PatchSHA；audit_final_local.py SNAPSHOT SCALAR_DIR本地18case误差、CI、动态视觉与严格门复算；describe_capacity_final.py SNAPSHOT SCALAR_DIR独立绑定已验收512基线，给出全部train/validation/test预算、配对demo探索CI、计算成本及闭合索引。无需模型前向或环境。外置包装错误保留后修复，不能重跑科学实验。

本次为当前序列最后一个自动容量值。若严格A失败，完整验收发布并删除patch监控，不自动扫宽度、正则或更多epoch。B仅在严格A独立通过后另版本实施。额外准备/准入/失败成本单列，不称等算力。重复使用演示的探索性限制和全部旧失败保留。
