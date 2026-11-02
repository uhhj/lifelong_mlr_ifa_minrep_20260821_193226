# 同GPU并行seed可行性：只读初审

2026-11-02 15:43实际只读容量：12逻辑CPU、训练affinity0–11、CPU组quota=-1，主机总内存31.34GiB、可用25.42GiB；单训练当前RSS5.18GiB、已观察峰值6.59GiB。3090 Default计算模式、当前3032/24576MiB显存。此前30秒GPU均30.03%。本瞬间GPU0%和进程D只是瞬时状态，不解释为新失败。尚未完整核各session内存cgroup层级或后续阶段峰值。

观察支持把双run作为首个并行候选，不支持直接承诺三run、六run或线性提速。独立seed的数据、模型、RNG、缓存和输出必须隔离，每seed内部Task6–9仍串行；保留.05全组训练及600正式完成后才.2的边界。当前可考虑L00542+43，后组最多两个训练slot；正式保持原串行协议。

现调度器p.wait串行且共用唯一root锁/STATUS，不能手动启动43再让原调度器重复启动。需先做版本化调度器和独立状态、端口、保存路径审计，核内存上界，在隔离诊断中验证并行全训练路径的逐步标量、state、门控、保存重放及环境初态/噪声隔离；诊断成本另列。当前未部署并行、未启动其他seed、未改变当前训练。

先评估普通独立进程的总完成吞吐，不在当前任务中切换MPS或GPU模式。普通多CUDA进程存在context时间片，GPU利用率不能按30%乘进程数预测。评价应看两run合计有效更新/秒和整组完成墙钟，不追求利用率100%本身。不改变batch、精度、检查或采集预算。

参考：[NVIDIA CUDA MPS Architecture](https://docs.nvidia.com/deploy/mps/architecture.html)、[When to Use MPS](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html)、[PyTorch Multiprocessing best practices](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html)。本地资源证据为capacity_readonly_20261102_154352/EVIDENCE.json。0新增optimizer、环境、信号或生产写入。整体实验仍PARTIAL，本报告不是并行验证PASS或科学接受。
