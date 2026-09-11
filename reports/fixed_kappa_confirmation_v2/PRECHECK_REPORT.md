**PRECHECK REPORT — PASS**

新seed45/46/47审计通过。只比较MLR与同批聚合κ=0.5。协议、官方checkpoint、原始数据和三seed缓存已固定并完成哈希核对。

单元测试、既有fixed-kappa回归测试、真实GPU κ=0/1端点验证通过。真实策略日志开/关比较的loss、梯度、参数、AdamW、scheduler与RNG逐位一致，含真实触发和机制快照恢复下一步检查。详见GPU_PARITY.json及原始日志。

两arm均通过torchrun真实缓存smoke；task6的逐batch损失序列和model/optimizer/scheduler/RNG哈希完全一致。正式task6的100epoch与评估一致性仍由队列强制闸门检查，不以smoke替代。

三seed缓存的完整CompletePolicy前端配置与历史生成配置逐项相同，use_language=True、use_state=False。BriefPolicy训练端消费固定H，不含上述前端开关。

正式训练尚未由本preflight脚本启动。只有单独正式队列写出STARTED.json并有存活PID/训练日志时，才能报告已启动。

带日志的timing包含诊断复制、范数和快照IO；不是纯算法开销。3个新seed的主报告先于历史及6seed探索性描述；机制数据不得反向影响训练。
