# 3090双任务并行试验与安全调度

## 最新 2026-11-02T16:48:00.640074+08:00：并行等价验证通过，调度已部署等待自然边界

用户要求实际尝试多seed。三次隔离试验合计18584 optimizer、0额外autograd、0环境、0生产epoch。两档λ/Task6–9、seed42/43串行对并行完整一轮的逐步标量、C/Q完整状态、探针、门控和保存加载下一步实际逐位相等。Task9去逐块指纹的计时为串行两份139.46秒→并行85.28秒，合计吞吐1.6354倍、墙钟减少38.85%。生产同时作为背景负载，不能外推整run/所有未来轨迹。并行计时窗口GPU均97.76%，显存峰值9050MiB，含生产+两个诊断进程，不是未来双生产利用率承诺。

完整验证的串行参照先完成6392调用；生产进入采集时触发剩余资源保护，诊断在任何并行调用前退出。失败原件保留，16:30只读资源恢复后唯一continue_parallel_v2续缺少6392调用，未重复串行参照或科学轮。累计工程34549 optimizer/83额外autograd单列，旧中断140已记候选+UNKNOWN保留。所有诊断已退出。详见PARALLEL_OPTIMIZATION_REPORT_20261102.md、ENGINEERING_PARALLEL_STATUS.json及工程PARALLEL_TRIAL_AUDIT.json。

新engineering_parallel_3090_20261102_v1合同00d50232d42461c1d78afaff694f53b575ca562bc752211e9085907633cbaca6、29冻结项，controller8608/watchdog8609已ARMED；仅旧等待调度器4347为T，当前launcher4689/GPU4694未收信号且继续。HANDOFF_COMPLETE尚不存在，不称生产并行已启用。L005_41完整400轮40采集；L005_42现222轮、45198接受、74696生产物理、20采集。check_readonly_20261102_164759及新audit_progress/PARALLEL_HANDOFF_LOCAL_AUDIT PASS，153532全部epoch/门控前缀及新GUARD前缀精确保留，135科学+5设备+8恢复+6检查及新工程项不变。

调度等L00542自然400轮40采集、launcher内核Z/exit0及prefix/hash通过后，结束旧等待调度器并取得同root锁。然后L00543单独训练→L005三seed600串行正式→L020三seed最多2个训练并行→L020三seed600串行正式；训练采集全局互斥。没有改变loss/阈值/批次/各seed任务顺序/RNG/预算。若一个并行训练失败，不再启动后续任务，让已启动同伴自然结束并报告，不盲重启。

只用新工程check/audit_progress/record_progress/final_readonly/audit_final_local；最终只认新工程controller.exit0且无自身进程。根及设备/恢复/检查工程旧exit保持不存在。全部6份嵌入CPU脚本也已外置适配新完成源，最终仍STATIC_ONLY，真实24CP/6恢复/24teacher/1200noise/54600gate及16次未来采集互斥验收待全部结束。未来每阶段自己的GPU准入/BRANCH_COPY_EQ/CHECK_EQ必须实际通过；3090迁移、开发集反复使用、预训练来源未闭合限制不变。禁止重复任何dispatch/controller/validation/build或完成轮。

下一150分钟在当前完整L00542预计结束较早侧复查，根据新实测调整；后续并行剩余时间按实际并发完成时间估计，不能简单累加或把1.64倍套全部队列。保持同一object-s-r-run，健康静默；全部科学验收解释发布后才删除。运输worker3700先核status/PID，凭据不输出。每次按AGENTS经publish.py轻量审查串行非force推MLR核远端SHA/receipt。


原始轻量证据、诊断源码、独立复算和部署来源均见engineering_parallel_3090_20261102_v1。当前为PARTIAL；没有正式闭环结果或全实验验收。
