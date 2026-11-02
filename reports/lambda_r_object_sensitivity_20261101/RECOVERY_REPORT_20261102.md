# 2026-11-02 RTX3090 恢复：PARTIAL，尚未科学最终验收

用户先要求继续训练，发现主机在暂停后重启、GPU由RTX4080SUPER改成RTX3090后，明确授权“允许在3090核验后续训”。原进程全部消失，暂停内存不可恢复。首次SIGCONT包装在任何信号前因旧PID不存在而退出，没有重启或重复生产。CPU检查确认原260轮、checkpoint完整state/优化器/scheduler/grads/modes/requires及全部RNG、历史、probe SHA一致。

独立GPU验证v1因probe scale重算差一个float32 ULP（约2.98e-8）在0optimizer/0autograd/0环境前停止，错误保留。v2精确使用原冻结Task8归一化常数（不是调参或放宽门控），核完整GPU恢复、29调用原准入、7块两条CPU/device快照路径。实际309 started==completed、11额外autograd、0环境、0生产轮；每步标量、两分支恢复、所选完整state/探针/决策精确。7块与旧GPU原记录全部逐字段相等。该观察只覆盖验证路径，不能宣称全run跨GPU逐位等价。

唯一新工程 `engineering_resume_3090_20261102_v1`，12:48:22启动controller2560、训练2566，合同 `46bf5fe3bf0acfe9f23441a6c64c35d29438e8abdaef27262ffa5a809e6741e3`，8冻结项；135科学及5旧工程冻结项保持。原root和旧工程controller.exit均不存在，最终实际成功来源只使用新工程controller.exit0且无自身进程。禁止重跑旧/新任何dispatch/controller/build/已完成轮。

原checkpoint另存私有interrupted_epoch260.pth，原4167条门控完整归档且原文件全部保留。第261轮前7块重建须逐字段等于已有记录，不重复追加门控行；前260轮不重训。中断部分已记录140候选调用与未记账部分UNKNOWN单列，不能把它说成零；309恢复诊断与此前7828诊断分开，合计8137optimizer/47autograd，不混入1171800生产预算。

实际 2026-11-02T12:49:55.526698+08:00，check_readonly_20261102_124955 / LOCAL_AUDIT / RECOVERY_LOCAL_AUDIT通过。当前261轮，接受57249、已完成轮生产物理98798，自身62隔离/30autograd、20采集、正式0。原260轮与全部门控前缀精确保留，controller2560/GPU2566健康；其余五run未启动。恢复后完整轮秒数 [48.368486642837524]，样本不足以推算全队列；旧慢速ETA失效。

后续顺序及loss/λ/阈值/采集/正式预算不变。每阶段自己的GPU准入和设备快照一致性核验继续。新run仍要求官方初始化tensor/RNG相等；用户已授权硬件变化，Task6跨GPU结果实际记录全部差异，不再伪称硬件相同或tensor必相等。L005_41前260轮在旧硬件、后续在3090，其余新run在3090，旧0.1基线不重跑。因此最终报告必须披露硬件混杂，不能称λ单因素硬件完全受控对照。

新工程check/audit_progress/record_progress/final_readonly/audit_final_local已外置适配；最终CPU仍STATIC_ONLY，待全部自然结束才真实24CP/6恢复/24teacher/1200初态noise/54600门控与恢复成本验收。之后根compare_primary_s、三档全结果/新旧任务/SF/FS/负单元/成本/开发集及预训练来源边界解释，发布后关闭监控。

同一object-s-r-run已ACTIVE，下一30分钟一次性核恢复吞吐，再依据当前完整run/整组评估剩余时间调整。现有运输worker3700，先核status/PID，不重启活跃worker或输出凭据。健康静默，只实质变化/完成/新失败/需行动通知。
