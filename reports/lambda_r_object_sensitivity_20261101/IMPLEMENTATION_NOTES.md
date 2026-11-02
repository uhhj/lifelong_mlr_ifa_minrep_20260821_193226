# 工程与科学边界

accepted OBJECT S 为科学源。updates/legacy_updates、memory/legacy_memory/support_memory、losses/legacy_losses、collect/evaluate及synthetic_checks逐字保留。layout仅输出身份、容量上界及授权λ_R改变。gate_admission候选显式系数改为读取RETAIN_WEIGHT，fork_tools方向诊断同样读取；probe和choose保持原代码。SOURCE_REVIEW与review差异绑定来源SHA。

runtime复用Q v2已运行通过的工程观测层（实际scheduler、独立物理调用计数、COPY_EQ），生产仍为S双候选。初始化和Task6与已接受S真实比对。所有其他真实阶段必须自己的准入，静态检查不替代GPU或CPU验收。

容量上界：新实验/data私有资产12GiB，保留2GiB；/root新预算128MiB，保留1GiB。预检与部署再次检查空闲GPU、旧实验真实退出、父数据/官方初始化/来源hash及新root不存在。

运输沿用execution/20261027_support_gate_forks中的已授权Credential Manager worker。先读status并核进程，不重启活跃worker，不打印凭据。queue后按result实际返回目录取证，不猜秒数。最终或部署请求绝不重复提交。

本轮baseline的OBJECT seed41/42是在复制优化前、43是在优化后训练；优化已完整路径逐位验证，只影响复制成本。已有3914诊断调用保持历史账，不计本轮。λ变化导致缓存、模型轨迹和门控接受变化属于完整方法响应，不是共同中间state单步因果。


## 2026-11-02 20:39：并行启动器故障已核实并恢复，原失败永久保留

20:13发现旧live controller.exit1，L00543原train_S.exit1。L00542自然400轮40采集exit0之后，PyTorch1.13的--standalone强制共用localhost:29400，43失去TCPStore后由自身torchrun发SIGTERM；不是本次已观察到的GPU故障。完成41/42与43原289轮、135科学及全部历史工程项不变。旧等待器已结束，禁止恢复任何旧controller。

新engineering_rendezvous_recovery_20261102_v1于deploy_remote_20261102_203611唯一部署，controller11239/launcher11241/GPU11246；合同23f1dda97de2ebd9a1b752db2f34ab23ccd4e2f4ed9461ce4e0d99887a5fa7c4，15项。已实际两个独立端口生命周期PASS、869optimizer/11autograd/0环境完整恢复、20原门控块逐字段及下一块双路径精确PASS，诊断已退出。首个下划线参数修复前零成本参数解析失败保留。禁止重复dispatch/controller/validation/build或已完成轮。

最新check_readonly_20261102_203858实际审计PASS，203637全部前缀保留；43恢复后的3轮实际完成，现292/400(Task8 92)，66828接受117956完整轮生产、62隔离30准入autograd20采集。41/42各400轮40采集；L020未开始。累计1092轮277828接受508556生产244自身隔离112autograd100采集0正式。原43的289轮和5079门控行精确保留，重建20块不重复追加。完整恢复及前三轮证据见RECOVERY_STARTUP_AUDIT。

原43/train_S.exit1、旧live/controller.exit1及FAILURE.json按合同SHA永久保留，不伪造成功。新43退出来源为新工程L005_43_resume.exit，最终全部完成来源只用新工程controller.exit0且无自身进程。根、旧设备/恢复/检查/原parallel出口仍不存在，旧live失败出口必须存在。只用新工程check/audit/record/final包装；六CPU最终wrapper已静态适配真实新来源和43特殊失败/恢复出口，仍STATIC_ONLY。

工程累计35418optimizer/94autograd单列；旧硬件中断140+本次400=540已记被放弃候选调用，未记账部分仍UNKNOWN。新增私有备份346445621B已计容量。不能声称总实际成本精确。硬件迁移/开发集复用/预训练来源未闭合解释不变。

恢复3轮均44.6359秒，只是Task8单任务小样本，不作新并行提速结论。当前43余训练采集暂按2–2.5小时，L0209–11小时及1200正式2–4小时均条件参考；总余13–18小时(20:39起约11月3日09:40–14:40)，CPU验收解释发布另计。旧18:35 ETA已被真实故障覆盖。下一30分钟一次性核恢复后稳定吞吐与Task9自己的准入，再按整run/整组正式余时调整，不固定短轮询。

未来L020仍最多两训练并行，显式--rdzv_backend=c10d --rdzv_endpoint=localhost:0 --rdzv_id=唯一UUID，不用--standalone；核实际端口不同。采集hook及18次事件仍在旧live目录，当前4次互斥已通过，其余14次待核；不得提前移除hook。全部最终验收后核无目标collect并核SHA，只清理自有pth和旧live/HOOK_ENABLED，保留证据。未来每阶段自己的GPU准入/BRANCH_COPY_EQ/CHECK_EQ，科学顺序.05全训练→600正式→.2全训练→600正式不变。

先完整读RENDEZVOUS_RECOVERY_REPORT_20261102.md、ENGINEERING_RENDEZVOUS_RECOVERY_STATUS.json、新工程STATIC_REVIEW/AUDIT_WRAPPER_REVIEW及最新证据。运输worker3700先核status/PID，不重启活跃worker、不输出凭据。失败先核进程/冻结/恢复/真实成本，不能盲重启。

