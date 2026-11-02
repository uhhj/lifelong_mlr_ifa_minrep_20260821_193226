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



## 2026-11-02 20:53 用户授权跨λ组并行已实际启动

用户明确“可以把0.05的这一个seed和0.2的一个seed并行吗”。已唯一部署engineering_cross_lambda_20261102_v1（deploy_remote_20261102_205304），合同ee7929e8ae178125508a0feead81c0fa153f7a725f62f46bdf99fe88c9fbcead、5工程项，controller11852/watchdog11854。仅冻结原等待controller11239；L00543原launcher11241/GPU11246未收到任何信号、没有重启。新增L02041 launcher11855/GPU11860，实际端口42689与现有42941不同，都显式独立c10d localhost:0和唯一UUID，不用--standalone。

check_readonly_20261102_205347实际审计PASS，204611全部epoch/门控/诊断前缀保持，135科学及全部工程冻结项不变。L00541/42各400轮40采集；L00543现307轮(Task9 7)、71834接受127968生产91隔离41准入autograd30采集，自己的Task9准入/BRANCH_COPY_EQ/CHECK_EQ通过。L02041完整3轮471接受/生产，自己的Task6四隔离及八autograd、初态tensor/RNG/COPY_EQ/CHECK_EQ通过；后续阶段不得套PASS。总1110轮283305接受519039生产277隔离131准入autograd110采集0正式。L02042/43未启动。

LIVE_COMMITTED/LIVE_LAUNCH已实际核真，但HANDOFF_COMPLETE尚不存在。根锁仍由冻结11239持有；等原43 launcher内核Z/exit0、400轮40采集和prefix/hash通过后，新owner才结束旧等待器、取得同一根锁，写前一恢复工程L005_43_resume.exit的真实结果。严禁手动恢复11239或重复任何controller/dispatch/build/validation/已完成轮，否则可能重复未来任务。若新owner在LIVE_COMMITTED后失效，watchdog保持11239冻结并写OWNER_LOST_AFTER_LIVE_COMMIT，先核现有进程与真实成本，不盲恢复。

用户新调度覆盖旧“.05正式全部完成才允许任何.2训练”的约束：当前仅提前L02041与L00543并行；43自然完成后立即优先L005三个seed600串行正式，已启动的L02041可继续训练；正式与所有训练采集共用原collection全局flock，环境不并行。L005600正式后才启动L02042/43，最多两个训练进程，最后L020600串行正式。训练源码/wrapper/每run任务顺序、loss/阈值/块长/样本和预算均未改变。零环境模拟实际原评估wrapper锁与采集型锁竞争，独立区间无重叠，原evaluator argv/cwd不变；不是新增正式或科学环境验证。新增工程optimizer/环境均0。

实际采集互斥5/18已通过，未来13次继续核对；6个正式评估锁尚未发生，只是静态/零环境机制验证通过，不能提前称实际全部互斥完成。原采集锁事件在engineering_live_parallel_3090_20261102_v1，新正式锁事件在本cross工程，最终合并核全部时间区间不重叠。旧live自有pth和HOOK_ENABLED保留至全部接受后，核无目标collect/原SHA再清理，不提前删除其他环境文件。

只用本cross工程check/audit_progress/record_progress/final_readonly/audit_final_local。唯一最终全部成功源改为本cross工程controller.exit0且无自身进程；前一恢复controller.exit必须保持不存在，原live controller.exit1/43原train_S.exit1和FAILURE原SHA永久保留。43真实训练完成仍要求前一恢复工程L005_43_resume.exit0。六CPU最终wrapper外置适配仍STATIC_ONLY：全部自然退出后真实24CP/6恢复/24teacher/1200noise/54600gate、所有成本及环境互斥验收，再compare_primary_s三档全结果、新旧任务/SF/FS/负单元和硬件/泄露边界解释发布。

工程累计35418optimizer/94autograd不变，旧140+新400=540已记录放弃候选调用及UNKNOWN未记账成本保留；此次无新GPU训练验证成本。此前完整恢复与原289轮/5079门控保留，3090迁移不等λ单因素硬件完全受控，旧0.1只读，不跑GOAL/C/Q/旧机制，不挑最佳λ或扩seed。

当前跨组并行仅刚启动，不能将历史1.71倍或原13–18小时规划直接当新提速/完成保证。下一30分钟一次性核L020首个保持阶段准入与实际并行吞吐，再按当前整run/整组正式余时安排，不固定高频，不按时间断言完成。完整事实见CROSS_LAMBDA_REPORT_20261102.md、ENGINEERING_CROSS_LAMBDA_STATUS.json与最新CROSS_LAMBDA_STARTUP_AUDIT。

运输旧机制目录.session_control worker3700，先核status/PID，不重启活跃worker、不输出凭据；本cross工程check复制旧目录queue，按result真实目录复制完整EVIDENCE，不解析截断output主体或猜目录秒数。每次审计照AGENTS审核轻量证据串行nonforce推MLR核remoteSHA/receipt；健康静默，只实质变化/完成/新失败/需行动通知。全部六run实际验收解释发布后才删除同一监控。

