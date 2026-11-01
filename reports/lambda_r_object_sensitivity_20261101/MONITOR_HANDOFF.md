## 2026-11-01 23:22 用户要求深入追查：原生热点已定位，生产未改

已完成非暂停原生IP+Python栈采样以及重复CPU计数，见NATIVE_BOTTLENECK_REPORT_20261101.md和capacity_readonly_20261101_231319/231448/231842/232037的实际证据及LOCAL审计。主线程3961原生样本零丢失，核实CUDA驱动内部内存复制循环、PyTorch cpu_equal和direct_copy_kernel；599个Python墙钟栈中复制/恢复/比较合计23.54%。GPU0–9%、通常2–4%，PCIe Gen4×16，未见Xid/PCIe错误或热功率限频。不能称GPU坏，也不能把所有几倍历史慢速都归于单一热点。

KVM虚拟机12vCPU、标称2800MHz；三次cycles/perf-task-clock约0.33–0.34十亿/秒，参考周期ENOENT且实际cpufreq不可见。只能把宿主机/vCPU有效执行资源或计数虚拟化列为需服务器侧核查的具体嫌疑，严禁将该虚拟计数声称为物理CPU已降至330MHz。未联系服务商、未重启/迁移/改线程/改精度/改驱动。本次所有原生观察0optimizer/0环境/0生产写入/目标信号。

最新check_readonly_20261101_232210/LOCAL_AUDIT实际PASS，230151全部epoch/门控/诊断前缀精确保留，135科学项和5工程项不变。L005_41 Task6/7各100轮、Task8完整59轮，共259轮56631接受97562生产，62自身隔离30准入autograd、20采集0正式；其他五run未开始。Task8近10轮235.03秒，尚无稳定恢复迹象。此前90–110小时全队列估时只是慢速持续情景，未实测优化前不能给可靠完成日期。

工程已唯一于deploy_remote_20261101_191752部署，合同b9ebd306e7ac3ce1fba2ce107ba5b2fa6b4fd20f62164790bf7ced4897b6086c。原等待调度器2163为T，当前launcher2164/训练2169继续；新owner3899/watchdog3900正常ARMED，HANDOFF_COMPLETE尚不存在，无失败或WATCHDOG_RELEASE。完整两λ八阶段一整轮逐位等价验证已实际PASS，7828 started==completed/36autograd/0环境另记，诊断3441已退出，不准重复验证dispatch。设备分支快照优化尚未用于生产；只等当前400轮40采集自然kernel exit0并核prefix/hash后，结束旧等待调度器、持原root锁接续未来五run，不信号训练、不重跑完成轮。禁止重跑任何dispatch/controller/build。

后续只用工程目录engineering_device_snapshot_20261101_v1的check/audit_progress/record_progress/final_readonly/audit_final_local。原root/controller.exit必须不存在，真实最终完成源仅工程controller.exit0且无自身进程。各新阶段须BRANCH_COPY_EQ及原自身GPU准入，不套验证PASS。若WATCHDOG_RELEASE出现，保留错误审计，不盲重复部署。最终CPU包装仍STATIC_ONLY，全部自然完成后才实际验收、根compare_primary_s及完整解释发布。

同一object-s-r-run监控下一480分钟在当前整run结束较早侧复查；这是按当前run余时安排，不固定高频。运输worker28064先读status核PID，不重启活跃worker、不输出凭据。完整性能证据已保存；科学协议不变，仅OBJECT两档各三seed，已有.1只读。健康静默，完成/新失败/需行动通知；六run全部实际验收解释发布前不得删除监控。

# OBJECT 最新S固定λ_R六run：当前交接

实际观察 2026-11-01T23:22:10.062176+08:00；check_readonly_20261101_232210/LOCAL_AUDIT=PASS_PROGRESS_NOT_FINAL。

唯一root `/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1`，controller3899，合同`6fde2ecbc3953f0af0709e70bbc3aaed3df9be57bf9e6bf59476985caafb135e`，135冻结项。唯一部署deploy_remote_20261101_152959。禁止重复compose/freeze/deploy/controller或已完成epoch。

|case|完整轮/400|接受更新|生产物理|采集/40|正式/200|
|---|---:|---:|---:|---:|---:|
|L005_41|259|56631|97562|20|0|
|L005_42|0|0|0|0|0|
|L005_43|0|0|0|0|0|
|L020_41|0|0|0|0|0|
|L020_42|0|0|0|0|0|
|L020_43|0|0|0|0|0|

累计259/2400轮、56631/633000接受、97562/1171800生产、62/546隔离、30/246准入autograd、20/240采集、0/1200正式。方向诊断最多216autograd另记。

下一480分钟复查，首次核自己的保持阶段准入及吞吐，此后依据整seed或整组正式实际剩余时间调整，不固定高频，不按时钟推断完成。

先完整读PROTOCOL.md、IMPLEMENTATION_NOTES.md、RUN_STATUS.json、ACTIVE_ROOT.json、FINAL_WRAPPER_REVIEW.json及最新EVIDENCE/LOCAL_AUDIT。用户只授权OBJECT六run：0.05/0.2各41/42/43，已有0.1三个完整S只读复用。GOAL不跑，C/Q/旧机制不恢复。每阶段100轮，最新S双候选10步门控、零epsilon、自己成功缓存q和原4槽分母不变；只改变保持合成项λ_R，准入同λ均匀支持参照不是原0.1轨迹。

顺序L005三seed训练→600正式→L020三seed训练→600正式。240采集20–29不补采，1200正式0–19原600步5zero配对Normal。每stage自己的GPU准入/COPY_EQ、官方初态及Task6对已有S全tensor实际通过；未来阶段不能套PASS。

运输使用旧机制目录现有.session_control worker，先读status并核PID，不重启活跃worker或输出凭据。本工程engineering_device_snapshot_20261101_v1/check复制旧目录queue，按result真实目录复制回来，audit_progress.py NEW PREVIOUS→record_progress.py NEW INTERVAL。任何错误先保留原失败、hash、进程/退出、完整恢复和已耗成本，禁止盲恢复部分块或环境attempt，不改科学协议。

只有自然engineering_device_snapshot_20261101_v1/controller.exit0且无自身进程后，才final_readonly实际CPU24CP/6恢复/24teacher缓存/1200初态noise/54600门控及预算验收→audit_final_local→compare_primary_s。FINAL_WRAPPER_REVIEW目前仅STATIC_ONLY。包装错误保留原证据后外置版本化修复，不重训重评。

随后自主完成FINAL_REPORT/ACCEPTANCE：三档全结果、新旧任务、SF/FS、全部负单元、门控接受与梯度/训练代理、成本边界。不按正式成绩选择最佳λ，不扩seed或负结果加实验。已有0.1的600回合被两次配对而非1200独立基线回合。保留开发集反复使用和预训练来源未闭合，敏感性分析不是独立确认或全局无泄露。

每次检查按根AGENTS经publish.py轻量审查code/report/hash/scalars/outcomes，串行nonforce推MLR核remoteSHA/localreceipt，禁CP/model/state/RNG/cache/凭据/未审bundle。健康进展静默，只完成、新失败或需行动通知。六run全部独立验收、解释及发布后才删除本新监控，不恢复旧监控。
