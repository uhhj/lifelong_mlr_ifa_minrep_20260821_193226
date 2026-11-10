# 当前任务内的有限实验跟踪

用户2026-11-10授权按各阶段预计耗时查询，并解决工程错误。本跟踪在当前会话执行，不是后台定时自动化；不新增、重启或扩展科学队列。

跟踪唯一根为ACTIVE_ROOT所列S_STRIPPED_MLR_IFA_PAIR_OBJECT_GOAL_20261110_v1，合同b54b0c2d8d8b490d6f7c394f7f99fa6689222861240454b4dd4eca870158b598。上轮六run已接受，不恢复。

检查点以当前一批并行完整run训练结束、随后整个套件正式评估结束为单位；不为Task6/7/8/9内部切换追加短周期查询。每次按自身阶段吞吐、剩余批次及保存/准入准备余量重新估计下一检查点。未有自身速度的阶段只使用明确标注的可比参考。

每次只读transport.py check_readonly.py，读取实际返回目录，audit_progress.py NEW PREVIOUS实际通过后才record_progress.py NEW。审查进程、真实退出、FAILURE、冻结代码合同、已完成轮和outcome前缀、optimizer账、环境互斥、容量及GPU状态；记录PARTIAL和观察时间，经publish.py精确审查发布，核远端SHA和本地receipt。

出现异常先保留原始证据、真实进程身份/退出、完整epoch边界、恢复状态元信息与已开始环境attempt。优先修复本地或外置工程包装，保持科学输入、损失、预算、样本流和已完成结果。不得盲重启训练/评估、覆盖冻结源码、重试正常科学负结果、删最终资产或付费扩容。任何需要恢复科学进程的工程处理，必须先实际审查其完整恢复与不重复已完成工作的证据。

只有唯一controller真实自然退出0且无自身训练/评估进程，才实际执行final_readonly.py及audit_final_local.py。最终验收48CP、12完整恢复、1089600生产更新、240隔离更新和2400正式初态/noise/attempt；外置最终包装错误可留证版本化修复，不重训不重评。完成实际验收、比较解释和发布后结束跟踪；不自动扩seed或扫参。

2026-11-10 21:53:47 CST实际检查通过：OBJECT_ALIGN_41和OBJECT_MLR_41各382/400轮，合计764轮197968生产更新，正式0。下一检查点约22:01 CST（当前这一对整run自然完成附近）；之后依据真实状态更新。

最终CPU外置兼容路径已准备：仅自然exit0无自身进程后transport.py final_readonly_engineering02.py→audit_final_local.py。保留冻结原包装不改；新包装只处理旧库EGL导入与CPU RNG读取，见FINAL_CPU_ENGINEERING02_REVIEW；尚未实际调用，不称PASS。
