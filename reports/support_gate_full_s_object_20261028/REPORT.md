# 最新完整 S OBJECT 三 seed：运行报告

## 2026-10-29 09:49 性能诊断与原队列保持

用户要求具体瓶颈与结果不变的优化可行性。PERFORMANCE_REPORT_20261029.md及PERFORMANCE_AUDIT_20261029.json记录120秒非阻塞外部采样：主线程训练更新45.16%、state捕获28.19%/恢复9.43%/检查2.85%、探针6.38%、读数据/保存各1.53%，GPU均23.68%、进程CPU99.70%单核，未见quota限制。无forward/backward/optimizer/环境或已完成epoch重跑；独立state复制验证454张量全值/dtype/stride/独立性/RNG实际PASS，省去CUDA→CPU后冗余CPUclone中位117.53→56.75ms，条件整段节时约14.6%，不是整训练实测。

候选performance_copy_candidate.py尚未部署，复制原语PASS不能等同完整训练等价。当前解释器无安全热替换接口，原controller无seed边界切换钩子；未来采用需另行版本化队列接管和完整准入，不能并行再启动队列。本轮未暂停/停止/重启任何科学进程，63冻结项保持。工具py-spy0.4.1仅隔离/data/mlr_ifa/diagnostics/support_gate_performance_20261029_v1，官方wheel及binary SHA留证，原环境未改，诊断包装已还原。

最后check_readonly_20261029_094951/LOCAL_AUDIT真实PASS，092535全部前缀保留：seed41完成训练和40采集；seed42 Task8完整85轮，总285轮64665接受/113630物理、20采集；seed43尚未开始。累计685epoch170165接受/308930完整轮物理、153隔离/71autograd、60采集，正式0。controller3013/GPU5060健康，旧机制无进程；独立最终CPU验收待全队列自然结束。监控仍240分钟围绕当前seed末尾复查，09:25总ETA只是条件估计，未把候选提速计入ETA。

## 最新 2026-10-29T09:49:52.166661+08:00：最新完整 S 进度复查

独立 root `/data/mlr_ifa/experiment_workspaces/SUPPORT_GATE_FULL_S_OBJECT41_42_43_20261028_v1`，controller 3013，合同 `8594c0b609b8bf948e7914fbf071567ed991d1698a9d8357b90fcb03dec3573d`，63 冻结项实际通过。唯一部署证据 deploy_remote_20261028_195120。禁止重跑 build_local/deploy/controller 或重复训练已完成 epoch。

seed41：400/400 完整 epoch，105500 接受更新 / 195300 已完成轮物理训练调用，采集已完成 40/40，正式 0/200。当前 {"arm": "S", "stage": 9, "phase": "training_collection", "time": 1793222243.7766988, "pid": 3019}。
当前阶段最近至多10轮均 126.182989 秒，线性纯计算余时 None 分钟；仅当前阶段、不含保存/采集/后续阶段与正式评估。若只有1轮则样本不足，不作稳定ETA。
seed42：285/400 完整 epoch，64665 接受更新 / 113630 已完成轮物理训练调用，采集已完成 20/40，正式 0/200。当前 {"arm": "S", "stage": 8, "phase": "train", "epoch_complete": 85, "steps": 64665, "physical_optimizer_calls": 113630, "time": 1793238563.2120788, "pid": 5060}。
当前阶段最近至多10轮均 73.693032 秒，线性纯计算余时 18.423257970809935 分钟；仅当前阶段、不含保存/采集/后续阶段与正式评估。若只有1轮则样本不足，不作稳定ETA。
seed43：0/400 完整 epoch，0 接受更新 / 0 已完成轮物理训练调用，采集已完成 0/40，正式 0/200。当前 {}。

合计 685/1200 完整 epoch，170165/316500 接受、308930/585900 完整轮物理训练调用。自身隔离 153、额外 autograd 71。当前快照 check_readonly_20261029_094951/LOCAL_AUDIT.json 实际 PASS_PROGRESS_NOT_FINAL。

Task6 无保持；Task7–9 的逐任务 q 来自各 seed 自己的 10 次成功采集缓存，使用固定 10 步 C/Q 完整状态门控。内部 C/Q 候选属于新 S，不是恢复旧机制对照。每 task100epoch、每 task10训练采集20–29、不补采；正式各 stage 已见任务0–19，600steps5zero配对Normal。三 seed120采集+600正式，273隔离更新/123准入autograd另记，训练诊断最多108autograd。原科学阈值、损失、块长和预算不变。

旧机制已按用户最新指令停止，保留 GOAL三seed12组 + OBJECT41 C/S 共14完整组及 Q3轮，G/U与正式0。停止证据在旧目录 deploy_remote_20261028_193702/EVIDENCE.json，旧 root/user_stop_20261028/RESULT.json；停止后进程为空，56冻结项通过，所有CP/恢复点/日志保留。旧中断及停止时未记账部分成本仍UNKNOWN。不得恢复旧队列，也不等待其最终验收。

队列顺序：train seed41完整6–9 → train42 → train43 → evaluate41 → evaluate42 → evaluate43。各阶段必须自己的准入通过，不能套首阶段 PASS。外置final_readonly.py和audit_final_local.py已适配新root、支持度和固定探针、9100块/seed、195300物理/seed与273隔离调用，FINAL_WRAPPER_REVIEW仅STATIC_ONLY，实际最终CPU仍未运行。仅controller.exit0且无自身进程后，将本目录final_readonly.py复制至旧运输目录，queue后按真实目录取证，执行audit_final_local.py FINAL再describe_final.py FINAL；CPU12CP/3全state/12缓存teacher/600初态noise和门控标量必须真实通过。包装故障保留原错误后仅外置版本化修复，不重训重评。自动表格后需核对新旧任务、SF/FS、训练代理与门控率的解释，不得自动宣布科学优势。

下一240分钟按当前整seed训练及采集的粗略完成点复查，再依据新的实际状态调整；具体实测和外推边界见本快照TIMING_AUDIT.json（如有）。这是检查安排，不是总完成承诺，不采用固定短周期。只实质变化/完成/新失败/需行动通知，健康无变化静默。

监控传输仍用旧目录现有worker4572，先读.session_control/status.json，不重启活跃worker或打印凭据。本目录check_readonly.py复制到旧目录同名包装，queue_remote.py check_readonly.py后从实际result.output第一行读真实目录，再把完整轻量证据复制到本目录。运行 audit_progress.py NEW [PREVIOUS]，再 record_progress.py NEW INTERVAL。所有stage/GATES前缀精确核对；异常先核进程/hash/退出/完整恢复和已耗成本，不盲重启环境或部分块。

按根AGENTS串行运行本目录publish.py以及旧目录停止报告的publish.py，审查轻量文件、nonforce推MLR、remoteSHA/localreceipt。禁止发布CP/状态/缓存/凭据。最终三个seed独立验收、分析报告、发布后再删除同一个s-object-seed监控。
