# 最新完整 S OBJECT 三 seed：运行报告

## 2026-10-29 14:23 用户请求进度、实际速度和剩余时间

最新check_readonly_20261029_142301实际PASS，131848全部epoch/门控前缀精确保留；63科学项和5工程项不变，controller7806/GPU8320健康，旧机制无进程。seed41/42各400轮和40采集完成；seed43 Task6完成100轮/10采集，Task7完成94轮，总194轮37038接受58376生产物理、33隔离19autograd；Task8/9未开始，正式三seed均0/200。合计994epoch248038接受448976生产物理、215隔离101准入autograd、90采集、正式0；额外工程3914optimizer/18autograd仍单列。

相同Task7前94轮观测：seed42均71.6979秒→优化seed43均46.2231秒，耗时减少35.5308%、吞吐比1.5511；中位54.8413→46.4213秒，减少15.3534%。原seed42存在最大228.94秒的慢速波动，跨seed数据mask/自己的缓存/资源不同，不能把35.5%全部归因于复制优化；没有新增同状态整训练性能对照。当前最近10轮45.1489秒，Task7剩6轮约4.515分钟纯计算。

未实测Task8/9用seed42同阶段参考，按当前速度比缩放与不计提速两种情景，余训练采集线性203.0–310.1分钟；计准备保存与波动规划余4–6小时，三seed600正式另留1–3小时，总余5–9小时（本次14:23观察起算，10月29日19:20–23:20），CPU独立验收/报告发布另计。正式当前未开始，1–3小时是历史参考余量，不是本轮测得。TIMING_AUDIT保存公式与假设。下一210分钟围绕整seed预计完成较早侧复查，覆盖旧270分钟安排；不按时钟推断完成，不固定高频。


## 最新 2026-10-29T14:23:02.390973+08:00：用户授权复制优化验证与安全交接

实际验证PASS_FULL_TRAINING_PATH_EXACT：Task6–9各一整轮全部batch，157/227/309/362批，C/Q完整候选与两分支恢复、每步标量、探针、门控、所选完整state和保存加载下一步逐位相等。额外诊断3914 optimizer、18 autograd、0环境、0生产接受epoch，独立记账；这是完整路径覆盖，不是重跑400epoch或已经验收三seed。原seed41已有CP/JSON及63冻结项实际保持。

工程合同 befceb11089903813ba65c839e604a44ab1d0c18d7f2dc8282436c581c5ed3cf，目录engineering_copy_20261029_v1，controller7806，当前 {"phase": "train", "seed": 43, "pid": 8315, "time": 1793249766.6886415, "controller": 7806, "engineering_contract": "befceb11089903813ba65c839e604a44ab1d0c18d7f2dc8282436c581c5ed3cf"}。仅CUDA→CPU后的重复CPUclone被去掉，CPU源clone保留。冻结训练文件未修改，wrapper在新seed43解释器导入时替换函数；原各阶段自身GPU准入保留，并核seed43实际阶段开始完整state复制。

seed42已自然完成400轮和40采集、launcher内核Z/exit_code0被记录后才结束原等待调度器3013；新调度器持有同一根锁，仅train43→evaluate41→evaluate42→evaluate43。
禁止重跑guard/validation/dispatch/controller/build或已完成epoch；不得恢复旧机制。原root/controller.exit保持不存在，最终真实完成来源为engineering_copy_20261029_v1/controller.exit==0且无自身进程，不可伪造原exit。

seed41：400/400轮，105500接受/195300已完成轮物理调用，采集40/40，正式0/200；阶段状态{"arm": "S", "stage": 9, "phase": "training_collection", "time": 1793222243.7766988, "pid": 3019}。
seed42：400/400轮，105500接受/195300已完成轮物理调用，采集40/40，正式0/200；阶段状态{"arm": "S", "stage": 9, "phase": "training_collection", "time": 1793249619.0206304, "pid": 5060}。
seed43：194/400轮，37038接受/58376已完成轮物理调用，采集10/40，正式0/200；阶段状态{"arm": "S", "stage": 7, "phase": "train", "epoch_complete": 94, "steps": 37038, "physical_optimizer_calls": 58376, "time": 1793254952.624099, "pid": 8320}。

合计994/1200完整epoch、248038/316500接受、448976/585900生产物理调用，自身准入215/273、101/123额外autograd，采集90/120、正式0/600。额外3914诊断调用不混入生产计数。

科学协议仍是官方OBJECT初始化、Task6–9各100轮，Task6无保持，Task7–9自己的成功缓存q与固定10步C参照门控、两个epsilon0、原4槽分母；每阶段10采集20–29不补采，正式0–19原600步5zero配对Normal。内部C/Q属于新S。禁止扩seed、扫参、改变损失阈值或采样预算、因负结果追加实验。旧机制永久停止，14完整组及Q3轮保留，旧未记账中断成本UNKNOWN。

监控沿用旧目录worker4572，先读.session_control/status，不重启或输出凭据。使用本工程目录check_readonly.py复制到旧目录后queue，按result真实目录复制证据回新目录；engineering_copy_20261029_v1/audit_progress.py NEW PREVIOUS，再engineering_copy_20261029_v1/record_progress.py NEW INTERVAL。禁止使用原record_progress覆盖工程实际controller和完成来源。

本工程final_readonly.py/audit_final_local.py已外置静态适配真实工程exit及单独诊断账目，FINAL_WRAPPER_REVIEW仍STATIC_ONLY；全部自然结束且无自身进程后才实际CPU12CP/3全state/12缓存teacher/600初态noise和全部门控标量验收，再运行根describe_final.py并解释新旧任务、SF/FS、负结果和训练代理局限。每个未来阶段必须自己的准入。

下一210分钟依据实际阶段吞吐复查，不把约14.6%条件潜在节时当实际整seed提速，不根据时间推断完成。仅实质变化/失败/完成/需行动通知，健康静默。每轮按根AGENTS经publish.py审查轻量code/report/hash/scalars/outcomes，串行nonforce推MLR核remoteSHA/localreceipt，禁CP/state/RNG/cache/凭据。全部三个seed独立验收、解释报告和发布完成后才删除同一监控。
