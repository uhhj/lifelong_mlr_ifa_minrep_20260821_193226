# 最新完整 S OBJECT 三 seed：运行报告

## 2026-10-29 17:59 整seed完成前实际复查

check_readonly_20261029_175935/LOCAL_AUDIT实际PASS，142301全部已记录epoch和门控前缀精确保留，63科学项及5工程源不变，controller7806/GPU8320正常，旧机制无进程。seed41/42各400轮和40采集保持；seed43 Task6/7/8各100轮、Task9完整65轮，365/400轮、92830接受/169960生产物理调用、30/40采集，正式三seed仍0/200。Task8/9各自准入PASS，四个实际阶段初态复制均逐位相等；总1165完整epoch、303830接受/560560生产物理、273自身隔离/123准入autograd、110采集、正式0/600。额外3914工程optimizer/18autograd单列，最终CPU与闭环仍待后续。

Task9近10轮均94.3683秒（82.45–103.95），全65轮均101.3470秒，剩35轮纯计算55.05–59.12分钟，另有保存和最后10采集。规划剩训练采集1–1.5小时、三seed正式另1–3小时，合计2–4.5小时（从17:59观察起，约10月29日20:00–22:30），CPU独立验收及报告发布另计。正式尚未实测，1–3小时沿用历史参考余量，不保证完成时间。下一75分钟在整seed结束附近复查，再按真实正式进度调整，不能按时钟推断完成。

相同阶段前缀与seed42跨seed比较：Task7全100轮平均耗时减少34.50%（中位13.28%）；Task8全100轮平均减少31.33%（中位22.65%）；Task9前65轮平均反而增加5.00%（中位4.37%）。不同seed自己的缓存、mask和资源状态不同，这些是观察差异，不能声称优化在全部阶段带来统一因果提速。TIMING_AUDIT保留公式、参考SHA和局限。正常推进无新失败或需用户操作；本段覆盖旧Task7进度、未测Task8/9推算和210分钟间隔。

## 最新 2026-10-29T17:59:37.191257+08:00：用户授权复制优化验证与安全交接

实际验证PASS_FULL_TRAINING_PATH_EXACT：Task6–9各一整轮全部batch，157/227/309/362批，C/Q完整候选与两分支恢复、每步标量、探针、门控、所选完整state和保存加载下一步逐位相等。额外诊断3914 optimizer、18 autograd、0环境、0生产接受epoch，独立记账；这是完整路径覆盖，不是重跑400epoch或已经验收三seed。原seed41已有CP/JSON及63冻结项实际保持。

工程合同 befceb11089903813ba65c839e604a44ab1d0c18d7f2dc8282436c581c5ed3cf，目录engineering_copy_20261029_v1，controller7806，当前 {"phase": "train", "seed": 43, "pid": 8315, "time": 1793249766.6886415, "controller": 7806, "engineering_contract": "befceb11089903813ba65c839e604a44ab1d0c18d7f2dc8282436c581c5ed3cf"}。仅CUDA→CPU后的重复CPUclone被去掉，CPU源clone保留。冻结训练文件未修改，wrapper在新seed43解释器导入时替换函数；原各阶段自身GPU准入保留，并核seed43实际阶段开始完整state复制。

seed42已自然完成400轮和40采集、launcher内核Z/exit_code0被记录后才结束原等待调度器3013；新调度器持有同一根锁，仅train43→evaluate41→evaluate42→evaluate43。
禁止重跑guard/validation/dispatch/controller/build或已完成epoch；不得恢复旧机制。原root/controller.exit保持不存在，最终真实完成来源为engineering_copy_20261029_v1/controller.exit==0且无自身进程，不可伪造原exit。

seed41：400/400轮，105500接受/195300已完成轮物理调用，采集40/40，正式0/200；阶段状态{"arm": "S", "stage": 9, "phase": "training_collection", "time": 1793222243.7766988, "pid": 3019}。
seed42：400/400轮，105500接受/195300已完成轮物理调用，采集40/40，正式0/200；阶段状态{"arm": "S", "stage": 9, "phase": "training_collection", "time": 1793249619.0206304, "pid": 5060}。
seed43：365/400轮，92830接受/169960已完成轮物理调用，采集30/40，正式0/200；阶段状态{"arm": "S", "stage": 9, "phase": "train", "epoch_complete": 65, "steps": 92830, "physical_optimizer_calls": 169960, "time": 1793267970.4511213, "pid": 8320}。

合计1165/1200完整epoch、303830/316500接受、560560/585900生产物理调用，自身准入273/273、123/123额外autograd，采集110/120、正式0/600。额外3914诊断调用不混入生产计数。

科学协议仍是官方OBJECT初始化、Task6–9各100轮，Task6无保持，Task7–9自己的成功缓存q与固定10步C参照门控、两个epsilon0、原4槽分母；每阶段10采集20–29不补采，正式0–19原600步5zero配对Normal。内部C/Q属于新S。禁止扩seed、扫参、改变损失阈值或采样预算、因负结果追加实验。旧机制永久停止，14完整组及Q3轮保留，旧未记账中断成本UNKNOWN。

监控沿用旧目录worker4572，先读.session_control/status，不重启或输出凭据。使用本工程目录check_readonly.py复制到旧目录后queue，按result真实目录复制证据回新目录；engineering_copy_20261029_v1/audit_progress.py NEW PREVIOUS，再engineering_copy_20261029_v1/record_progress.py NEW INTERVAL。禁止使用原record_progress覆盖工程实际controller和完成来源。

本工程final_readonly.py/audit_final_local.py已外置静态适配真实工程exit及单独诊断账目，FINAL_WRAPPER_REVIEW仍STATIC_ONLY；全部自然结束且无自身进程后才实际CPU12CP/3全state/12缓存teacher/600初态noise和全部门控标量验收，再运行根describe_final.py并解释新旧任务、SF/FS、负结果和训练代理局限。每个未来阶段必须自己的准入。

下一75分钟依据实际阶段吞吐复查，不把约14.6%条件潜在节时当实际整seed提速，不根据时间推断完成。仅实质变化/失败/完成/需行动通知，健康静默。每轮按根AGENTS经publish.py审查轻量code/report/hash/scalars/outcomes，串行nonforce推MLR核remoteSHA/localreceipt，禁CP/state/RNG/cache/凭据。全部三个seed独立验收、解释报告和发布完成后才删除同一监控。
