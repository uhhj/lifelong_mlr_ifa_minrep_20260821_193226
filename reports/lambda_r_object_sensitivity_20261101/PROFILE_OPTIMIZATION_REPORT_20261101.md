# 最新状态：完整路径等价通过，已部署等待自然边界

## 2026-11-01 19:18：完整等价验证通过，安全交接已ARMED，当前训练未切换

capacity_readonly_20261101_191620实际读取VALIDATION PASS：两档λ的Task6–9共8阶段，各完整一轮157/227/309/362个batch，全部每步标量、C/Q分支完整state、探针、门控决定、所选state、保存加载下一步及方向诊断逐项相等。诊断进程3441已退出，实际7828 started==completed、36额外autograd、0环境、0生产epoch，单独记账。原135科学项及诊断使用的只读基线CP/cache SHA精确保留。覆盖完整路径，不等于验证全部400轮，也不是科学最终接受。

已完整审查工程payload及check/audit/record/final的真实完成来源适配。19:17:58唯一部署engineering_device_snapshot_20261101_v1，controller3899/watchdog3900，工程合同b9ebd306e7ac3ce1fba2ce107ba5b2fa6b4fd20f62164790bf7ced4897b6086c、5工程冻结项。禁止再次执行dispatch_handoff、任何原/新controller、验证dispatch或已完成轮。

check_readonly_20261101_191813/LOCAL_AUDIT及ENGINEERING_HANDOFF_LOCAL_AUDIT实际PASS，184739全部epoch/门控/诊断前缀精确保留。原等待调度器2163仅被PID级SIGSTOP，当前launcher2164/训练2169未接收信号且继续运行；新owner3899与watchdog3900为S，ARMED存在、HANDOFF_COMPLETE尚不存在，无FAILURE/WATCHDOG_RELEASE。当前仍L005_41原训练代码，优化尚未用于任何生产run。

新owner等待当前L005_41自然完成400轮40采集、launcher内核Z且exit_code0，并核完整记录/前缀/hash后才结束旧等待调度器、取得同一root锁、记录真实train_S.exit0并接续五个未来run。仅替换各新解释器内四次门控块快照，保持各tensor原设备独立clone，轮末CP和准入仍原CPU格式；每阶段实际BRANCH_COPY_EQ及自身GPU准入保留。原root/controller.exit必须不存在；最终唯一完成来源为engineering_device_snapshot_20261101_v1/controller.exit0且无自身进程。若WATCHDOG_RELEASE出现，应审计并通知，不能宣称交接成功或盲重复部署。

本次L005_41 Task6完整100轮、Task7完整95轮，共195轮37265接受58830生产、33自身隔离19准入autograd、10采集0正式；其他五run未开始。当前Task7近10轮57.1295秒、全95轮110.8830秒，尚余5轮约4.76分钟纯计算。Task8/9未测，按批次数比例的当前整run剩余纯计算条件范围286–556分钟，保存/准备/余30采集另计，不能将近期变快归因于尚未启用的优化。全部六run目前没有可靠完成日期。下一240分钟在当前整run结束预计较早侧复查，覆盖旧20分钟验证等待间隔，不固定短轮询。详见本快照TIMING_AUDIT。

运输使用旧目录.session_control现有worker28064，先读status核PID，不重启活跃worker或输出凭据。以后只用工程目录check_readonly.py复制至旧目录queue；按result实际目录复制回本目录，再工程audit_progress.py NEW PREVIOUS→工程record_progress.py NEW INTERVAL。根旧记录器/旧final包装不再适用。工程final包装仍仅静态，全部自然完成后才实际CPU24CP/6恢复/24teacher/1200noise/54600门控验收，随后根compare_primary_s和完整解释发布。7828诊断调用/36autograd必须单列，不混科学预算。原源、原证据及失败记录保留。

只做OBJECT λ=.05/.2各三seed，已有.1只读；科学损失、门控阈值、批次、采样及预算不变。两成潜在整体节时仍是条件推测，不能用92.05%复制微基准节时冒充整run提速。健康静默，只实质变化/完成/新失败/需行动通知；全部六run独立验收解释发布后才删除同一object-s-r-run监控。


---

以下为18:53以前定位与验证历史，进度由上文覆盖。

# 2026-11-01 状态搬运优化：定位完成，完整路径验证进行中

目前已找到可优化的具体环节，但尚未将候选代码用于生产。当前 L005_41 始终由原训练进程2169运行，没有收到暂停、终止或重启信号；135科学冻结项不变，18:47:41完整审计确认全部历史epoch/门控前缀精确保留。其他五个run尚未启动。

## 实测定位

独立 py-spy 0.4.2 使用 `--nonblocking --idle --threads --rate 10`，连续三个60秒窗口。共1797个主线程样本，报告采样错误0；非阻塞读取仍可能产生不完整栈，不能把比例当精确函数计时。工具是从PyPI下载并核SHA的独立二进制，没有安装或升级训练环境，没有注入训练代码或读取局部变量。官方非阻塞行为见 https://github.com/benfred/py-spy#how-can-i-avoid-pausing-the-python-program 。

|类别|主线程样本占比|
|---|---:|
|C/Q候选更新整体|61.94%|
|其中反向传播调用|29.60%（包含在上一行中）|
|其中优化器执行|7.74%（包含在候选更新中）|
|其中梯度有限值检查|5.40%（包含在候选更新中）|
|完整状态快照复制|12.02%|
|完整状态恢复|5.62%|
|状态一致性比较|3.84%|
|门控探针|6.79%|
|检查点保存|2.56%|
|主batch加载|1.17%|
|其他或不完整栈|6.07%|

因此，减少保存频率或增加数据加载线程没有依据作为首选。反向传播栈包含原生CUDA执行或等待，本次不暂停进程，无法将它继续分解为GPU内核、驱动等待和原生CPU函数；仍不能证明几倍慢速完全由某一个资源限制引起。

证据：capacity_readonly_20261101_183133/EVIDENCE.json及PROFILE_LOCAL_AUDIT.json。前次资源采样未发现可见CPU限流或GPU热/功率限频，观察到IO等待，但它不单独解释轮内耗时变化。

## 已实际测试的候选

仅将每个门控块内四次状态快照由“转CPU并独立保存”改为“在各tensor原设备独立clone”。CPU来源仍独立clone；RNG、模型、优化器、scheduler、梯度、training mode和requires_grad均完整保留。生产更新公式、batch、候选数量、门控指标和阈值不变。轮末检查点和准入仍使用原CPU快照，保存格式不变。

独立进程以当前已保存完整state进行10轮交替CPU/device复制—恢复微基准，0优化器调用、0环境回合，复制恢复后的全部state数值逐项相等。每轮四次capture、两次restore及一次一致性检查：

- 原CPU路径均值5.094秒、中位6.425秒。
- 原设备路径均值0.405秒、中位0.477秒。
- 十对均更快，微基准平均耗时减少92.05%。

这不是整轮提速92%：它只针对采样中约21.48%的状态管理开销。即便该比例可外推，潜在整体节时也只是约两成的量级，实际值须在生产自然切换后观测。微基准共享当前GPU，时间有波动，不是隔离硬件性能结论。单独诊断最大allocated显存约1.76GB。

证据：capacity_readonly_20261101_184108/EVIDENCE.json及MICROBENCH_LOCAL_AUDIT.json。首次运输因模型构造stdout日志混入JSON而解析失败，远端实际exit0及RESULT已保留；之后只读取既有RESULT，没有重复微基准。原失败ATTEMPT在capacity_readonly_20261101_183808保留。

## 完整训练等价验证（未完成）

18:42:53唯一启动诊断进程3441，目录 `/data/mlr_ifa/diagnostics/lambda_r_branch_training_20261101_v1`，入口SHA `c9f511fff1d40adcb8e95b2b69ffbb1adbcbcac155044894932b4bc370c4e247`。禁止重复dispatch。

验证覆盖两档λ各Task6–9的一整轮所有157/227/309/362个batch，分别从共同完整state比较原路径和候选：全部每步标量、C/Q两分支完整state、探针、门控选择、所选state，以及保存加载下一步和方向诊断。计划7828隔离优化器调用、最多36额外autograd、0环境、0生产epoch。已有S的CP和缓存仅作隔离诊断输入，不会复制到新run生产缓存。

18:50:35实际ledger：λ=.05 Task6整轮通过；Task7已有22块/220batch比较相等。已持久记录1198完成诊断调用及6额外autograd；当前可能正在下一块，因此该数不是最终实际消耗。生产继续训练。完整验证未PASS，不能套微基准结果或提前声称可安全切换。验证共享GPU，会暂时影响生产墙钟吞吐；不改变生产state/RNG或科学预算。

## 下一步执行要求

同一监控20分钟检查已有诊断ledger，不重复启动。只有真实VALIDATION.json为PASS、8个阶段覆盖齐全、7828开始/完成调用一致、诊断进程已退出且基线/科学hash一致，才继续版本化部署。

本地engineering_device_snapshot_20261101_v1目前仅静态准备：train_wrapper仅AST替换train.main内93/95/98/100行四个capture，未修改冻结源；controller/watchdog只能在原等待调度器边界工作，不信号当前训练。当前L005_41完整400轮/40采集自然结束、launcher内核exit0和历史前缀验真后，才用新解释器启动后续五run。原root/controller.exit必须保持不存在，新完成来源改为工程controller.exit。真实部署尚未执行。

部署前必须再审查这些静态脚本与真实完成来源适配的check/audit/record/final包装；验证失败保留全部已耗成本和错误，不切换、不重训生产。最终CPU验收仍必须真实执行，不套STATIC_ONLY。发布与监控保持到六run及最终报告完成。
