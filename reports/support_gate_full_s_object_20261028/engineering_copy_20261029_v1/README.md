# CPU 状态复制优化：版本化交接

用户 2026-10-29 授权：完整训练路径等价验证通过后，在 seed43 启动前安全切换；不打断 seed42，不重跑任何已完成部分。

唯一优化是 CUDA tensor 的 `detach().cpu().clone()` 改为 `detach().cpu()`；CPU tensor 仍 clone，保留独立存储。损失、精度、随机数、优化器、门控、batch、缓存采集和评估不变。63 项冻结源文件不修改，新解释器由外置 wrapper 在导入训练模块前替换复制函数。

`validate_training.py` 使用 seed41 已完成模型/缓存只读副本，在独立诊断进程运行 Task6–9 各一整轮全部 batch 的成对原版/优化版路径。逐块核对每步标量、两候选完整状态、探针、门控及所选状态；强制恢复两分支；另核保存加载和下一步、方向诊断。覆盖完整训练路径和全部尾部短块，不是重新训练 400 轮或生产结果验收。计划额外 3914 次诊断 optimizer 调用，额外方向 autograd 实际记账，0 环境、0 正式接受 epoch。所有成本与 585900 生产训练调用和 273 原准入调用分列。

12:18 的 `guard.py` 只 SIGSTOP 等待中的原 controller3013；没有向 seed42 launcher5055、训练5060或进程组发信号。seed43尚未开始。90分钟 watchdog 在未接管时自动释放原调度器，防止无人持有的停顿。

只有实际 VALIDATION PASS 后才能运行新 controller。它接替 watchdog 监护，等待 seed42 launcher 的内核状态 Z 且 exit_code=0，并核对400轮、40采集和报告，再结束已暂停的原调度器、获取同一根锁。seed42 train_S.exit=0 的来源是实际内核退出状态，另存独立证据。新队列仅 train43 → evaluate41 → evaluate42 → evaluate43；不是重复启动原完整队列。

最终成功来源为本工程目录 `controller.exit=0`，原根 controller.exit 必须保持不存在。最终 CPU 包装需使用本版本实际成功来源及工程合同，并独立核额外诊断账目；不得伪造原 controller 自然完成。新 seed43 每阶段仍执行原自身准入，首次实际阶段完整状态另做复制精确对照。

当前状态必须以远端证据和本地最新交接为准；代码存在或静态检查不代表验证通过或已经切换。此版本禁止重跑部署/guard/controller/validator。
