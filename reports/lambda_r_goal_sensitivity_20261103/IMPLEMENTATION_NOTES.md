# GOAL λ_R 六 run：工程适配

这是独立 GOAL 实验目录。当前仅完成本地源文件准备；未冻结、未部署、未启动训练或 GPU 验证。不得把静态解析或 OBJECT 的验证结果写成 GOAL 已通过。

科学源来自已接受 GOAL S 的各 seed。runtime 的实际100轮调度和独立调用计数复用已验证的 GOAL Q 启动修复，保留 S 双候选更新。只把保持合成系数改为0.05/0.2；同λ均匀支持准入、方向诊断随之使用同一系数。RA、模型、loss其余项、门控阈值/10步块、采样及各任务100轮不变。

`controller_parallel.py` 是新的单次调度入口，不含 OBJECT 的冻结、接管、恢复或信号代码。至多两个独立 world_size=1 训练，每次显式 c10d localhost:0 和唯一 UUID。0.05三seed优先，空槽允许0.2 seed41；0.05训练齐全后优先其600正式，已启动0.2 seed41继续；随后0.2另外两seed和最后600正式。失败保留，已运行子进程自然结束，不重启或杀掉。

`environment_locked.py` 在准确原collect/evaluate命令前获取同一个GOAL环境flock；保留原解释器、脚本argv、cwd和环境。无需写入site-packages，不继承OBJECT的pth。必须实际零环境模拟争用核验，生产中合并审计24次采集和6次正式的acquired/released区间。

`train_wrapper.py` 仅替换三个候选分支快照为独立同设备clone、所选状态检查为同设备完整比较、有限值判断为合并布尔检查。保存、准入、采集前后状态仍用CPU格式。每个真实阶段必须实际COPY_EQ、BRANCH_COPY_EQ、首更新CHECK_EQ及自己的GPU准入。设备检查和clone不是减少梯度更新或改变loss。

GOAL独立验证源 `engineering_validation_20261103_v1/validate.py` 比较原CPU快照/检查与优化路径，两个λ各Task6–9完整一轮全部batch。计划5704隔离optimizer、最多36额外autograd、0环境、0生产epoch。逐batch标量、两候选完整state/恢复、探针、决策、所选state及保存加载下一步逐位相等，started/completed必须一致。诊断只读已接受GOAL seed41 CP/cache作为输入，不把这些缓存用于新的生产run。此路径验证不等于全部未来seed轨迹证明，每阶段自身准入继续。

硬件是用户授权的RTX3090，历史0.1来自先前硬件。官方初始化tensor/RNG及软件精度开关严格相等；记录真实Task6与历史0.1的tensor比较及硬件差异，不提前声明相等，不声称λ单因素完全受控。没有继承OBJECT的归一化1ULP例外；GOAL最终CPU所有scale默认仍须精确。

部署硬边界：OBJECT全六实际接受、解释报告发布、按SHA清理自有hook；GOAL实际容量预检通过；GOAL独立完整路径验证和环境锁模拟通过；source/validation SHA一致；完整源码及最终包装审查。然后才唯一freeze/deploy。禁止重复compose覆盖当前适配，禁止重复诊断或已完成科学运行。

未来最终接受仍需真实本root/controller.exit0、无自身进程、24CP/6完整恢复/24teacher缓存/1200初态noise/40200门控与853200生产调用验收。两组分别与同一历史0.1的600回合配对。先前工程诊断、开发集复用、预训练来源未闭合均明确另记。
