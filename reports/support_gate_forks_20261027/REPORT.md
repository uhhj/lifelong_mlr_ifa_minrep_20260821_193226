## 用户最新授权变更：2026-10-28 19:37 CST

停止剩余机制实验，保留已有结果，立即优先最新完整S OBJECT41/42/43。已实停所有机制进程，S100轮、Q3轮保留，GOAL12组及OBJECT C共14组完整训练；G/U与机制1120正式评估取消。不得恢复机制队列。原协议仅作历史，机制验收不再是新任务前置条件。停止证据deploy_remote_20261028_193702/EVIDENCE.json绑定所有recovery SHA，部分轮未记账成本仍UNKNOWN。新工作目录 ../20261028_support_gate_full_s_object；新完整方法训练/采集/正式评估协议不变，独立合同和准入；当前正在构建，尚未部署。保留同一监控并改指新任务。

本次用户询问总耗时：ETA_PLANNING.json以恢复后S最近10轮118.006秒、历史GOAL的G/S比值作跨case外推，剩余训练粗估15–20小时，含1120正式评估及验收的18–30小时仅作低置信度排期范围，不是实测全队列ETA或保证上界。Q/G/U及当前主机正式评估尚无自身吞吐，出现新实测即替换；不含后续完整S OBJECT三个seed。

## 最新 2026-10-28T19:09:24.815070+08:00：恢复后进度复查

用户“已启动”后，16:17 SSH/GPU恢复，16:19 CPU/PCI/驱动诊断通过。16:28:12已唯一启动版本化 engineering_resume_20261028_v1/controller_resume.py，controller2020、GPU进程2037。禁止再次部署/启动原controller或本恢复入口。工程合同 f853cc7498f0a634e8d8a5c631630bfd0353452d525bfda5f14c0721bdb77050，原56冻结项及工程4项SHA实际通过。原13组完成epoch文件与S前7轮逐字段精确保留。

最新实际快照 check_readonly_20261028_190924，GPU完整state恢复、原29调用准入及下一步双重重放实际PASS，诊断后恢复原checkpoint全state，再从第8轮继续。S现已完整82/100轮；累计1382完整epoch、287884接受、343384已记账训练调用，原隔离406加本次31=共437，恢复额外autograd11、环境0，正式0/1120。中断时未记账部分轮物理成本仍UNKNOWN，不能声称总实际成本精确或为零。原checkpoint已另存私有 interrupted_20261028_epoch7.pth，原失败日志/工程attempt保留。

恢复后已完成75轮，最近至多10轮均118.006426秒；恢复后轮时有波动，不沿用旧50分钟预测。当前线性纯计算估计35.40分钟仅作暂定，不含保存/准备或Q/G/U与正式评估。下一45分钟在当前S整run预计结束附近复查，再按实际当前run吞吐更新同一监控，不恢复固定短周期；这是检查安排而非完成承诺。单次GPU利用率采样不能判断持续瓶颈。容量已把额外346305333B私有备份计入峰值，root留1GiB后余363790336B，data扣剩余和2GiB后余34256370891B。

后续队列为S剩余→原Q/G/U各自准入与100轮→原4case正式评估。原根controller.exit保持不存在；最终成功标记改为 engineering_resume_20261028_v1/controller.exit==0 且无自身进程。现有 final_readonly.py/audit_final_local.py/describe_final.py 尚需外置版本化适配这一真实完成来源和恢复额外31调用、UNKNOWN成本，不能伪造根exit；仅全部训练/评估完成后才执行真实最终验收。当前record_progress.py仍硬编码旧deployment，不使用；使用audit_progress→audit_transition→engineering_resume_20261028_v1/audit_resume.py，再record_resume.py SNAPSHOT INTERVAL记录。capacity_readonly.py仍固定next_epoch7，续训后不要原样执行。

此次工程恢复未改变科学损失/阈值/采样/预算，没有重训任何已完成轮；前7轮记录相同不等于能补回中断未保存部分成本。整体PARTIAL_RESUMED_TRAINING，科学最终验收与闭环结论待后续。机制验收分析发布后仍须构建部署并完成仅最新完整S OBJECT41/42/43，不能删除监控。本段覆盖此前失联/GPU不可见与原controller29838活跃描述；下方历史记录只作追溯。

## 最新 2026-10-28T19:05:25.905987+08:00：恢复后进度复查

用户“已启动”后，16:17 SSH/GPU恢复，16:19 CPU/PCI/驱动诊断通过。16:28:12已唯一启动版本化 engineering_resume_20261028_v1/controller_resume.py，controller2020、GPU进程2037。禁止再次部署/启动原controller或本恢复入口。工程合同 f853cc7498f0a634e8d8a5c631630bfd0353452d525bfda5f14c0721bdb77050，原56冻结项及工程4项SHA实际通过。原13组完成epoch文件与S前7轮逐字段精确保留。

最新实际快照 check_readonly_20261028_190524，GPU完整state恢复、原29调用准入及下一步双重重放实际PASS，诊断后恢复原checkpoint全state，再从第8轮继续。S现已完整80/100轮；累计1380完整epoch、287160接受、342660已记账训练调用，原隔离406加本次31=共437，恢复额外autograd11、环境0，正式0/1120。中断时未记账部分轮物理成本仍UNKNOWN，不能声称总实际成本精确或为零。原checkpoint已另存私有 interrupted_20261028_epoch7.pth，原失败日志/工程attempt保留。

恢复后已完成73轮，最近至多10轮均125.832844秒；恢复后轮时有波动，不沿用旧50分钟预测。当前线性纯计算估计41.94分钟仅作暂定，不含保存/准备或Q/G/U与正式评估。下一50分钟在当前S整run预计结束附近复查，再按实际当前run吞吐更新同一监控，不恢复固定短周期；这是检查安排而非完成承诺。单次GPU利用率采样不能判断持续瓶颈。容量已把额外346305333B私有备份计入峰值，root留1GiB后余363790336B，data扣剩余和2GiB后余34256370891B。

后续队列为S剩余→原Q/G/U各自准入与100轮→原4case正式评估。原根controller.exit保持不存在；最终成功标记改为 engineering_resume_20261028_v1/controller.exit==0 且无自身进程。现有 final_readonly.py/audit_final_local.py/describe_final.py 尚需外置版本化适配这一真实完成来源和恢复额外31调用、UNKNOWN成本，不能伪造根exit；仅全部训练/评估完成后才执行真实最终验收。当前record_progress.py仍硬编码旧deployment，不使用；使用audit_progress→audit_transition→engineering_resume_20261028_v1/audit_resume.py，再record_resume.py SNAPSHOT INTERVAL记录。capacity_readonly.py仍固定next_epoch7，续训后不要原样执行。

此次工程恢复未改变科学损失/阈值/采样/预算，没有重训任何已完成轮；前7轮记录相同不等于能补回中断未保存部分成本。整体PARTIAL_RESUMED_TRAINING，科学最终验收与闭环结论待后续。机制验收分析发布后仍须构建部署并完成仅最新完整S OBJECT41/42/43，不能删除监控。本段覆盖此前失联/GPU不可见与原controller29838活跃描述；下方历史记录只作追溯。

## 最新 2026-10-28T17:16:18.286101+08:00：恢复后进度复查

用户“已启动”后，16:17 SSH/GPU恢复，16:19 CPU/PCI/驱动诊断通过。16:28:12已唯一启动版本化 engineering_resume_20261028_v1/controller_resume.py，controller2020、GPU进程2037。禁止再次部署/启动原controller或本恢复入口。工程合同 f853cc7498f0a634e8d8a5c631630bfd0353452d525bfda5f14c0721bdb77050，原56冻结项及工程4项SHA实际通过。原13组完成epoch文件与S前7轮逐字段精确保留。

最新实际快照 check_readonly_20261028_171617，GPU完整state恢复、原29调用准入及下一步双重重放实际PASS，诊断后恢复原checkpoint全state，再从第8轮继续。S现已完整25/100轮；累计1325完整epoch、267250接受、322750已记账训练调用，原隔离406加本次31=共437，恢复额外autograd11、环境0，正式0/1120。中断时未记账部分轮物理成本仍UNKNOWN，不能声称总实际成本精确或为零。原checkpoint已另存私有 interrupted_20261028_epoch7.pth，原失败日志/工程attempt保留。

恢复后已完成18轮，最近至多10轮均97.923692秒；恢复后轮时有波动，不沿用旧50分钟预测。当前线性纯计算估计122.40分钟仅作暂定，不含保存/准备或Q/G/U与正式评估。下一130分钟在当前S整run预计结束附近复查，再按实际当前run吞吐更新同一监控，不恢复固定短周期；这是检查安排而非完成承诺。单次GPU利用率采样不能判断持续瓶颈。容量已把额外346305333B私有备份计入峰值，root留1GiB后余363806720B，data扣剩余和2GiB后余34256436427B。

后续队列为S剩余→原Q/G/U各自准入与100轮→原4case正式评估。原根controller.exit保持不存在；最终成功标记改为 engineering_resume_20261028_v1/controller.exit==0 且无自身进程。现有 final_readonly.py/audit_final_local.py/describe_final.py 尚需外置版本化适配这一真实完成来源和恢复额外31调用、UNKNOWN成本，不能伪造根exit；仅全部训练/评估完成后才执行真实最终验收。当前record_progress.py仍硬编码旧deployment，不使用；使用audit_progress→audit_transition→engineering_resume_20261028_v1/audit_resume.py，再record_resume.py SNAPSHOT INTERVAL记录。capacity_readonly.py仍固定next_epoch7，续训后不要原样执行。

此次工程恢复未改变科学损失/阈值/采样/预算，没有重训任何已完成轮；前7轮记录相同不等于能补回中断未保存部分成本。整体PARTIAL_RESUMED_TRAINING，科学最终验收与闭环结论待后续。机制验收分析发布后仍须构建部署并完成仅最新完整S OBJECT41/42/43，不能删除监控。本段覆盖此前失联/GPU不可见与原controller29838活跃描述；下方历史记录只作追溯。

## 最新 2026-10-28T16:38:09.868882+08:00：GPU恢复并已实际续训

用户“已启动”后，16:17 SSH/GPU恢复，16:19 CPU/PCI/驱动诊断通过。16:28:12已唯一启动版本化 engineering_resume_20261028_v1/controller_resume.py，controller2020、GPU进程2037。禁止再次部署/启动原controller或本恢复入口。工程合同 f853cc7498f0a634e8d8a5c631630bfd0353452d525bfda5f14c0721bdb77050，原56冻结项及工程4项SHA实际通过。原13组完成epoch文件与S前7轮逐字段精确保留。

最新实际快照 check_readonly_20261028_163808，GPU完整state恢复、原29调用准入及下一步双重重放实际PASS，诊断后恢复原checkpoint全state，再从第8轮继续。S现已完整8/100轮；累计1308完整epoch、261096接受、316596已记账训练调用，原隔离406加本次31=共437，恢复额外autograd11、环境0，正式0/1120。中断时未记账部分轮物理成本仍UNKNOWN，不能声称总实际成本精确或为零。原checkpoint已另存私有 interrupted_20261028_epoch7.pth，原失败日志/工程attempt保留。

恢复后已完成1轮，最近均201.645410秒，明显慢于原环境；单轮样本不足，不沿用旧50分钟预测。当前线性纯计算估计309.19分钟仅作暂定，不含保存/准备或Q/G/U与正式评估。下一30分钟复查以核实恢复后稳定吞吐，再按当前整run剩余时间更新同一监控，不恢复固定短周期。单次GPU利用率采样不能判断持续瓶颈。容量已把额外346305333B私有备份计入峰值，root留1GiB后余363806720B，data扣剩余和2GiB后余34254499019B。

后续队列为S剩余→原Q/G/U各自准入与100轮→原4case正式评估。原根controller.exit保持不存在；最终成功标记改为 engineering_resume_20261028_v1/controller.exit==0 且无自身进程。现有 final_readonly.py/audit_final_local.py/describe_final.py 尚需外置版本化适配这一真实完成来源和恢复额外31调用、UNKNOWN成本，不能伪造根exit；仅全部训练/评估完成后才执行真实最终验收。当前record_progress.py仍硬编码旧deployment，不使用；使用audit_progress→audit_transition→engineering_resume_20261028_v1/audit_resume.py，再record_resume.py SNAPSHOT INTERVAL记录。capacity_readonly.py仍固定next_epoch7，续训后不要原样执行。

此次工程恢复未改变科学损失/阈值/采样/预算，没有重训任何已完成轮；前7轮记录相同不等于能补回中断未保存部分成本。整体PARTIAL_RESUMED_TRAINING，科学最终验收与闭环结论待后续。机制验收分析发布后仍须构建部署并完成仅最新完整S OBJECT41/42/43，不能删除监控。本段覆盖此前失联/GPU不可见与原controller29838活跃描述；下方历史记录只作追溯。

用户要求继续执行后，16:15:31 CST再次只读复查；实际目录check_readonly_20261028_161526，仍在SSH认证前失败（Error reading SSH protocol banner），authenticated=false、remote_commands=0。错误SHA见connectivity_readonly_20261028_1615/LOCAL_AUDIT.json。worker4572保持READY_IN_MEMORY且存活，未重启。没有新GPU/进程/hash/容量观察，最后核实仍12:30训练中断、GPU不可见、OBJECT41 S保存7轮，13组训练完成、1307完整epoch；原接受/物理账目和部分轮成本未知标记保持。当前PARTIAL_REMOTE_OBSERVATION_UNAVAILABLE，尚未续训；服务器SSH及GPU须恢复后才能核对并执行版本化恢复。保留同一60分钟监控；恢复后先核冻结hash、进程、原失败与完整恢复，再从第8轮继续，禁止覆盖已完成阶段。全部机制及后续仅最新完整S OBJECT三seed任务仍有效。

15:59:39 CST只读复查仍在SSH认证前失败，authenticated=false、remote_commands=0，实际目录check_readonly_20261028_155933；错误SHA绑定于connectivity_readonly_20261028_1559/LOCAL_AUDIT.json。worker4572仍READY_IN_MEMORY且存活，未重启。没有新GPU/进程/hash/容量观察；最新真实状态仍为12:30训练中断、GPU不可见、OBJECT41 S保存7轮，总1307完整epoch，原账目与中断部分轮成本未知标记均保留。保持PARTIAL_REMOTE_OBSERVATION_UNAVAILABLE，未恢复/重训/改参/删除/付费。继续同一监控60分钟复查；相同基础设施故障无新需行动情况静默。恢复后先核冻结hash、进程、失败证据与完整状态，再版本化从第8轮续跑，全部机制及后续仅最新完整S OBJECT三seed授权保持。

14:59:23 CST只读复查仍在SSH认证前失败，authenticated=false、remote_commands=0，实际目录check_readonly_20261028_145918；错误SHA见connectivity_readonly_20261028_1459/LOCAL_AUDIT.json。worker4572仍READY_IN_MEMORY且存活，未重启。没有新的服务器GPU/进程/hash/容量观察，最后核实仍为12:30训练中断、GPU不可见、OBJECT41 S保存7轮，总1307完整epoch；原账目及中断部分轮成本未知的标记保留。当前PARTIAL_REMOTE_OBSERVATION_UNAVAILABLE，未恢复或重训。继续同一监控60分钟复查传输和GPU；持续同一基础设施故障无新需行动情况静默。连接与GPU恢复后先核冻结hash、进程、原失败及完整恢复，再版本化从第8轮继续；全部机制及后续仅最新完整S OBJECT三seed仍须完成。

13:59:59 CST只读复查仍在SSH认证前失败（Error reading SSH protocol banner），authenticated=false、remote_commands=0，实际目录check_readonly_20261028_135954；错误SHA绑定于connectivity_readonly_20261028_1359/LOCAL_AUDIT.json。worker4572仍READY_IN_MEMORY且存活，未重启。当前GPU、进程、容量和训练状态无法重新观察；最后核实仍为12:30训练中断、GPU不可见，OBJECT41 S保存7轮，13组训练完成、1307完整epoch、260734接受更新、316234已记录物理调用、406隔离调用、正式0。中断部分轮成本仍未知，未恢复训练。保持PARTIAL_REMOTE_OBSERVATION_UNAVAILABLE，保留原中断和CPU恢复证据；继续同一监控60分钟复查，已知基础设施故障无新需行动情况静默。恢复后先核进程/hash/完整状态和原失败成本，再版本化从第8轮续跑；全部机制及随后仅最新完整S OBJECT三seed授权不变。

12:59:41 CST本轮只读请求再次在SSH认证前失败，authenticated=false/remote_commands=0；实际目录check_readonly_20261028_125936，错误SHA见connectivity_readonly_20261028_1259/LOCAL_AUDIT。worker4572仍READY_IN_MEMORY且进程存在，未重启。本轮无法重新确认GPU/进程/容量；最后真实快照仍12:30的check_readonly_20261028_122959及12:31 CPU诊断：训练中断、GPU不可见、OBJECT41 S保存7轮，13组训练完成，总1307完整epoch/260734接受/316234已记录物理调用、406准入、正式0。恢复文件SHA和先前中断事实保留，不将通信失败解释为新的训练变化。当前状态PARTIAL_REMOTE_OBSERVATION_UNAVAILABLE，上一已核状态PARTIAL_INTERRUPTED_GPU_UNAVAILABLE。继续同一监控60分钟复查传输与GPU，无重训/恢复/修改科学源/删除/付费。该已知基础设施故障暂无新增用户操作要求，保持静默。连接与GPU恢复后先核冻结hash、进程、原失败与完整恢复，再版本化从第8轮续跑；所有机制及后续仅最新完整S OBJECT三seed任务保持。

12:30 CST SSH已恢复，实际check_readonly_20261028_122959：56冻结项及071915全部已记录epoch/门控前缀精确保留，但实验controller/训练进程为空；OBJECT41 S日志有SIGHUP，controller.exit与train_S.exit均不存在。nvidia-smi返回9、无法连接驱动；12:31只读capacity_readonly_20261028_123122确认NVIDIA PCI设备为空、驱动模块为空、仅nvidiactl节点。不能称正常完成。主机uptime110秒提示近期启动，不能据此确定先前中断原因。状态PARTIAL_INTERRUPTED_GPU_UNAVAILABLE。 GOAL三seed12组及OBJECT41 C共13组各100轮完成；OBJECT41 S完整7/100轮，Q/G/U未启动，正式0/1120。完整epoch合计1307、接受260734、对应已记账物理调用316234、隔离406；中断部分epoch未记录成本未知，不把它写成零。S recovery.pth已真实CPU加载，next_epoch7/step71834、2534物理调用、history与7轮账目精确相同，模型/优化器/scheduler/grads/modes/requires和Python/NumPy/CPU/CUDA RNG字段齐全，父CP与probe SHA通过，SHA df2c9101552a50f0478b36b1df942166c614b598a6d11af0d595fb4c9da7bb97；这尚不是GPU恢复验收。 本轮仅版本化修改本地只读check包装，使GPU查询失败仍能保存其他证据，原v1与122908失败保留；冻结远端训练源未改，未重启/重训/重评/删除/付费。GPU当前不可见，需服务器侧恢复GPU分配/驱动后才能续跑；监控60分钟检查可见性，恢复后先核进程/hash/完整恢复和原失败成本，版本化实现从第8轮继续，不运行原controller或train入口覆盖完成阶段。全部机制验收分析发布后仍必须执行仅最新完整S OBJECT41/42/43。

11:54:59 CST只读请求仍在SSH认证前失败（Error reading SSH protocol banner），authenticated=false、remote_commands=0。实际目录check_readonly_20261028_115454与请求秒数不同，已按本地清单定位；新connectivity_readonly_20261028_1154/LOCAL_AUDIT绑定实际错误SHA。worker4572仍READY_IN_MEMORY且进程存活，未重启。当前服务器训练/评估、进程、冻结hash和容量未知；最后真实训练观察仍07:19:17的071915快照，1304完整epoch和原账目保持不变，旧ETA无效。保持PARTIAL_REMOTE_OBSERVATION_UNAVAILABLE，未新增科学运行/恢复/改参/删除/付费。继续同一监控60分钟传输复查，已有通信故障无新需用户行动情况保持静默。恢复后先检查原进程、冻结hash、退出及完整状态，并对071915审计前缀；不盲重启。机制全部对照及其后仅最新完整S OBJECT41/42/43仍待完成，保留监控。

10:55:38 CST只读复查仍在SSH认证前失败，authenticated=false、remote_commands=0，实际目录check_readonly_20261028_105532；原错误SHA绑定于connectivity_readonly_20261028_1055/LOCAL_AUDIT.json。worker4572仍READY_IN_MEMORY且进程存活，未重启。当前训练/评估、进程、冻结hash、容量均未知；最后成功训练观察仍07:19:17的071915快照（1304完整epoch），旧ETA无效，不推算完成。保持PARTIAL_REMOTE_OBSERVATION_UNAVAILABLE，未操作科学运行，继续同一监控60分钟传输恢复复查。恢复后先审计071915到新快照前缀/进程/hash/退出及完整状态，禁止盲重启。机制全部对照和后续仅最新完整S OBJECT41/42/43仍须完成；已有通信故障无新需行动情况保持静默。

连接复查 2026-10-28T09:57:42.421798+08:00：SSH仍在认证前失败，远程命令0，当前实验状态未知。已尝试查找可用服务器管理页面，但电脑操作工具初始化因本机kernel assets路径缺失失败，未访问或修改服务商页面。worker4572保持，原训练未操作；错误SHA见connectivity_readonly_20261028_0957/LOCAL_AUDIT.json。继续60分钟传输复查。

08:55:32 CST复查：SSH仍在认证前失败，remote_commands=0，worker4572未重启。最新训练观察仍为07:19:17，当前状态未知；原错误保留在check_readonly_20261028_085527/ATTEMPT.json，新绑定见connectivity_readonly_20261028_0855/LOCAL_AUDIT.json。将传输复查退避为60分钟，旧训练ETA已标失效。

本轮连接状态：PARTIAL_REMOTE_OBSERVATION_UNAVAILABLE（2026-10-28T08:24:32.285406+08:00）。两次SSH在认证前失败，远程命令0；独立端口检查TCP连接成功但读取0字节即断开，未返回SSH banner。不能据此判断训练失败或停止。以下07:19:17为最后成功观察，非当前进度；未重启worker/训练、未改参。原失败证据和SHA见connectivity_readonly_20261028_0820/LOCAL_AUDIT.json。30分钟后复查传输，恢复后先检查冻结hash、进程和原账本，不盲重启。

# 支持度与门控共同起点机制实验

状态：PARTIAL_TRAINING；观察 2026-10-28T07:19:17.083839+08:00。

已实际接受完整epoch账目 1304；主更新 259648；候选实际训练优化器调用 315148；隔离准入 406；正式环境 0/1120。尚无科学收益或最终验收结论。

{
  "GOAL_41": {
    "C": {
      "epochs": 100,
      "accepted_updates": 18500,
      "physical_optimizer_calls": 18500,
      "last_epoch_seconds": 8.987632513046265
    },
    "S": {
      "epochs": 100,
      "accepted_updates": 18500,
      "physical_optimizer_calls": 18500,
      "last_epoch_seconds": 15.96706748008728
    },
    "Q": {
      "epochs": 100,
      "accepted_updates": 18500,
      "physical_optimizer_calls": 18500,
      "last_epoch_seconds": 14.320943355560303
    },
    "G": {
      "epochs": 100,
      "accepted_updates": 18500,
      "physical_optimizer_calls": 37000,
      "last_epoch_seconds": 37.433425188064575
    },
    "gate_blocks_observed": 1900,
    "gate_Q_accepts": 1085
  },
  "GOAL_42": {
    "C": {
      "epochs": 100,
      "accepted_updates": 18500,
      "physical_optimizer_calls": 18500,
      "last_epoch_seconds": 9.262686491012573
    },
    "S": {
      "epochs": 100,
      "accepted_updates": 18500,
      "physical_optimizer_calls": 18500,
      "last_epoch_seconds": 15.774545192718506
    },
    "Q": {
      "epochs": 100,
      "accepted_updates": 18500,
      "physical_optimizer_calls": 18500,
      "last_epoch_seconds": 14.738142251968384
    },
    "G": {
      "epochs": 100,
      "accepted_updates": 18500,
      "physical_optimizer_calls": 37000,
      "last_epoch_seconds": 43.64219784736633
    },
    "gate_blocks_observed": 1900,
    "gate_Q_accepts": 764
  },
  "GOAL_43": {
    "C": {
      "epochs": 100,
      "accepted_updates": 18500,
      "physical_optimizer_calls": 18500,
      "last_epoch_seconds": 8.96446180343628
    },
    "S": {
      "epochs": 100,
      "accepted_updates": 18500,
      "physical_optimizer_calls": 18500,
      "last_epoch_seconds": 15.5825777053833
    },
    "Q": {
      "epochs": 100,
      "accepted_updates": 18500,
      "physical_optimizer_calls": 18500,
      "last_epoch_seconds": 14.856539487838745
    },
    "G": {
      "epochs": 100,
      "accepted_updates": 18500,
      "physical_optimizer_calls": 37000,
      "last_epoch_seconds": 44.89176559448242
    },
    "gate_blocks_observed": 1900,
    "gate_Q_accepts": 1111
  },
  "OBJECT_41": {
    "C": {
      "epochs": 100,
      "accepted_updates": 36200,
      "physical_optimizer_calls": 36200,
      "last_epoch_seconds": 17.098241329193115
    },
    "S": {
      "epochs": 4,
      "accepted_updates": 1448,
      "physical_optimizer_calls": 1448,
      "last_epoch_seconds": 31.040600299835205
    },
    "Q": {
      "epochs": 0,
      "accepted_updates": 0,
      "physical_optimizer_calls": 0,
      "last_epoch_seconds": null
    },
    "G": {
      "epochs": 0,
      "accepted_updates": 0,
      "physical_optimizer_calls": 0,
      "last_epoch_seconds": null
    },
    "U": {
      "epochs": 0,
      "accepted_updates": 0,
      "physical_optimizer_calls": 0,
      "last_epoch_seconds": null
    }
  }
}

{
  "case": "OBJECT_41",
  "arm": "S",
  "observed_epochs": 4,
  "recent_epoch_seconds": 31.42976540327072,
  "remaining_training_minutes": 50.28762464523315,
  "preparation_saving_extra": true,
  "not_total_queue_prediction": true
}

首组GPU真实准入已证明lambda=0与原C完整更新相等、q=1与原S完整更新相等、10步C重放、完整恢复及探针/梯度测量不改变训练状态；其余组必须各自通过。额外候选和探针计算单列，不称等算力。

机制实验保留C/S/Q/G四组与OBJECT41的U开发对照。之后仅最新完整S运行OBJECT41/42/43完整序列，旧方法不新增运行；不得将机制阶段结束误作用户全部任务完成。参见PROTOCOL和IMPLEMENTATION_NOTES。

本地一次辅助编辑命令因PowerShell引号产生Python解析错误，未执行目标修改；部署仍使用完整已审科学PROTOCOL，未改科学源码或预算。最终外置wrappers仅编译，须实际运行验收。
