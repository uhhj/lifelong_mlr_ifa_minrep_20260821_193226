## 2026-11-09T22:41:31.150910+08:00 最新只读检查

check_readonly_20261109_224130及进度审计PASS，215139全部前缀保持。OBJECT41/42完整400轮40采集退出0；43已338轮(Task9 38)，30采集，Task9自己的固定q1准入29/11、恢复重放和四阶段COPY_EQ/BRANCH_COPY_EQ/CHECK_EQ实际通过。总1138轮541012生产110采集0正式，GOAL未开始；148科学+7工程冻结项不变，controller2416/2417/GPU3049正常，11采集互斥完整。

43近10轮56.8189秒，余62轮纯58.71min，训练保存最后10采集规划65–85min；OBJECT600正式60–90min，OBJECT整体125–175min/11月10日00:45–01:40；六run总525–685min/07:30–10:10，CPU验收解释发布另计。GOAL及正式未实测，不按时间断言完成；新resume目录本快照TIMING_AUDIT/PROGRESS_REPORT为证。严格OBJECT闭环完成才GOAL，未改队列。

## 2026-11-09T21:51:40.308442+08:00 实际进度与估时

check_readonly_20261109_215139与audit_progress PASS，190024全部前缀保持。OBJECT41/42各400轮40采集且新工程Q1_41_train.exit/Q1_42_train.exit真实0；43自然接续282轮(Task8 82)、20采集，自己的Task7/8固定q1准入与COPY_EQ/BRANCH_COPY_EQ/CHECK_EQ通过，Task9未来不套PASS。总1082轮502376生产100采集0正式，GOAL未启动；科学148+工程7项不变，controller2416/2417、GPU3049正常，10采集锁全部互斥退出。

43 Task8近10轮41.6975秒，余18轮12.51分钟纯；Task9未测，按旧同seed比例及本轮其他seed条件估整run纯98.83–116.16分钟，含准备保存20采集留115–145分钟。OBJECT600正式留60–90，OBJECT整体余175–235分钟/11月10日00:45–01:45；GOAL训练340–420+正式60–90，六run总余575–745分钟，约9.5–12.5小时/11月10日07:30–10:30。CPU验收解释发布另计，不含后续ALIGN。保持OBJECT正式全部完成才GOAL；本次没有调度修改/训练重启。详细来源公式见本快照TIMING_AUDIT与PROGRESS_ETA_REPORT。

## 2026-11-09T19:00:25.279185+08:00 恢复后速度/GPU与用户指定套件顺序

check_readonly_20261109_190024与audit_progress PASS，上一185519全部前缀保留；OBJECT41/42完整353/346轮、Task9 53/46，43及GOAL未开始；60采集0正式。GPU30样本均90.33%、76–96%，5994MiB，40–43°C、2670MHz；恢复后剔除重建首轮各5轮均62.1723/63.5158秒，较暂停前慢3.50%/4.30%，暂未见异常级减速，样本少不称长期稳定。详情新resume目录本快照TIMING_AUDIT/SPEED_ETA_REPORT/QUEUE_ORDER_AUDIT。

用户明确OBJECT训练后先闭环再下一个套件；实际冻结controller已严格OBJECT三train→600正式且全部exit0→GOAL，同序满足，无调度修改。当前两run规划55–80min、OBJECT43 220–270min、OBJECT正式60–90min，OBJECT全组余335–440min（11月10日00:35–02:20）；再GOAL训练340–420min及正式60–90min，六run总余735–950min，约12–16小时/11月10日07–11时，CPU独立验收解释发布另计，不含后续ALIGN。不创建定时监控、不按时钟推断完成。

## 2026-11-09T18:55:20.027568+08:00 用户扩容完成后已核验续训（覆盖旧暂停）

主机实际重启，原进程消失；数据可用44.83GiB/root1.32GiB。读取 execution/20261109_q1_user_resume/REPORT.md、USER_RESUME_STATUS、SOURCE_REVIEW、FINAL_WRAPPER_REVIEW。实际完整GPU恢复及原10/17门控块精确PASS，隔离540optimizer/0环境；原中断已记540+UNKNOWN单列。原347/340完整epoch和7149/6897门控行保留；唯一dispatch_resume_once_v2于18:53启动新controller2416/OBJECT2417/GPU2430/2428，18:55实际348/341轮、进度审计PASS。科学148项不变；新工程7项合同9c5b7e0b23e1beabb4bf28216f6a75f372d6d2f68b3f6515c27debdd8eda34bb。

严禁再调用任何dispatch/validate/build/旧controller；只用新resume目录check_readonly、audit_progress；OBJECT最终用其final_readonly→audit_final_local→compare_primary_s（STATIC_ONLY）。真实全局成功源engineering_reboot_resume_20261109_v1/controller.exit0，OBJECT为OBJECT_controller.exit0，恢复41/42为Q1_41_train.exit/Q1_42_train.exit；原全局和OBJECT出口保持不存在。GOAL原controller尚未启动，仍用其实际自然完成和原最终包装。旧暂停记录永久保留但已被本次明确继续授权覆盖。

后续ALIGN仍未启动，其既有launch_after_q1/deploy门槛目前引用旧全局出口，实际运行前须外置版本化适配此真实成功源并核当前两套件独立接受和发布，禁止造旧exit。空间已足，仍需部署前新鲜预检。

## 2026-11-09T18:43:51.840233+08:00 用户要求暂停扩容（覆盖此前继续安排）

当前两个调度器及两组 launcher/train 共六进程均 SIGSTOP/T，身份已核。证据 execution/20261109_q1_user_pause/LOCAL_AUDIT.json 和 REPORT.md。148 冻结项不变；当前实验 PARTIAL_PAUSED_BY_USER_IN_MEMORY。不得自动恢复或启动后续 ALIGN。仅用户明确继续后先核 boot、进程 start、hash、挂载容量和恢复边界；若重启，内存中部分轮丢失，不得盲恢复/重跑。未强制 checkpoint，暂停时间不计正常吞吐。

## 2026-11-09 18:37 临时清理后容量仍不足

只清理1个旧残片+34安装下载缓存，总95.79MiB，旧模型/结果/当前文件0删除，148冻结项不变。最新系统盘available1.32GiB、data6.74GiB（当前保存继续波动）；ALIGN仍需8GiB新鲜门槛且q1尚会增长。证据见同级20261109_align_space_cleanup/REPORT.md与SPACE_READINESS_20261109_1837.json。不能为启动放宽预算、删模型或占用保留块。准备合同及代码不变。

# MLR+IFA + ALIGN：部署准备完成，尚未启动

观察：2026-11-09T17:09:18.970265+08:00。文本合同：`6a6298283056986941016d1f487f8169fcbfe760679088e675f51ceaa015f7d7`。

六份代码由各自已接受Random源逐项匹配后适配；保留其模型、IFA、更新顺序、AdamW、逐更新scheduler及原评估器。唯一训练目标变化是ALIGN下一动作与有效窗口BC mask；没有RA/E/R/q/候选门控。Random人口、样本成员、顺序及每stage100轮保持。

实际CPU审计覆盖六case与24stage：特征SHA、原Random成员/顺序、原始HDF5下一动作、完整窗口边界mask、初始状态绑定、阶段人口全部通过。无效窗口留在原人口中，只屏蔽BC，IFA与sampler不删样本。仅复用各自完整current标签，不从RA选样缓存借成员。

代码准备包括：训练及5调用/stage隔离准入、独立端口最多两训练的有限队列、串行原评估器、断言失败停止新启动且不终止已启动训练、进度前缀审计、CPU最终验收及六指标/SF/FS/全部负单元比较器。正常与失败两种队列模拟实际通过，0训练0环境。全部GPU准入和最终CPU包装仍待真实执行，不预先记PASS。

## 启动顺序

当前q1 OBJECT和GOAL全部自然结束 → 两套件独立CPU接受/比较及FINAL_REPORT发布 → 新鲜容量和GPU检查 → 唯一部署 → OBJECT三训练/600正式 → GOAL三训练/600正式 → 本轮独立CPU接受/比较/解释发布。当前控制器未增加新队列或接收信号。

启动入口：`python launch_after_q1.py OBJECT_FINAL_DIR GOAL_FINAL_DIR`。入口核实际LOCAL_FINAL_AUDIT、PRIMARY_S_COMPARISON、已发布FINAL_REPORT与remote MLR SHA，再绑定真实旧合同/退出及无进程；缺任一条件在部署前停止。不要直接重复任何controller或dispatch。实际失败要先保留进程、退出、冻结hash和成本再审计，不自动重训。

## 容量与本地空间

六run历史同形状4CP+1恢复文件合计约3.87 GiB；考虑两训练原子保存临时副本、轻量日志与余量，以6 GiB新输出峰值加2 GiB保留作门槛。16:59实际data可用7.15 GiB，当前不足门槛约0.85 GiB，且q1仍会增长。不能把当前空闲重复当作新实验可用空间，启动时必须重新核验。

只保存/发布源码、标签身份/哈希、选样名单、配置和已有逐回合结果；未下载任何模型、latent、示范动作或恢复状态，未删除任何旧模型。新buffer只链接已核SHA的只读特征payload，避免复制GB级特征。原始只读返回中的大元数据及含动作示例的探查仅本地保留，不发布。

## 当前训练保护

16:59只读核实原q1合同和148冻结项全部一致，controller2638/2639、launcher2641/2642、GPU工作进程2650/2652存在。无暂停、重启、源码替换或旧结果覆盖。新root与private目录均不存在。

## 接受与解释边界

新实验实际更新预算544800、2400轮、0训练期采集、1200正式；隔离准入计划120另记。旧Random1200正式只读复用。当前仅准备接受，不是实验接受；旧硬件和历史global动作随机流不同，最终不将所有差异归为严格单因素因果。保持开发集重复使用与预训练来源未闭合限制；不按结果挑seed/版本或补评。

先读PROTOCOL、PREPARATION_STATUS、SOURCE_REVIEW、INPUT_AUDIT、CONFIGURATION_REVIEW、PREPARATION_VALIDATION、UPLOAD_REVIEW和FINAL_WRAPPER_REVIEW。当前只等条件，不运行任何新GPU诊断。
