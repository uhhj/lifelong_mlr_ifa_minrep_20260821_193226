## 2026-11-10T16:03:42.756526+08:00 恢复后正式评估推进

# ALIGN恢复后进度与实测评估速度

观察2026-11-10T16:03:42.756526+08:00，PARTIAL。新工程进度审计PASS，152934全部训练/准入/正式前缀精确保留；原科学和工程合同verify通过，三个OBJECT模型及恢复文件SHA不变。controller40721/OBJECT41评估40723与环境子进程正常，无新错误、重启或重复训练。

|case|训练/400|正式/200|状态|
|---|---:|---:|---|
|OBJECT41|400|110|Task8模型的Task8第二批进行中|
|OBJECT42|400|0|等待串行评估|
|OBJECT43|400|0|等待串行评估|
|GOAL41/42/43|各0|各0|等待OBJECT600正式全部完成|

OBJECT三run的Task6–9各自ALIGN准入均已通过。累计1200完整轮、316500生产、60自身隔离、0训练采集、110/1200正式；原失败attempt和exit1保持。首个恢复单元四类RNG精确PASS，环境锁目前由OBJECT41独占，没有新增工程失败。

11个实际完成vector（每批10回合）均200.3391秒，近5批169.7299秒；600步批次约223–231秒，两个提前结束批次分别217/210步、82.26/84.66秒。不同任务终止长度直接影响墙钟，不能把所有批次套同一速度。

OBJECT剩49批线性参考138.61–163.61分钟，加初始化、后任务/seed波动规划160–220分钟，即约18:45–19:45。GOAL三训练两波仍为未测规划100–130分钟，其600正式按当前实现预留190–250分钟。全队列训练+正式余450–600分钟，约7.5–10小时，11月10日23:35至11月11日02:05附近；CPU最终验收、比较发布另计。这覆盖早先60–90分钟/套件及当天18–20时完成的旧估计，不按时钟断言完成。

本轮没有改训练/推理/评估代码，没有重训或重评。后续继续用engineering_assets_resume_20261110_v2的check/audit及最终包装，真实成功源仍仅新工程controller.exit0且无自身进程；旧退出1和失败证据保留。


## 2026-11-10T15:29:38.072208+08:00 故障已修复，实际评估恢复

# ALIGN资源故障已修复并实际恢复评估

实际观察2026-11-10T15:29:38.072208+08:00，PARTIAL_RECOVERED_ACTUAL_FORMAL_ROLLOUT_CONFIRMED。OBJECT41已完成10/200正式回合，首个完整vector记录已验真；controller40721及评估进程正常。OBJECT三seed原400轮/12阶段ALIGN准入和所有模型/恢复文件SHA保留，0已完成轮重跑、0新增optimizer诊断。GOAL未开始，仍在OBJECT600正式全部完成后启动。

原故障为新source目录缺少libero/libero/assets。读取服务器既有/data/mlr_ifa/src/libero/libero/assets的585文件（422313213bytes）并冻结完整哈希，仅建立六个链接，没有复制大资产、下载模型或改评估源码。原10子进程在场景XML打开阶段失败，原日志/attempt/exit1/RNG文件永久保留。首个恢复包装缺少多进程main保护，spawn子进程在环境构造前断言退出，原失败也保留；v2增加main保护和run_module alter_sys=True，维持原评估模块的多进程启动语义。

原首单元Python/NumPy/CPU/CUDA随机状态经两次恢复均逐项精确通过，最后guard在正式环境调用前核验；原模型、动作采样和评估函数不变。两个失败attempt独立保留，不计为已完成正式回合或训练收益；不抹去历史环境构造尝试。原总controller.exit1、OBJECT41 evaluate.exit1、恢复v1 exit1均保持，不伪造成功。

唯一活动工程engineering_assets_resume_20261110_v2，合同404ffb7e4a04f4d52091401e77df82e20086cfd2c7001f88c7c449642fd75e12。禁止重复deploy/controller或恢复旧调度器。后续只用该工程transport.py check_readonly.py → audit_progress.py NEW PREVIOUS。最终仅该工程controller.exit0且无自身进程后，用该工程final_readonly.py→audit_final_local.py→基础compare_baseline.py；最终包装已静态适配真实完成源，仍待实际CPU执行。最初冻结科学合同ce307dca2bce082e36803d266067462cb35a95f421e371f74f7d20d42a5dda8f保持。

没有改变OBJECT→GOAL的顺序，没有启动后台聊天定时监控。有限远端队列会继续OBJECT600正式→GOAL三seed训练（最多两并行）→GOAL600正式；出错保留现场并停止新增运行。当前只确认修复和实际评估进展，不从首个vector外推全套件稳定吞吐。


## 2026-11-10T15:11:31.619770+08:00 评估失败覆盖旧健康状态及ETA

# ALIGN进度：OBJECT训练完成，评估工程故障

实际观察2026-11-10T15:11:31.619770+08:00。状态PARTIAL_STOPPED_EVALUATION_ASSET_FAILURE，不能将LOCAL_AUDIT中的预算/前缀PASS_PROGRESS_NOT_FINAL当成运行健康或最终接受。

OBJECT41/42/43均400完整轮、105500生产更新、四阶段各5次隔离ALIGN准入通过，自己的train.exit均0。43于2026-11-10T14:25:47.379322+08:00自然退出。累计1200/2400轮、316500/544800生产、60/120隔离、0训练采集、0/1200完成正式。GOAL三个seed未启动。135548所有已记录epoch/准入前缀保留，冻结合同verify通过。

调度器随即启动OBJECT41评估，于2026-11-10T14:26:04.991228+08:00失败并退出1。实际日志FileNotFoundError指向新部署source/libero/libero/assets/scenes/libero_floor_base_style.xml；环境子进程加载场景失败，主进程记录ConnectionResetError。错误属于新目录资产部署遗漏，先前只导入预检未覆盖真实资产加载，不是已定位的GPU故障。当前nvidia-smi正常，4080SUPER利用率0%、显存10MiB，无本轮进程。

原attempt有environment_construct_start/complete、vector_attempt_start（Task6/task6、初态0–9）和technical_exception，0完成回合。异步构造complete日志不代表所有子进程成功；不能将该attempt抹去或声称环境成本为零。原controller.exit1/evaluate.exit1、日志SHA、失败记录、已完成训练和模型全部保留。本轮检查只读，没有重启、重训、重复评估或修改资产。

此前18:10–19:50估时已失效，当前队列已经停止，不能继续按原速度倒计时。恢复前需要版本化核验既有资产来源/路径、失败attempt与RNG边界，保留原失败及实际成本，再从未完成评估接续；OBJECT600正式完成后才允许GOAL，不能跳过该顺序或重新运行原controller。最终完成源也须按真实恢复工程适配，禁止伪造旧exit0。


## 2026-11-10T13:55:49.031630+08:00 最新进度

# ALIGN最新进度与估时

观察2026-11-10T13:55:49.031630+08:00，PARTIAL。实际check与audit_progress PASS，124649全部完整epoch/诊断前缀保持，冻结合同verify通过。controller39717和OBJECT43训练39906健康，无新失败或重启。

|case|完整轮/400|当前阶段|Task8准入|Task9准入|正式/200|
|---|---:|---|---|---|---:|
|OBJECT41|400|train.exit0|PASS|PASS|0|
|OBJECT42|400|train.exit0|PASS|PASS|0|
|OBJECT43|219|Task8 19/100|PASS|尚未进入|0|
|GOAL41/42/43|各0|未开始|未执行|未执行|各0|

41/42自己的四阶段五调用ALIGN准入全部通过；43初始化tensor/RNG及Task6/7/8准入通过，每次均核禁用原更新等价、有效mask、空mask可微零、保存恢复下一步精确及恢复初态，不能套未来PASS。合计1019/2400轮、255271/544800生产更新、55/120隔离、0训练采集、0/1200正式。43真实13:32:13启动；41/42完整训练dispatch至exit分别60.16/59.99分钟，含准备保存准入。

41/42 Task8均9.2932/9.3721秒、Task9均10.6758/10.7058秒；43当前单训练Task8近10轮8.0465秒，余81轮约10.86分钟纯计算。Task9尚未实测，参考同套件已完成run及43当前速度缩放，43整run剩纯epoch约26.23–28.68分钟，计保存准备准入规划35–45分钟。

OBJECT600正式尚未开始，暂留60–90分钟；OBJECT整套训练+正式余95–135分钟，即今日15:30–16:10附近。GOAL三训练按两个波次留100–130分钟、600正式60–90分钟。全部六run训练及1200正式合计余255–355分钟，即约4小时15分–6小时，今日18:10–19:50附近。最终独立CPU验收、结果比较和发布另计。GOAL和正式尚未实测，不能将规划当完成保证；详细公式和来源见本快照TIMING_AUDIT.json。

保持OBJECT三训练→600正式→GOAL三训练→600正式，最多两个训练；本次没有改科学代码、调度、训练或评估。已有S/q1及旧失败证据保持。后续仅v4/transport.py check_readonly.py检查，禁止重复launch/controller/完成轮。


## 2026-11-10T12:46:50.419889+08:00 最新进度


观察2026-11-10T12:46:50.419889+08:00，PARTIAL。实际check与独立进度审计通过，上一123348全部完整epoch/诊断前缀保留，新合同与冻结文件verify通过。controller39717与两个GPU训练进程正常，无新失败或重启。

|case|完整轮/400|当前阶段|生产更新|正式/200|
|---|---:|---|---:|---:|
|OBJECT41|143|Task7 43/100|25461|0|
|OBJECT42|144|Task7 44/100|25688|0|
|OBJECT43|0|未开始|0|0|
|GOAL41/42/43|各0|未开始|各0|各0|

合计287/2400轮、51149/544800生产、20/120自身隔离、0训练采集、0/1200正式。OBJECT41/42各自Task6/7准入的禁用ALIGN等价、有效mask、空mask可微零、完整状态保存恢复下一步精确和恢复初态均通过；未来阶段不得套PASS。

Task7近10轮41/42分别6.8359/6.7997秒，当前Task7余约6.49/6.35分钟。按Task7每batch耗时缩放，未测Task8/9约9.3/10.9秒每轮，当前两run剩纯epoch约40分钟、完整OBJECT单run约52分钟、GOAL单run约38分钟；这些不是后阶段实测。

规划当前两run45–65分钟，OBJECT43另55–80分钟，OBJECT600正式60–90分钟；OBJECT整套余160–235分钟（约15:27–16:42）。随后GOAL三训练两波100–150分钟和600正式60–90分钟。六run训练+1200正式合计320–475分钟，用户口径约5.5–8小时（约今日18:10–20:45）。完整CPU验收、比较和发布另计。Task8/9、GOAL和正式吞吐未实测，不作为完成保证；本次未改科学代码、调度、训练或评估。

详细公式和来源SHA见本快照TIMING_AUDIT.json。全部OBJECT闭环完成后才进入GOAL；已有S、q1和旧失败证据只读保留。


## 2026-11-10T12:33:48.710397+08:00 ALIGN六run队列已唯一启动

q1两套件已实际独立接受、完整比较及FINAL_REPORT发布，启动门槛已核远端提交cc648e8629ea7476f59db30be240325edb288f69。唯一engineering_startup_20261110_v4/launch_after_q1.py部署成功，controller39717，合同ce307dca2bce082e36803d266067462cb35a95f421e371f74f7d20d42a5dda8f。修复旧q1完成源、LIBERO路径环境及算法类重复注册；训练损失/更新方法、选样输入和评估器保持。首次v1导入失败零更新/零环境的原证据永久保留。禁止重复任何launch/deploy/controller或完成轮。

实际快照check_readonly_20261110_123348进度审计及STARTUP_AUDIT通过。OBJECT41/42各自初始化tensor/RNG与Random父源精确，Task6自己的5次隔离ALIGN准入通过；已完成轮分别16/15，累计4867生产更新、10隔离、0训练采集、0正式；未来Task7–9准入不得套PASS。其余四run未开始。既有q1与所有S只读。

此后只用engineering_startup_20261110_v4/transport.py check_readonly.py获取实际新合同快照，再用基础目录audit_progress.py NEW PREVIOUS。最终用该工程目录final_readonly.py（已适配新合同）和基础目录audit_final_local.py/compare_baseline.py；仍须真实controller.exit0且无自身进程。没有定时监控或自动聊天验收服务，远端有限controller会按OBJECT三训练/600正式→GOAL三训练/600正式自然运行；失败停止新增运行并保留既有结果。

## 2026-11-10 两套件最终接受，继续已授权ALIGN

OBJECT final_readonly_20261110_121750和GOAL final_goal_readonly_20261110_121810均实际远程CPU及本地PASS_COMPLETE_LOCAL_ACCEPTANCE。24CP/6完整恢复/24teacher缓存/1200初态noise/47400门控通过，验收前后科学轻量SHA完全相同，新增0optimizer/0环境。见execution/20261109_q1_user_resume/ACCEPTANCE.json、FINAL_REPORT.md、FULL_COMPARISON.json、FULL_METRICS.csv。六run全部独立接受，不重复任何q1训练、正式或最终CPU请求。

本轮完整S/q1：OBJECT AUC86.4931/81.7361、FinalAll85/74.1667；GOAL AUC87.2569/82.9167、FinalAll85/82.0833。前次OBJECT q1最终87.9167高于旧S85，方向反转；两轮均保留，不挑有利结果。GOAL seed41有效保持权重均1，存在执行/缓存差异限制；不是校准净因果。

用户明确验收报告发布后立即启动已准备MLR+IFA+ALIGN六run。新外置engineering_completion_gate_20261110_v2已仅适配真实q1工程出口，原准备合同保持，改runtime/CONTRACT，其余训练评估输入944文本中的科学部分逐项不变；新合同db88ef7991d4cfd6e77b6897977779ed5d37a85682cd63c14441bb63bddcd7c3。新鲜容量data可用44848771072B、GPU空闲。用该目录launch_after_q1.py带上述两个final路径唯一部署，必须先核两个接受和FINAL_REPORT已发布。不能运行旧launch或伪造旧exit。当前尚未部署；完成发布后接续。OBJECT三训练/600正式后GOAL同序，最多两个训练并行。

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
