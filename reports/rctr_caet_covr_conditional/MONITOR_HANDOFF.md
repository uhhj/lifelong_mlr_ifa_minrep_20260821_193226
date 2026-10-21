## 最新检查 2026-10-21 10:44 CST（PARTIAL）

CAET OBJECT42正常Task9，359完整epoch/90658主步/0正式。Task7/8已100epoch，各22695/22700和30897/30900步辅助激活；Task9已59epoch，21358/21358步激活。三个效果预测器均10epoch，共1240辅助优化步，实际训练正对1062/2796/2514，动作梯度有限非零PASS。这些不是闭环增益。

新增只读入库核验：Task6–8配额781/740/780，成员有序ID、源/目标载荷均与父RA一致；四阶段IFA名单及初始化RNG同父。204源码/输入/特征hash及已完成阶段退出0、本地CAET_PROGRESS_PASS，无实验恢复或重复预检。修正外置监控之前查找通用STORAGE_COMMIT文件的空结果，改为实际ra_coreset/storage/task*.json与retain完成标记，不改服务器源码。原首次读数另存CAET_DETAILS_first及receipt。

证据progress_20261021_104229。剩余主训练实测约11分钟，完整评估参考79分钟，入库及最终验收另计，预计12:15–12:45仅参考；30分钟检查不变。数据盘14.64GiB/系统5.51GiB，LIBERO50未GPU准入。使用check_progress_readonly.py --caet及audit_caet_progress.py，不重复RCTR完成审计。

## 最新检查 2026-10-21 10:17 CST（PARTIAL）

CAET OBJECT42 正常训练Task8，261完整epoch/57249主步/0正式。Task7已完成100epoch，22695/22700步实际效果辅助激活；Task8已完成61epoch，18847/18849步激活。Task8效果拟合10epoch/430辅助优化步、2796实际训练正对、动作梯度有限非零PASS。204源码及输入/特征hash、已完成阶段退出0、本地CAET_PROGRESS_PASS；无恢复或重复预检。主训练剩余约0.61小时，完整评估参考1.32小时，另计后续标签/拟合/入库/验收，整体预计12:20–13:00仅参考。

本轮使用新增固定只读 `check_progress_readonly.py --caet <snapshot目录>` 与 `audit_caet_progress.py`，仅核验活跃CAET，RCTR完成证据沿用progress_20261021_094302，不重复已完成RCTR的重哈希/复算。检查证据progress_20261021_101253。新增本地监控代码不改变服务器冻结源码。数据盘约14.79GiB、系统5.51GiB；LIBERO50数据已齐，仍未GPU准入。当前30分钟检查不变，正常运行不重启。

# 三方案与LIBERO-50执行交接（以本版为准）

最新2026-10-21 09:46 CST：progress_20261021_094302，RCTR_OBJECT42全部完成，400epoch105500step/4CP严格加载/200正式20vector/176冻结源码输入/入库IFA/无残留/全部exit0及本地独立验收PASS。精确AUC229/320=71.5625%，FWT70、NBT-3.0555556、OldFinal65、FinalAll63.75，与RA五指标全部相同，200条成功失败结果逐条相同。三采集阶段0认证、105500步辅助激活0，无RCTR监督干预，不能把本轮作为辅助损失有效或有害的证据。按原AUC门槛不扩RCTR41/43或GOAL，controller已自动切换CAET_OBJECT42。CAET真实4隔离更新恢复0环境预检PASS、204冻结源码/输入/特征PASS、Task6张量同父；122epoch20694步、正式0。Task7效果模型10epoch310辅助更新、3660全局正对、拟合实际1062正对，动作梯度有限非零；Task7主训练4994步中4993步辅助激活。剩余主训练按当前吞吐约1.11小时，评估参考1.32小时，另计后续标签/拟合/入库/最终验收，当前预计约12:15–13:00但非承诺。服务器RUN_STATUS队列字段可能停留RCTR旧值，因为controller阻塞等待CAET launcher；以DECISION、真实进程和CAET本身状态为准。使用check_progress_readonly.py --details新增completion/caet字段及audit_rctr_completion_caet_progress.py本地复算；旧audit_progress_stages仅RCTR历史分支，不作为CAET进度报告。没有重启任何实验或修改冻结科学源。LIBERO50仍未GPU准入；数据盘约14.93GiB，系统5.51GiB。

最新2026-10-21 09:15 CST：progress_20261021_091231，RCTR正式170/200条，17完成vector+1进行中，8完整cell，0技术异常；evaluate PID49522与10环境子进程正常。400epoch105500步、三阶段0认证及全部辅助激活0不变。176源码/冻结输入及逐分支、episode/attempt本地复算PASS。17批实测均值230.7秒，剩余评估约12分钟，最终验收另计；仍PARTIAL，不提前判全指标。只读details首次SSH读取超时，已保留失败与旧脚本，将该固定只读命令I/O超时由60改180秒后成功；未改服务器科学代码或重启实验。数据盘15.72GiB/系统5.51GiB；LIBERO50未GPU准入。继续30分钟检查，后续只按完整退出验收和原精确AUC门槛推进。

最新2026-10-21 08:44 CST：progress_20261021_084241，RCTR正式100/200条，10个vector完成，0技术异常；400epoch105500步及三阶段0认证/全部辅助激活0不变。176源码与冻结输入、逐episode及attempt独立复算PASS，evaluate PID49522仍在运行，当前环境进程正在轮换，未重启。按10个已完成vector均值216.8秒估计剩余评估约36分钟，最终验收另计。首批完整cell暴露本地只读审计误读task键，原失败和旧脚本已保留；现从cell文件名解析并交叉验证日志身份、模型/初态哈希，重新复算PASS。仅修本地审计，无服务器科学源/预算/模型变化。仍PARTIAL，不提前计算全指标或判断扩展。数据盘15.72GiB、系统5.51GiB；LIBERO50仍未GPU准入。

最新2026-10-21 08:14 CST：progress_20261021_081224，400epoch/105500主步全部完成，evaluate PID49522及10环境子进程正常。Task6/7/8采集460/456/472分支（合计1368探索20确认），认证记录均0，三个ACCEPTANCE PASS；Task8有4个快照进入确认，均未过。四任务训练辅助激活均0，不能将后续分数差解释为RCTR监督收益。没有修改既定AUC扩展门槛。正式10/200条、1完成vector+1进行中、0技术异常；原0–19/seed与记录唯一性复算PASS，176源码输入hash PASS，7个train/collect退出均0。首完成vector约283.8秒，剩余评估暂估1.50小时，未计最终验收。用新版check_progress_readonly.py --details读取formal/worker_processes，audit_progress_stages.py已适配evaluate阶段和记录/attempt复算；服务器mon进程筛选不含coreset_exp.evaluate，不能因其processes列表未列评估而误报退出。外置本地脚本变化不影响科学冻结源。仍30分钟检查，LIBERO50尚未GPU准入。

最新2026-10-21 07:41 CST：progress_20261021_074031，仍正常collect8，420已提交辅助分支（404探索16确认）、51221策略步；未完成采集验收，不能把412次成功当作认证记录。300epoch/69300主步、正式0条不变。176源码/冻结输入、逐分支账目和本地复算PASS，无重复start；无需恢复。Task8已观测均值10.54秒/分支，剩余最大计划量若同速约15分钟，之后仍需Task9训练和正式评估。为观察交接将原自动化改30分钟，不新建。LIBERO50保持数据就绪未GPU准入，GPU正常，数据盘15.85GiB/系统5.51GiB。每次采集/评估完成按原合同验收，门槛不变。

最新2026-10-21 06:38 CST：progress_20261021_063655，collect8 PID48670正常，累计300epoch/69300主步、正式0条。Task7完成456探索、0确认、27758策略步；456分支全成功，未出现配对挽救，0认证记录，collect7.exit0。Task8训练100epoch/30900步、辅助激活0，train8.exit0。Task6/7认证及逐分支attempt独立复算PASS，无重复start；176源码/冻结输入PASS。Task8观察87分支（83探索4确认），8110策略步，尚未完成认证验收。初期7.99秒/分支，当前剩余探索确认若同速约0.93小时，非硬上界且不代表整队列完成；保留每小时检查。数据盘约15.88GiB/系统盘5.51GiB，LIBERO50未GPU准入。仍使用check_progress_readonly.py及audit_progress_stages.py，无实验重启或科学修改。不要把Task7辅助全成功或Task8中间成功当正式结果/认证记录。

最新检查2026-10-21 05:32 CST：progress_20261021_053022，已进入collect7（PID48346）。Task6采集460分支/215043策略步通过验收，456探索中仅1个候选进入独立确认，4确认分支未通过，最终0认证训练记录；53个成功分支不能当作训练记录。Task7的100epoch/22700步已完成，辅助激活0次，累计200epoch/38400步。train6/collect6/train7均exit0，176源码及冻结输入PASS。collect7观察时16分支，正式0条；初期约8.24秒/分支，但均早期成功，不能据此承诺全程ETA。当前阶段若同速剩余约1.12小时，暂将同一自动化改每小时。后续使用新版check_progress_readonly.py --details与audit_progress_stages.py，读取各阶段、逐分支认证及训练辅助激活统计；旧audit_progress_readonly.py仅用于历史collect6证据。改动仅本地外置只读脚本，服务器科学源/合同/预算未变，无重启。LIBERO50仍未GPU准入。

最新检查2026-10-21 05:03 CST：progress_20261021_050025，只读复算PASS；collect6已提交442分支（438探索+4确认）、206403策略步、正式0条。176源码及冻结输入PASS，进程正常，无恢复或重启。近期41.46秒/分支，剩余探索上限约0.21小时、确认最多另0.51小时；保留30分钟检查间隔。53个成功辅助分支不代表认证训练记录。后续若阶段转换，先适配外置只读进度脚本，不修改冻结科学源码。

用户要求持续推进、自动获取缺失数据、自主选推荐工程方案，不只轮询。当前PARTIAL，不可把部署或CPU测试称为实验完成。先读BRANCH_RULES.md、proposals三份全文、RUN_STATUS.json、REPORT.md及progress_20261021_server_queue/AUDIT.json。

最新检查2026-10-21 04:29 CST：progress_20261021_042827，collect6已提交393分支（389探索+4确认）、181017策略步、正式0条；176源码与冻结输入PASS，无工程恢复。52条辅助成功不能当作52条认证训练记录。近期33.14秒/分支，Task6剩余探索约0.62小时、确认最多另0.41小时；为检查即将到来的阶段交接，同一自动化改为30分钟。阶段交接后应恢复较低检查频率。外置check_progress_readonly.py --details与audit_progress_readonly.py目前写死task6/100epoch，阶段变化时只改外置读数逻辑，勿把正常进入train7误报为协议失败；原冻结服务器科学源不变。

最新检查2026-10-21 02:31 CST：见progress_20261021_022912。RCTR仍collect6，178个已提交辅助分支、正式0条，训练100epoch/15700步不变；176源码及全部冻结输入核验PASS。当前实测约44.05秒/分支，Task6剩余探索上限约3.40小时，确认最多另0.59小时；非整队列ETA。没有服务器工程恢复或重复启动。旧通用SSH桥本地调用被审批策略拒绝，固定只读SSH客户端check_progress_readonly.py已正常连接（主机密钥已固定，getpass内存凭据）。先调用该脚本保存新快照，再用--details指向该新目录核查冻结hash及committed；audit_progress_readonly.py复算本轮。不要为绕过工具审批复用通用桥。

## 科学顺序

旧RCTR A2已按用户要求停止，不恢复。RCTR→CAET→COVR：各OBJECT42精确AUC严格高于同seed RA且工程PASS，才固定补OBJECT41/43；三个OBJECT等权平均AUC严格高于对应RA平均，才执行该方法GOAL41/42/43，随后下一方法。没有OldFinal或2/3附加门槛。最后RA+IFA LIBERO50 41/42/43。旧基线全部只读。

## 服务器与队列

SSH root@jq1.9gpu.com:11920已有授权，凭据只内存/getpass，不存文件日志。使用check_progress_readonly.py的固定只读命令与已固定主机密钥；旧通用桥受审批限制，不复用该桥绕过限制。
Q=/data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020。
当前controller_v2.py PID47521，查controller_v2.pid.json/exit/log、CONTROLLER_V2_HANDOFF.json。原controller PID46681在无子launcher时显式SIGTERM交接，原日志/pid/源保留，不制造原exit0。controller_v2使用原branch_gate与独立queue_verifier，LIBERO50用libero50_verifier。registry18项：9OBJECT+6RCTR/COVR GOAL+3LIBERO50；每个未来worker仍须独立真实GPU准入。
使用Q/monitor_snapshot_v2.py，原monitor_snapshot.py遗漏新controller和下载，不宜独用。实际记录时间取JSON unix，不取快照文件名。
14个未启动OBJECT/GOAL入口采用worker_entrypoint_env_v2.py，解决CPU导入时空CUDA导致EGL非法编号；科学源和bundle未改，CPU_NATIVE_IMPORT_retry.log证明CUDA张量占用0。旧entry/launcher保留，新registry版本明确记账。不得改已冻结共享文件来绕过哈希。

## 原RCTR工作区（已完成；保留启动历史，当前CAET见顶部）

W=/data/mlr_ifa/experiment_workspaces/RCTR_OBJECT_seed42_full_20261021；O=/data/mlr_ifa/experiments/RCTR_OBJECT_seed42_full_20261021。source W/env.sh。supervisor46513、pipeline46522、collect6 46593。Task6 100epoch/15700步结束且实际张量/初始化RNG同父，正在collect6。source176文件冻结。预检第二次PASS28工程步0optimizer，第一次缺assets链接失败发生第一个环境步前，保留。
顺序train6/collect6/train7/collect7/train8/collect8/train9/evaluate/audit/final_process；400epoch105500主步、4CP、200正式20vector；采集最多1368探索+144确认，prefix和工程尝试独立记账。committed JSON为唯一完成权威，不重试正常失败。

## LIBERO-50

最新2026-10-21 01:58 CST：50/50已全部无损获取，libero50_acquisition.exit=0；额外完整SHA/示范/BDDL/初态审计LIBERO50_FULL_ASSET_AUDIT.exit=0，READINESS_20261021.json为DATA_AND_CPU_READINESS_PASS_NOT_GPU_ADMITTED。共享预训练实际1,941,000步；每seed下游859,400步，5阶段每epoch880/1071/1684/2057/2902个batch。合计4,519,200步，不含隔离预检。不要再重复下载或完整数据哈希审计，仅在新准入合同要求时核验。GPU耗时尚未实测，不能沿用OBJECT的小时级估计。RCTR当前134采集分支/0正式记录，继续正常运行。

服务器acquire_libero50.sh PID46818负责自动获取，查LIBERO50_SERVER_STATUS.json、CURRENT、libero50_acquisition.exit及libero50_acquisition/*.json。固定HF revision f13aa24a3da8c43c7225569f28c562979fa0e35a。数据位于/data/mlr_ifa/datasets/libero50_acquired及/root/mlr_ifa_datasets/libero50_acquired，逻辑路径用链接统一。只对自己的临时下载做逐数组/属性无损验证后删除raw，原实验数据不能删。本地重复下载25160已停且文件保留。
LIBERO50_PARTITION.json为LOTUS固定25预训练+25新增任务，全部BDDL/初态已找到。论文公开代码未提供完整50预训练权重；事前工程选择见libero50_preparation/IMPLEMENTATION_DECISIONS.md。共享新预训练seed42、25任务50演示、100epoch、batch10、FP32、CLIP base冻结而LoRA8/alpha8及新投影/FiLM/state/T/head训练，最终ep99，不能冒充官方checkpoint。三个下游seed使用同一份。当前未运行预训练。
LIBERO50_SOURCE_BUNDLE_final.json SHA370dd4645bb319f0f1739fa13b5a84fa82d86c7c0e0078be0da28d8caad718ae，196文件。14CPU性质PASS及真实数据CPU完整→Brief GMM history1/4/8完全一致，0更新/0环境/0CUDA字节；不是GPU验收。新正式入口自动prepare→4隔离GPU恢复更新（下游另1零动作接口步）→pipeline。预训练另4隔离更新0环境。未通过不能训练。
下游每seed5阶段每阶段5新任务，100epoch/batch10、BC+.1meanIFA/gamma.1/warmup500逐步、RA每任务ceil.5及半配额硬锚点、原距离标定42与sampler0。IFA多当前任务规则已事前披露；新特征逐seed导出；新buffer硬链接自身不可变feature以省空间，载荷逐字节hash。每次RA选择额外独立全部删除代价验证。5CP、1500episodes/150vectors/seed，共4500；原0–19、10lane、env[s,s+1]、600步5zero、stage global RNG、argmax分量Normal。不给额外Random/Coreset基线，不扩seed。
workspace Q/libero50/shared_pretraining与seed41/42/43；新输出/data/mlr_ifa/experiments/LIBERO50_reproduced_pretraining_seed42_20261021及RA_Coreset_LIBERO50_seed{s}_20261021。
外置package_libero50_light.py仅包装允许元数据，完成后在PYTHONPATH对应source和LIBERO50环境下调用；audit_libero50_light.py本地纯标准库复算全部4500/矩阵/三seed均值样本SD。未完成不能报指标。检查磁盘并保留原资产，不能为了空间改科学预算或精度。

## 适用性边界与恢复

CAET GOAL真实probe在三个seed阶段7正对均为0，stage8/9有正对。未登记CAET GOAL，不能放宽阈值或伪造配对；该条件若触发必须如实处理，其他独立工作照常推进。不是说整套所有科学条件已通过。
工程错误先确认进程停、合同输入一致、保留原失败；训练只从完整epoch恢复model/optimizer/scheduler/grads/modes/allRNG/cursor，入库预提交名单幂等。collection技术重采必须同固定prefix物理/观察哈希并记全attempt。部分global RNG正式评估不能盲重启。冻结科学源/合同不可悄改重贴；新版本/外置适配说明原因。只读汇总可外置修复，不重跑环境。

## 发布与结束

每次检查/实验/只读审计按AGENTS运行publish.py，精确提交到publish_mlr_branch远端MLR，核对SHA并存receipt。排除所有BUNDLE JSON、asset_sources、权重、demos、latent/buffer、原actions/状态/噪声/RNG二进制。本地report只含PARTIAL观察，不代表完成。自动化rctr-caet-covr-libero50保持有效；全部适用分支及LIBERO50完整验收交付后才删除，不启动新付费资源。
