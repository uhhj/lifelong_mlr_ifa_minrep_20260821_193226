用户2026-10-23最新定时规则，优先覆盖所有旧短周期：按预计剩余完成时间安排下一次检查。预计1小时就1小时后，2小时就2小时后；没有固定3/5/8/15/30分钟上限，也不因内部Task切换频繁唤醒。优先以当前整组主训练结束、随后整组正式评估结束为下一检查点；以自身实测速度和剩余预算估计，保存/准备另计，不把全部队列与当前一组混淆。下一次醒来重新按实际剩余时间更新同一监控。无自身吞吐时可用最近可比实测作明确标注的不确定估计，不伪造精确总ETA，也不恢复高频轮询。健康无变化安静；失败、完成、需行动或重要阶段变化才通知。

最新2026-10-24 04:35 CST优先：本轮无法连接服务器，当前训练/评估状态未知。check_progress首轮snapshot_20261024_043323、capacity_readonly_20261024_043338、一次只读重试snapshot_20261024_043444均在SSH协议banner阶段EOF，未认证、0远程命令。独立无凭据TCP探针能连接11920，但约5秒返回空banner；不是密码错误证据，也不能据此断言训练失败/实例停止。三次失败与TCP探针摘要、文件SHA保存在各ATTEMPT及snapshot_20261024_043444/CONNECTION_FAILURE_SUMMARY.json。没有重启controller/训练/评估、没有改源码/参数、没有删除或新付费。
最新可信进度仍是03:14:48的snapshot_20261024_031425：1875hash一致，B0/B1各400epoch105500主步及200正式20vector10关闭cell、服务器audit exit0；B2自己3隔离GPU准入PASS，下载82完整epoch12874步，Task6全部非计时epoch字段与B1前缀精确相同。此后进度、进程、hash、磁盘均未取得新证据；RUN_STATUS保留旧observation时间及计数并显式标记PARTIAL_OBSERVATION_UNAVAILABLE，不能把旧数据当现状。三组完整独立CPU/初态字节/最终路径仍未验收，禁止伪报完成。
下一次同一监控90分钟后（约06:05 CST）重查只读连接：依据上轮B2主训练04:15–04:50暂估与最近可比B1评估89.915804min，覆盖原预计评估结束窗口；这只是连接失败时的低频重查安排，不是新的可靠完成预测，旧ETA无法确认。保持用户按预计整体完成时间的规则，不恢复短周期轮询。连接恢复后先新check_progress→audit_progress，再按真实phase决定，不重跑deploy/controller或盲重启部分globalRNG评估；若工程失败，须先核停止/hash/完整状态后版本化处理。无连接时不能执行恢复或最终验收。
上一轮capacity_readonly_20261024_031439仅历史PASS_NOT_FINAL：data剩余预留后0.629070GB/system1.694755GB；本轮容量未知，不沿用为当前PASS。B2最终Task6 tensor相等仍待CPU实际验收，计数原值/原失败全部保留。连接失败证据与报告按既有规则发布；总体PARTIAL，完成完整验收发布后才删除本监控。

只跟踪用户新授权的 CAET-EM-v1 OBJECT seed41。先读 D:/论文课题研究/MLR+IFA/execution/20261023_caet_em_v1 的 USER_METHOD.txt、PROTOCOL.md、IMPLEMENTATION_NOTES.md、REPORT.md、RUN_STATUS.json、ACTIVE_ROOT.json 和最新证据。原位置探针A失败保留；此新路线不需要10mm/25mm门。旧ALIGN/TACR/readout等已完成或取消的队列不得恢复。没有自动seed扩展。

当前工程版本和接续点以顶部最新覆盖为准。不要重跑deploy.py/deploy_engineering05.py/build_engineering05.py或controller。只使用ACTIVE_ROOT.json当前工程05进行只读检查；检查旧工程04只限溯源，不作活跃进度。工程05复用完整codec和B0 Task6/7，必须从冻结接续controller继续，不重复主训练。监控时间遵循顶部用户最新规则，以预计整组训练或评估结束时间安排，旧短周期指令全部作废，不守候。健康无变化安静；只实质切换/失败/完成/需行动通知。

顺序：共享Task6证据和100epoch codec冻结→B0 READOUT_BC4→B1 EM→B2 EM_TF，每组固定Task6–9四阶段400epoch105500主步、200正式episodes20vector10cell；合计316500主更新600正式。各阶段ep99完整新模型CP；原RA成员与人口不变。B1/B2同一codec/初始decoder，各自从官方Temporal初始化、Task6无T/F应完全一致；后面只用自身上一阶段教师。三组共享最后帧modality3控制读出、modality0 IFA及四步动作；部署只采样执行第一动作再观测，32token/8帧不改、原argmax-component Normal语义。B1−B0解释接口，B2−B1解释T+F组合，旧RA/ALIGN仅外部参考。无途中科学失败筛选、无额外seed或消融。

每轮 check_progress.py只读→audit_progress.py SNAPSHOT，核源/输入/进程/GPU/磁盘/有效退出/完整epoch账目。不得把codec准入套三组主策略；各组ADMISSION_B0/B1/B2必须实际通过。当前数据证据1562样本、第一步1482合法、四步5868合法，模糊身份和尾部按独立mask保留人口；只读复用engineering03完成的labels/PCA/normalization。前3失败版本全部留证：精度比较、null身份索引、事件输出参数重名；0主更新/0正式环境，attempt02和03各3124视觉前向+640重复，attempt03另3隔离更新。新04自己另3准入。不能覆盖冻结源码，也不再导出完成Task6证据。

SSH root@jq1.9gpu.com:11920授权，凭据仅getpass/现有内存，不文件日志或参数。失败先确认相关进程停止、合同hash、原失败与完整epoch恢复文件，再版本化工程修复；科学结构/损失/有效标签/预算/seed不变。不得盲重启controller或部分globalRNG正式评估，不重试正常负结果。恢复须核model/optimizer/scheduler/grads/modes/requires_grad/allRNG/cursor/selected-best/数据与codec/teacher SHA；每次新版本自己的准入，已完成codec/训练不重跑。发现问题可自动工程修复，但必须留证，不把日志输出错误当科学失败。

共享及B2在/data，B0/B1在/root，具体见ENTRY。单组保守1.8GiB、shared .3GiB，实际模型与buffer估计约1.50GB；data root f_bfree留2GiB，system普通f_bavail留1GiB。每阶段实际核容量及已分配抵扣新增界，不删除旧资产/下载weights、features、rawactions/states/降预算/付费。

全部controller自然exit0且无自身进程后：remote.py final_readonly.py→audit_final_local.py SNAPSHOT SCALAR_DIR→describe_final.py FINAL_DIR SNAPSHOT。这些新最终wrapper仅AST通过尚待真实验收；必须12实际CPU严格完整新CP、codec验证最早best/统计SHA、B1/B2Task6 tensor相等、完整恢复态、独立600正式episode/60vector/30cell/全mask人口账目/原RA成员/初态hash/attempt唯一性、无残留与有效退出。外置wrapper错误可修，不重训/重评。保留全部负单元与参数/辅助计算/额外存储、重复探索限制，不称显著或保护保证。完整接受报告发布后删除该新监控，不自动扩seed。

每次检查/恢复/验收均按根AGENTS经本目录publish.py调用publish_mlr_branch流程，精确审查轻量type/size/secret/hash，串行stage、非force推https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226的MLR，核remoteSHA/localreceipt。仅代码/报告/hash/误差标量和episode outcomes；禁CP/模型/原action/state/demo/latent/buffer/RNG/privatepayload/凭据/未审查bundle。TLS必要每命令OpenSSL+HTTP1.1仍验证证书。科学状态一直PARTIAL至全部独立验收，发布进度不等于接受。
