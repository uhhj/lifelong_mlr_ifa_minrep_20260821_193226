用户2026-10-23最新定时规则，优先覆盖所有旧短周期：按预计剩余完成时间安排下一次检查。预计1小时就1小时后，2小时就2小时后；没有固定3/5/8/15/30分钟上限，也不因内部Task切换频繁唤醒。优先以当前整组主训练结束、随后整组正式评估结束为下一检查点；以自身实测速度和剩余预算估计，保存/准备另计，不把全部队列与当前一组混淆。下一次醒来重新按实际剩余时间更新同一监控。无自身吞吐时可用最近可比实测作明确标注的不确定估计，不伪造精确总ETA，也不恢复高频轮询。健康无变化安静；失败、完成、需行动或重要阶段变化才通知。

最新2026-10-24 00:46 CST：snapshot_20261024_004257/LOCAL_AUDIT、EVALUATION_PROGRESS_AUDIT与TRANSITION_AND_ADMISSION_AUDIT实际PASS_PROGRESS_NOT_FINAL。观察00:43:21，1875冻结hash一致；B0四阶段400epoch105500更新与200正式/20vector/10关闭cell完整，evaluate/audit均自然exit0，服务器4CP各104键审计与训练CP及METRICS SHA绑定。本地逐episode/attempt唯一性、成功向量、初态记录hash/seed/步数、全部诊断身份mask绑定通过；完整三组CPU/初态字节/最终路径仍待实际验收，不把服务器PASS当整体本地最终PASS、不提前下三组收益结论。B0正式vector合计75.932371min，部分环境提前完成导致首批外推偏保守，原估计保留。
队列已自然进入B1_train7，controller124820/GPU129237正常。B1自身3隔离GPU优化器调用、全状态下一步恢复/三最新模态依赖/1–8长度/未来隔离/效果路径/空mask/GMM语义实际PASS，22826763参数；Task6 train/diagnose exit0。下载B1完整177epoch33179主步（Task6 100/15700，Task7 77/17479），metadata较快照前进已记录。Task6/7人口身份顺序payload mask与B0相同，逐epoch sampler顺序及optimizer白名单精确复核，T/F/额外保护解码恒0。B2未自身准入，不套B1报告；没有重启改参。
B1 Task7最近10轮8.876714秒，剩72321主步以自身Task7每batch外推47.134569min纯训练，加保存/诊断/准备暂参考01:35–01:45 CST；Task8/9自身吞吐未测。按用户规则下一55分钟后检查B1整组主训练完成附近，不按内部阶段短周期唤醒。B1正式评估与B2训练未测，总队列ETA未知。
data root4.914348GB/system普通5.127582GB，全额再预留2.1/3.6GiB与2/1GiB后余0.512007/0.188369GB仍正；此界保守重复预留已分配，后续不足须先按稳定私有已分配抵扣新增界，不能误报容量失败。无删除/下载payload/降预算/付费。codec冻结不变，最终外置成本校正仍保留原diag计数。整体PARTIAL，一轮保存发布后结束。

只跟踪用户新授权的 CAET-EM-v1 OBJECT seed41。先读 D:/论文课题研究/MLR+IFA/execution/20261023_caet_em_v1 的 USER_METHOD.txt、PROTOCOL.md、IMPLEMENTATION_NOTES.md、REPORT.md、RUN_STATUS.json、ACTIVE_ROOT.json 和最新证据。原位置探针A失败保留；此新路线不需要10mm/25mm门。旧ALIGN/TACR/readout等已完成或取消的队列不得恢复。没有自动seed扩展。

当前工程版本和接续点以顶部最新覆盖为准。不要重跑deploy.py/deploy_engineering05.py/build_engineering05.py或controller。只使用ACTIVE_ROOT.json当前工程05进行只读检查；检查旧工程04只限溯源，不作活跃进度。工程05复用完整codec和B0 Task6/7，必须从冻结接续controller继续，不重复主训练。监控时间遵循顶部用户最新规则，以预计整组训练或评估结束时间安排，旧短周期指令全部作废，不守候。健康无变化安静；只实质切换/失败/完成/需行动通知。

顺序：共享Task6证据和100epoch codec冻结→B0 READOUT_BC4→B1 EM→B2 EM_TF，每组固定Task6–9四阶段400epoch105500主步、200正式episodes20vector10cell；合计316500主更新600正式。各阶段ep99完整新模型CP；原RA成员与人口不变。B1/B2同一codec/初始decoder，各自从官方Temporal初始化、Task6无T/F应完全一致；后面只用自身上一阶段教师。三组共享最后帧modality3控制读出、modality0 IFA及四步动作；部署只采样执行第一动作再观测，32token/8帧不改、原argmax-component Normal语义。B1−B0解释接口，B2−B1解释T+F组合，旧RA/ALIGN仅外部参考。无途中科学失败筛选、无额外seed或消融。

每轮 check_progress.py只读→audit_progress.py SNAPSHOT，核源/输入/进程/GPU/磁盘/有效退出/完整epoch账目。不得把codec准入套三组主策略；各组ADMISSION_B0/B1/B2必须实际通过。当前数据证据1562样本、第一步1482合法、四步5868合法，模糊身份和尾部按独立mask保留人口；只读复用engineering03完成的labels/PCA/normalization。前3失败版本全部留证：精度比较、null身份索引、事件输出参数重名；0主更新/0正式环境，attempt02和03各3124视觉前向+640重复，attempt03另3隔离更新。新04自己另3准入。不能覆盖冻结源码，也不再导出完成Task6证据。

SSH root@jq1.9gpu.com:11920授权，凭据仅getpass/现有内存，不文件日志或参数。失败先确认相关进程停止、合同hash、原失败与完整epoch恢复文件，再版本化工程修复；科学结构/损失/有效标签/预算/seed不变。不得盲重启controller或部分globalRNG正式评估，不重试正常负结果。恢复须核model/optimizer/scheduler/grads/modes/requires_grad/allRNG/cursor/selected-best/数据与codec/teacher SHA；每次新版本自己的准入，已完成codec/训练不重跑。发现问题可自动工程修复，但必须留证，不把日志输出错误当科学失败。

共享及B2在/data，B0/B1在/root，具体见ENTRY。单组保守1.8GiB、shared .3GiB，实际模型与buffer估计约1.50GB；data root f_bfree留2GiB，system普通f_bavail留1GiB。每阶段实际核容量及已分配抵扣新增界，不删除旧资产/下载weights、features、rawactions/states/降预算/付费。

全部controller自然exit0且无自身进程后：remote.py final_readonly.py→audit_final_local.py SNAPSHOT SCALAR_DIR→describe_final.py FINAL_DIR SNAPSHOT。这些新最终wrapper仅AST通过尚待真实验收；必须12实际CPU严格完整新CP、codec验证最早best/统计SHA、B1/B2Task6 tensor相等、完整恢复态、独立600正式episode/60vector/30cell/全mask人口账目/原RA成员/初态hash/attempt唯一性、无残留与有效退出。外置wrapper错误可修，不重训/重评。保留全部负单元与参数/辅助计算/额外存储、重复探索限制，不称显著或保护保证。完整接受报告发布后删除该新监控，不自动扩seed。

每次检查/恢复/验收均按根AGENTS经本目录publish.py调用publish_mlr_branch流程，精确审查轻量type/size/secret/hash，串行stage、非force推https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226的MLR，核remoteSHA/localreceipt。仅代码/报告/hash/误差标量和episode outcomes；禁CP/模型/原action/state/demo/latent/buffer/RNG/privatepayload/凭据/未审查bundle。TLS必要每命令OpenSSL+HTTP1.1仍验证证书。科学状态一直PARTIAL至全部独立验收，发布进度不等于接受。
