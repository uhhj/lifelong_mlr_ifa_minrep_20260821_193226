用户2026-10-23最新定时规则，优先覆盖所有旧短周期：按预计剩余完成时间安排下一次检查。预计1小时就1小时后，2小时就2小时后；没有固定3/5/8/15/30分钟上限，也不因内部Task切换频繁唤醒。优先以当前整组主训练结束、随后整组正式评估结束为下一检查点；以自身实测速度和剩余预算估计，保存/准备另计，不把全部队列与当前一组混淆。下一次醒来重新按实际剩余时间更新同一监控。无自身吞吐时可用最近可比实测作明确标注的不确定估计，不伪造精确总ETA，也不恢复高频轮询。健康无变化安静；失败、完成、需行动或重要阶段变化才通知。

最新2026-10-24 15:28 CST优先：用户请求即时检查。snapshot_20261024_152405实际观察15:24:28，LOCAL_AUDIT与EVALUATION_PROGRESS_AUDIT实际PASS_PROGRESS_NOT_FINAL；1875冻结项无变化，recovery06 controller2086/evaluate2364及10环境健康，phase B2_evaluate，无恢复重启改参。三组各400完整epoch105500主更新全部完成；B0/B1各200正式20vector10cell及服务器audit exit0只读保留。B2已190/200、19完整vector+1进行中、9关闭cell、0技术异常，逐episode/attempt/记录初态hash/seed/步数/成功向量和加载绑定通过，不算部分性能。250metadata全SHA核验，无下载前进差异。
B2 19批均271.648944秒，其中一批67.717427秒为提前结束，原始时长保留。最后stage9/task9/group1于快照已进行65.773998秒，线性余3.431249分钟，加关闭/服务器审计另计；评估结束参考15:28–15:33，独立最终验收另计。按用户预计完成点规则，将同一监控改为5分钟后检查最后评估及自然退出，不等待或触发新环境。
capacity_readonly_20261024_152446及LOCAL_AUDIT实际PASS_NOT_FINAL：data root3.521217GB扣余新增和2GiB后余0.261233GB；system普通4.649153GB扣余新增和1GiB后余1.688832GB。仅stat抵扣稳定私有已分配，无payload读取/写入/前向/更新/环境，不删资产或新付费。
最终仍须全部自然exit0、无自身进程后执行final_readonly→audit_final_local→describe_final，实际12CP/600episode/60vector/30cell及初态字节、完整恢复/codec/teacher/RA绑定验收。不得把进度PASS作最终接受。科学rootengineering05合同2f6af444986a5710cbc84572f25ec15f9b3d44ea72aacbd01fa3ad4e624681b0保持；recovery06执行合同34d1fe1b29502d55f183c8ccd7d9a6ba8266768421374c2561f724904d22a61b保持。不要重新deploy/controller或重训已完成阶段。Task8前90轮保留和重复完整epoch0已真实绑定，B1/B2 Task6 tensor实际CPU相等证据保留；未知中断半轮与额外恢复5更新单列。
B0用户授权的动作/GMM CPU诊断已完整验收发布，不重跑；最终describe_final已追加其10524窗口331前向CPU134.611113秒SHA成本绑定，最终路径仍待实际调用。原包装失败/工程失败/diag计数原值全保留，不自动扩seed。总体PARTIAL，全部独立验收报告发布后才删除监控。

只跟踪用户新授权的 CAET-EM-v1 OBJECT seed41。先读 D:/论文课题研究/MLR+IFA/execution/20261023_caet_em_v1 的 USER_METHOD.txt、PROTOCOL.md、IMPLEMENTATION_NOTES.md、REPORT.md、RUN_STATUS.json、ACTIVE_ROOT.json 和最新证据。原位置探针A失败保留；此新路线不需要10mm/25mm门。旧ALIGN/TACR/readout等已完成或取消的队列不得恢复。没有自动seed扩展。

当前工程版本和接续点以顶部最新覆盖为准。不要重跑deploy.py/deploy_engineering05.py/build_engineering05.py或controller。只使用ACTIVE_ROOT.json当前工程05进行只读检查；检查旧工程04只限溯源，不作活跃进度。工程05复用完整codec和B0 Task6/7，必须从冻结接续controller继续，不重复主训练。监控时间遵循顶部用户最新规则，以预计整组训练或评估结束时间安排，旧短周期指令全部作废，不守候。健康无变化安静；只实质切换/失败/完成/需行动通知。

顺序：共享Task6证据和100epoch codec冻结→B0 READOUT_BC4→B1 EM→B2 EM_TF，每组固定Task6–9四阶段400epoch105500主步、200正式episodes20vector10cell；合计316500主更新600正式。各阶段ep99完整新模型CP；原RA成员与人口不变。B1/B2同一codec/初始decoder，各自从官方Temporal初始化、Task6无T/F应完全一致；后面只用自身上一阶段教师。三组共享最后帧modality3控制读出、modality0 IFA及四步动作；部署只采样执行第一动作再观测，32token/8帧不改、原argmax-component Normal语义。B1−B0解释接口，B2−B1解释T+F组合，旧RA/ALIGN仅外部参考。无途中科学失败筛选、无额外seed或消融。

每轮 check_progress.py只读→audit_progress.py SNAPSHOT，核源/输入/进程/GPU/磁盘/有效退出/完整epoch账目。不得把codec准入套三组主策略；各组ADMISSION_B0/B1/B2必须实际通过。当前数据证据1562样本、第一步1482合法、四步5868合法，模糊身份和尾部按独立mask保留人口；只读复用engineering03完成的labels/PCA/normalization。前3失败版本全部留证：精度比较、null身份索引、事件输出参数重名；0主更新/0正式环境，attempt02和03各3124视觉前向+640重复，attempt03另3隔离更新。新04自己另3准入。不能覆盖冻结源码，也不再导出完成Task6证据。

SSH root@jq1.9gpu.com:11920授权，凭据仅getpass/现有内存，不文件日志或参数。失败先确认相关进程停止、合同hash、原失败与完整epoch恢复文件，再版本化工程修复；科学结构/损失/有效标签/预算/seed不变。不得盲重启controller或部分globalRNG正式评估，不重试正常负结果。恢复须核model/optimizer/scheduler/grads/modes/requires_grad/allRNG/cursor/selected-best/数据与codec/teacher SHA；每次新版本自己的准入，已完成codec/训练不重跑。发现问题可自动工程修复，但必须留证，不把日志输出错误当科学失败。

共享及B2在/data，B0/B1在/root，具体见ENTRY。单组保守1.8GiB、shared .3GiB，实际模型与buffer估计约1.50GB；data root f_bfree留2GiB，system普通f_bavail留1GiB。每阶段实际核容量及已分配抵扣新增界，不删除旧资产/下载weights、features、rawactions/states/降预算/付费。

全部controller自然exit0且无自身进程后：remote.py final_readonly.py→audit_final_local.py SNAPSHOT SCALAR_DIR→describe_final.py FINAL_DIR SNAPSHOT。这些新最终wrapper仅AST通过尚待真实验收；必须12实际CPU严格完整新CP、codec验证最早best/统计SHA、B1/B2Task6 tensor相等、完整恢复态、独立600正式episode/60vector/30cell/全mask人口账目/原RA成员/初态hash/attempt唯一性、无残留与有效退出。外置wrapper错误可修，不重训/重评。保留全部负单元与参数/辅助计算/额外存储、重复探索限制，不称显著或保护保证。完整接受报告发布后删除该新监控，不自动扩seed。

每次检查/恢复/验收均按根AGENTS经本目录publish.py调用publish_mlr_branch流程，精确审查轻量type/size/secret/hash，串行stage、非force推https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226的MLR，核remoteSHA/localreceipt。仅代码/报告/hash/误差标量和episode outcomes；禁CP/模型/原action/state/demo/latent/buffer/RNG/privatepayload/凭据/未审查bundle。TLS必要每命令OpenSSL+HTTP1.1仍验证证书。科学状态一直PARTIAL至全部独立验收，发布进度不等于接受。
