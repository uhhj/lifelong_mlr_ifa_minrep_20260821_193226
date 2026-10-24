用户2026-10-23最新定时规则，优先覆盖所有旧短周期：按预计剩余完成时间安排下一次检查。预计1小时就1小时后，2小时就2小时后；没有固定3/5/8/15/30分钟上限，也不因内部Task切换频繁唤醒。优先以当前整组主训练结束、随后整组正式评估结束为下一检查点；以自身实测速度和剩余预算估计，保存/准备另计，不把全部队列与当前一组混淆。下一次醒来重新按实际剩余时间更新同一监控。无自身吞吐时可用最近可比实测作明确标注的不确定估计，不伪造精确总ETA，也不恢复高频轮询。健康无变化安静；失败、完成、需行动或重要阶段变化才通知。

最新2026-10-24 14:12 CST优先：snapshot_20261024_140806实际观察14:08:18，LOCAL_AUDIT、EVALUATION_PROGRESS_AUDIT及B2_TRAINING_AND_TRANSITION_AUDIT实际PASS_PROGRESS_NOT_FINAL。1875冻结项和科学合同不变，recovery06 controller2086/evaluate2364及环境正常。B2四阶段400完整epoch105500主更新全部完成，train6–9/diagnose6–9均exit0；三组主更新总316500，codec1900辅助更新另计。B2主epoch实录72.551298min，保存/准备/诊断/恢复额外与未知中断半轮另列，不称墙钟总时间。Task8原前90轮逐字段精确保留、最终100轮69300步，RECOVERY06_RESOLUTION实际PASS、重复完整epoch0。全部人口/身份顺序/payload/mask和逐epoch sampler/optimizer白名单与B1/B0绑定；Task6非计时字段同B1，先前CPU tensor相等证据保留。教师7–9当前数据与前阶段CP/codec/合同SHA绑定通过，本轮不是新CPU教师payload最终验收。
B2正式20/200、2完成vector+1进行中、1关闭cell、0技术异常；逐episode/attempt唯一性、记录初态hash/seed/步数/成功向量和加载记录绑定PASS，不算局部性能。两批288.530705/279.778361秒，均284.154533；14:08扣当前batch162.782514秒后余82.533318min线性计算。加加载/收尾暂参考15:35–15:50 CST，独立最终审计另计，只有两批非保证；按用户规则同一监控下一90分钟后检查整组评估结束附近，不内部阶段短轮询。
本地评估进度wrapper初次误要求新cell在首vector未完成前已有trials文件，FileNotFoundError保存在WRAPPER_ATTEMPT_01及原源。仅外置修为无complete事件且snapshot/download均未列文件时接受空trials，真实复算PASS，WRAPPER_RESOLUTION绑定；没有服务器修改/训练或环境重启。冻结源码保持不变。
capacity_readonly_20261024_140830/EVIDENCE与LOCAL_AUDIT实际PASS_NOT_FINAL：data root3.521987GB扣余新增和2GiB后余0.261286GB，system普通4.649193GB扣余新增和1GiB后余1.688873GB。仅稳定私有单链接stat抵扣，无payload读取/写入/前向/更新/环境；已保全恢复副本已计空闲，不重复信用。无删除/降预算/付费。
科学root仍engineering05、合同2f6af444986a5710cbc84572f25ec15f9b3d44ea72aacbd01fa3ad4e624681b0；recovery06仅编排，执行合同34d1fe1b29502d55f183c8ccd7d9a6ba8266768421374c2561f724904d22a61b。禁止任何重新deploy/controller或重训已完成阶段。旧中断/失败/私有恢复副本和额外5隔离更新证据全保留。全部controller自然exit0且无自身进程后才运行final_readonly→audit_final_local→describe_final；这些最终路径仍需真实完整12CP/600episode/60vector/30cell及初态字节验收，不能沿用进度PASS。
用户另授权的B0第一步/未来三步/GMM诊断已在B0_action_diagnostics_20261024_135651完整CPU只读验收并单独发布0580a65e3c08a832bb4fde3f642164ae9bc94f3c，不重跑。额外10524窗口331前向、CPU134.611113秒、0更新/环境/CUDA；最终describe_final已追加SHA绑定的独立成本项，仅AST待最终实际调用。所有原diag高估计数保留并外置校正。总体PARTIAL，无自动新seed；完成完整验收报告发布后才删除监控。

只跟踪用户新授权的 CAET-EM-v1 OBJECT seed41。先读 D:/论文课题研究/MLR+IFA/execution/20261023_caet_em_v1 的 USER_METHOD.txt、PROTOCOL.md、IMPLEMENTATION_NOTES.md、REPORT.md、RUN_STATUS.json、ACTIVE_ROOT.json 和最新证据。原位置探针A失败保留；此新路线不需要10mm/25mm门。旧ALIGN/TACR/readout等已完成或取消的队列不得恢复。没有自动seed扩展。

当前工程版本和接续点以顶部最新覆盖为准。不要重跑deploy.py/deploy_engineering05.py/build_engineering05.py或controller。只使用ACTIVE_ROOT.json当前工程05进行只读检查；检查旧工程04只限溯源，不作活跃进度。工程05复用完整codec和B0 Task6/7，必须从冻结接续controller继续，不重复主训练。监控时间遵循顶部用户最新规则，以预计整组训练或评估结束时间安排，旧短周期指令全部作废，不守候。健康无变化安静；只实质切换/失败/完成/需行动通知。

顺序：共享Task6证据和100epoch codec冻结→B0 READOUT_BC4→B1 EM→B2 EM_TF，每组固定Task6–9四阶段400epoch105500主步、200正式episodes20vector10cell；合计316500主更新600正式。各阶段ep99完整新模型CP；原RA成员与人口不变。B1/B2同一codec/初始decoder，各自从官方Temporal初始化、Task6无T/F应完全一致；后面只用自身上一阶段教师。三组共享最后帧modality3控制读出、modality0 IFA及四步动作；部署只采样执行第一动作再观测，32token/8帧不改、原argmax-component Normal语义。B1−B0解释接口，B2−B1解释T+F组合，旧RA/ALIGN仅外部参考。无途中科学失败筛选、无额外seed或消融。

每轮 check_progress.py只读→audit_progress.py SNAPSHOT，核源/输入/进程/GPU/磁盘/有效退出/完整epoch账目。不得把codec准入套三组主策略；各组ADMISSION_B0/B1/B2必须实际通过。当前数据证据1562样本、第一步1482合法、四步5868合法，模糊身份和尾部按独立mask保留人口；只读复用engineering03完成的labels/PCA/normalization。前3失败版本全部留证：精度比较、null身份索引、事件输出参数重名；0主更新/0正式环境，attempt02和03各3124视觉前向+640重复，attempt03另3隔离更新。新04自己另3准入。不能覆盖冻结源码，也不再导出完成Task6证据。

SSH root@jq1.9gpu.com:11920授权，凭据仅getpass/现有内存，不文件日志或参数。失败先确认相关进程停止、合同hash、原失败与完整epoch恢复文件，再版本化工程修复；科学结构/损失/有效标签/预算/seed不变。不得盲重启controller或部分globalRNG正式评估，不重试正常负结果。恢复须核model/optimizer/scheduler/grads/modes/requires_grad/allRNG/cursor/selected-best/数据与codec/teacher SHA；每次新版本自己的准入，已完成codec/训练不重跑。发现问题可自动工程修复，但必须留证，不把日志输出错误当科学失败。

共享及B2在/data，B0/B1在/root，具体见ENTRY。单组保守1.8GiB、shared .3GiB，实际模型与buffer估计约1.50GB；data root f_bfree留2GiB，system普通f_bavail留1GiB。每阶段实际核容量及已分配抵扣新增界，不删除旧资产/下载weights、features、rawactions/states/降预算/付费。

全部controller自然exit0且无自身进程后：remote.py final_readonly.py→audit_final_local.py SNAPSHOT SCALAR_DIR→describe_final.py FINAL_DIR SNAPSHOT。这些新最终wrapper仅AST通过尚待真实验收；必须12实际CPU严格完整新CP、codec验证最早best/统计SHA、B1/B2Task6 tensor相等、完整恢复态、独立600正式episode/60vector/30cell/全mask人口账目/原RA成员/初态hash/attempt唯一性、无残留与有效退出。外置wrapper错误可修，不重训/重评。保留全部负单元与参数/辅助计算/额外存储、重复探索限制，不称显著或保护保证。完整接受报告发布后删除该新监控，不自动扩seed。

每次检查/恢复/验收均按根AGENTS经本目录publish.py调用publish_mlr_branch流程，精确审查轻量type/size/secret/hash，串行stage、非force推https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226的MLR，核remoteSHA/localreceipt。仅代码/报告/hash/误差标量和episode outcomes；禁CP/模型/原action/state/demo/latent/buffer/RNG/privatepayload/凭据/未审查bundle。TLS必要每命令OpenSSL+HTTP1.1仍验证证书。科学状态一直PARTIAL至全部独立验收，发布进度不等于接受。
