用户2026-10-23最新定时规则，优先覆盖所有旧短周期：按预计剩余完成时间安排下一次检查。预计1小时就1小时后，2小时就2小时后；没有固定3/5/8/15/30分钟上限，也不因内部Task切换频繁唤醒。优先以当前整组主训练结束、随后整组正式评估结束为下一检查点；以自身实测速度和剩余预算估计，保存/准备另计，不把全部队列与当前一组混淆。下一次醒来重新按实际剩余时间更新同一监控。无自身吞吐时可用最近可比实测作明确标注的不确定估计，不伪造精确总ETA，也不恢复高频轮询。健康无变化安静；失败、完成、需行动或重要阶段变化才通知。

最新2026-10-24 03:16 CST：snapshot_20261024_031425/LOCAL_AUDIT、EVALUATION_PROGRESS_AUDIT及TRANSITION_AND_B2_ADMISSION_AUDIT实际PASS_PROGRESS_NOT_FINAL。观察03:14:48，1875hash不变，controller124820/GPU135609健康，phase B2_train6。B1完整400epoch105500主步及200正式/20vector/10关闭cell全部完成，evaluate/audit自然exit0；本地全episode/attempt唯一性、成功向量、初态记录hash/seed/步数/策略加载绑定通过，服务器4CP审计及METRICS SHA绑定。B1 vector实测总89.915804min，仍待三组完整独立CPU/初态字节/最终路径验收，不提前作收益结论。B0同样完整，只读复用，无重训重评。
B2已通过自己的3隔离GPU优化器调用/0环境/完整下一步恢复/最新三模态依赖/1–8长度/未来隔离/效果路径/空mask/GMM及T/F梯度/codec冻结准入，22826763参数。下载82完整epoch12874主步，Task6人口1562/157batch，每轮5868合法380屏蔽；快照后epoch前进已记录。已记录82轮的全部非计时字段（包括loss、LR、sampler顺序、计数）与B1相应Task6逐项精确相等；dataset与optimizer白名单也相同。此为元数据前缀一致，不冒充最终Task6 tensor相等；该项仍须最终CPU实际验收。Task6 T/F关闭，Task7–9尚未实测。
B2最近10轮4.892962秒，Task6余18轮约1.467889min；B1 Task7–9实测54.539020min作为最近可比基线，合计剩余基线56.006908min。B2后续T/F双解码反向/教师缓存及准备增加耗时尚未测，明确另加约20min规划余量（不是实测也不是保证上界），暂按约75分钟后检查整组主训练结束附近，训练完成参考04:15–04:50 CST。不为Task切换提前短轮询；B2评估及全队列最终完成ETA仍未知，后续实际吞吐再修正。
capacity_readonly_20261024_031439及LOCAL_AUDIT实际PASS_NOT_FINAL：稳定B2已分配0.366088GB抵扣余新增界；data root4.547899GB扣余新增和2GiB后余0.629070GB，system普通4.655075GB扣余新增和1GiB后余1.694755GB。仅stat，无payload读取/服务器写入/前向/更新/环境。无删除/降预算/付费，无恢复重启改参；codec冻结不变、科学状态PARTIAL。全部600正式和最终本地验收报告发布后删除监控，不自动扩seed。

只跟踪用户新授权的 CAET-EM-v1 OBJECT seed41。先读 D:/论文课题研究/MLR+IFA/execution/20261023_caet_em_v1 的 USER_METHOD.txt、PROTOCOL.md、IMPLEMENTATION_NOTES.md、REPORT.md、RUN_STATUS.json、ACTIVE_ROOT.json 和最新证据。原位置探针A失败保留；此新路线不需要10mm/25mm门。旧ALIGN/TACR/readout等已完成或取消的队列不得恢复。没有自动seed扩展。

当前工程版本和接续点以顶部最新覆盖为准。不要重跑deploy.py/deploy_engineering05.py/build_engineering05.py或controller。只使用ACTIVE_ROOT.json当前工程05进行只读检查；检查旧工程04只限溯源，不作活跃进度。工程05复用完整codec和B0 Task6/7，必须从冻结接续controller继续，不重复主训练。监控时间遵循顶部用户最新规则，以预计整组训练或评估结束时间安排，旧短周期指令全部作废，不守候。健康无变化安静；只实质切换/失败/完成/需行动通知。

顺序：共享Task6证据和100epoch codec冻结→B0 READOUT_BC4→B1 EM→B2 EM_TF，每组固定Task6–9四阶段400epoch105500主步、200正式episodes20vector10cell；合计316500主更新600正式。各阶段ep99完整新模型CP；原RA成员与人口不变。B1/B2同一codec/初始decoder，各自从官方Temporal初始化、Task6无T/F应完全一致；后面只用自身上一阶段教师。三组共享最后帧modality3控制读出、modality0 IFA及四步动作；部署只采样执行第一动作再观测，32token/8帧不改、原argmax-component Normal语义。B1−B0解释接口，B2−B1解释T+F组合，旧RA/ALIGN仅外部参考。无途中科学失败筛选、无额外seed或消融。

每轮 check_progress.py只读→audit_progress.py SNAPSHOT，核源/输入/进程/GPU/磁盘/有效退出/完整epoch账目。不得把codec准入套三组主策略；各组ADMISSION_B0/B1/B2必须实际通过。当前数据证据1562样本、第一步1482合法、四步5868合法，模糊身份和尾部按独立mask保留人口；只读复用engineering03完成的labels/PCA/normalization。前3失败版本全部留证：精度比较、null身份索引、事件输出参数重名；0主更新/0正式环境，attempt02和03各3124视觉前向+640重复，attempt03另3隔离更新。新04自己另3准入。不能覆盖冻结源码，也不再导出完成Task6证据。

SSH root@jq1.9gpu.com:11920授权，凭据仅getpass/现有内存，不文件日志或参数。失败先确认相关进程停止、合同hash、原失败与完整epoch恢复文件，再版本化工程修复；科学结构/损失/有效标签/预算/seed不变。不得盲重启controller或部分globalRNG正式评估，不重试正常负结果。恢复须核model/optimizer/scheduler/grads/modes/requires_grad/allRNG/cursor/selected-best/数据与codec/teacher SHA；每次新版本自己的准入，已完成codec/训练不重跑。发现问题可自动工程修复，但必须留证，不把日志输出错误当科学失败。

共享及B2在/data，B0/B1在/root，具体见ENTRY。单组保守1.8GiB、shared .3GiB，实际模型与buffer估计约1.50GB；data root f_bfree留2GiB，system普通f_bavail留1GiB。每阶段实际核容量及已分配抵扣新增界，不删除旧资产/下载weights、features、rawactions/states/降预算/付费。

全部controller自然exit0且无自身进程后：remote.py final_readonly.py→audit_final_local.py SNAPSHOT SCALAR_DIR→describe_final.py FINAL_DIR SNAPSHOT。这些新最终wrapper仅AST通过尚待真实验收；必须12实际CPU严格完整新CP、codec验证最早best/统计SHA、B1/B2Task6 tensor相等、完整恢复态、独立600正式episode/60vector/30cell/全mask人口账目/原RA成员/初态hash/attempt唯一性、无残留与有效退出。外置wrapper错误可修，不重训/重评。保留全部负单元与参数/辅助计算/额外存储、重复探索限制，不称显著或保护保证。完整接受报告发布后删除该新监控，不自动扩seed。

每次检查/恢复/验收均按根AGENTS经本目录publish.py调用publish_mlr_branch流程，精确审查轻量type/size/secret/hash，串行stage、非force推https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226的MLR，核remoteSHA/localreceipt。仅代码/报告/hash/误差标量和episode outcomes；禁CP/模型/原action/state/demo/latent/buffer/RNG/privatepayload/凭据/未审查bundle。TLS必要每命令OpenSSL+HTTP1.1仍验证证书。科学状态一直PARTIAL至全部独立验收，发布进度不等于接受。
