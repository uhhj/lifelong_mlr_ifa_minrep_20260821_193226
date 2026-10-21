# 当前授权与服务器交接

用户2026-10-21最新明确启动CAET-v2 OBJECT41、43和RA+IFA LIBERO50 41、42、43。此范围覆盖历史只seed42限制。先读本目录PROTOCOL、IMPLEMENTATION_NOTES、RUN_STATUS、REPORT、gate1两份LOCAL_AUDIT、launch证据。源219文件与已验收42完全相同，各自绑定父RA，不重训重评基线。COVR全部永久取消，旧controller/A2不可恢复，CAET-v1 GOAL科学阻塞仍未解决，不纳入本队列。

新Q=/data/mlr_ifa/experiment_workspaces/CAET_v2_OBJECT_extension_LIBERO50_20261021，controller PID71536。顺序CAET41→CAET43→L50共享预训练+41→42→43。controller、QUEUE、ORCHESTRATION合同只读，严禁盲重启；失败先核对真实进程/exit/hash，保留失败并版本化恢复，正常失败不重试。不得在活跃worker运行时重复启动。SSH root@jq1.9gpu.com:11920，凭据仅内存/getpass。

CAET W=Q/object_seed{s}，O=/data/mlr_ifa/experiments/CAET_v2_OBJECT_seed{s}_20261021，P=/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed{s}_20261019。每seed400epoch105500主步4CP200records20vector；GMM/global RNG s/envs,s+1/0–19/600步；BC+.1meanIFA gamma.3，原RA有序成员781/740/780/660，sampler0与距离标定42。效果模型支持/留出门、迁移.05与负BC梯度冲突门、旧组NLL epsilon.05 dualrate.1/mu界[0,1]全不变。Task7零支持整阶段RA回退；8/9需实际留出才启用，包括保护。两seed的第一关已PASS，禁止重跑；41已8隔离GPU更新0环境PASS，勿重复。43仅排队，不可称GPU通过。当前41已9epoch1413步，507source/input hash与本地audit_progress PASS，0正式，任务6最终tensor待100epoch。

检查：python check_queue.py只读快照队列和实际GPU进程；python check_progress.py 41或43保存完整该seed快照，再python audit_progress.py <目录> <seed>。后者必须真实GPU_PREFLIGHT存在才能用，不能把尚未产生文件误判训练故障。只检查活跃seed，不重复已完成seed42。正常无异常不重启。每次保存报告并publish.py发布，禁止二进制payload/BUNDLE归档发布。

完成：package_final.py <seed>只在该supervisor和pipeline均0、PROCESS_EXIT_ACCEPTANCE PASS无worker时下载白名单轻量；audit_final.py <包目录> <seed>独立重算（当前仅语法检查，41/43尚未实证）。新seed是干净启动，不要求42旧失败recovery_environment01文件。四CP100键严格加载、Task6实际tensor/初始化RNG/IFA与父一致、原成员载荷/全部逐epoch主步/支持门和迁移保护账目/200记录20attempt全验收；完整后另做三seed41/42/43汇总、样本SD和逐seed负结果，seed42引用原final_20261021_231144，不重训重评。历史v1分别报告，因果时序缺陷保留。收集效果模型额外更新曝光及耗时。

L50 entry是本Q/LIBERO50_seed{s}.json与.sh；引用旧Q独立worker_entrypoint_libero50.py和原冻结bundle，不调用旧总controller。source=本Q/libero50/seed{s}/source，shared W=本Q/libero50/shared_pretraining，shared O=/data/mlr_ifa/experiments/LIBERO50_reproduced_pretraining_seed42_20261021。41 O=/root/mlr_ifa/experiments/RA_Coreset_LIBERO50_seed41_20261021（系统盘），42/43 O=/data/mlr_ifa/experiments/RA_Coreset_LIBERO50_seed{s}_20261021。官方50数据已50/50完整审计不重复下载，shared1941000+下游各859400合计4519200更新不减。共享真实4隔离GPU更新0环境，下游各4隔离+1零动作原生接口；此刻均未通过GPU，必须真实执行。5阶段各5任务100epoch，gamma.1，5ep99/1500records150vector每seed；4500正式总计。FP32/原RA/IFA/stateTrue/原GMM、全局RNG原协议，不引入CAET或残差。共享pretrain可能很长，真实吞吐后估时。

资源待办必须主动解决：当前数据约9.13GiB（41初始恢复已占）系统5.50GiB。每seed45388feature文件约2.82GiB，加检查点/恢复/共享预训与两CAET，总累计存储可能不足。队列每入口至少4.5GB（L50）、1.6GB（CAET）空闲保护，不保证整个未来容量。依据真实产物做工程存储方案；可以在新、尚未冻结入口采用版本化路径分配，或审核仅本轮新产物的逐字节验证无损归档，禁止删旧实验/更改冻结源/默改合同或减少预算。不得等满盘才处理；不新付费资源。当前科学源未改。若更改调度需保留旧队列/退出证据，安全边界交接，不杀活跃训练或评估来省磁盘。原L50轻量包装/审计位于execution/20261020_three_methods/package_libero50_light.py、audit_libero50_light.py，需核对新路径后使用，保持payload禁下载。

本地publication通过publish_mlr_branch远端MLR，精确stage/类型体积secret/hash审查、非force、remoteSHA核验并存receipt。当前是PARTIAL启动证明，不算任何新seed完成。正常监控每小时，近阶段完成可30分钟；实际有异常自主处理，不能绕过科学失败或盲重启部分global RNG评估。全部完成后交付OBJECT三seed和L50三seed完整结果，不自动扩更多seed/GOAL。总监控保留其他科学阻塞的真实状态。


## 最新2026-10-22 00:54 CST
progress_seed41_20261022_004440完整349epoch87038步/0正式、507冻结hash独立PASS；queue_20261022_005409轻量378epoch97536步，Task9 epoch77，无重启。Task6全部tensor同RA。8/9留出均PASS（MAE .273378/.258683；均值.490445/.472762；置换.294773/.294356），额外850+990=1840更新。00:45机制8迁移7602/冲突6224/保护2915，9前49epoch迁移4741/冲突3129/保护0；随后resource_mechanism_20261022_004649约束至task9epoch54：epoch49首次一个组超限、mu升.126333后回0；不要延用零保护称全阶段没有激活，也不误称闭环改善。Task8 fit seconds=-5.327923，wall-clock异常保留标不可用，不重训/不把负数当成本。所有双对偶账目独立复算PASS。
预计01:02结束训练，73分钟参考完整评估+验收，41约02:15–02:50；下一轮核正式记录attempt和评估进程，不能提前完成或部分AUC。43未GPU准入，L50未GPU准入。
容量已新增独立、未改旧合同的后备：capacity_recovery_v1.py与Q/capacity_recovery_v1/CONTRACT.json，watcher PID72538，ARMED_WAITING_ORIGINAL_EXIT。先查check_queue新增字段；它仅在旧controller.exit1且FAILURE明确RESOURCE_CAPACITY、已完receipt验证/GPU无实验/pending输出不存在/队列锁释放后，自动启动剩余L50 42/43，使用/data statvfs.f_bfree真实root空闲且保留2GiB。不是现在重启旧队列或与当前并发。普通可用9.34GB、root全部空闲14.21GB，原守卫只计前者。没有tune2fs/改挂载/删除资产/改科学源；禁止把原guard失败当实验失败或改原exit。后备有效退出和receipt都在Q/capacity_recovery_v1，最终需结合原prefix与后备suffix核验。原队列exit0则后备NO_RECOVERY_NEEDED退出0；非容量错误停止留证，由助手单独修复，不能盲重试。8纯CPU边界测试通过，不是GPU测试。全资源预算仍需实际监测，保留2GiB不足就阻塞，不绕过。
只读codec探针storage_codec_probe_20261022_004847：本轮preflight文件346MB和100buffer已逐字节逆变换SHA一致；byte-plane+zlib恢复文件82.56%、样本buffer66.67%体积；无服务器写入/归档/删除，禁止把探针当作已释放空间。不需要现在删除或归档活动文件。原IMPLEMENTATION_NOTES称共享预训优先系统盘只是早期安排，实际冻结entry共享仍/data，41下游/system，以entry和本hand-off实际路径为准。


## 2026-10-22 01:31更新

最新2026-10-22 01:31 CST：progress_seed41_20261022_013003完成400epoch105500主步，train6–9/retain6–9全部exit0，507冻结hash与Task6实际tensor同RA、本地逐epoch/逐步账目PASS。Task8/9迁移实际7602/9103步，冲突关闭6224/6998，保护非零2915/4186步；两预测器留出PASS，拟合额外1840更新。Task7全RA回退。正式60/200条、6完成vector+1进行中、0技术异常；evaluate PID72624及10环境正常。新增外置audit_evaluation_progress.py <目录> <seed>，逐episode初态hash/index/envseed/600步及attempt唯一性/成功向量独立PASS，cell summary留最终包。6批均249.01秒，剩余约58分钟加最终审计，预计02:30–03:00 CST供参考。仍PARTIAL，无部分AUC或性能结论。未重启、未改科学源、未重复预检；43和LIBERO50仍排队未GPU准入。容量watcher仍ARMED未接管，data普通空闲9.205GB/root全空闲14.075GB，system普通5.902GB；无需对活动payload操作。下一轮继续核41正式评估，完成后package_final/audit_final 41并确认43自动准入。


## 2026-10-22 01:58更新

最新2026-10-22 01:58 CST：progress_seed41_20261022_015731，seed41正式130/200条，13完成vector+1进行中，0技术异常，evaluate72624及10环境正常；400epoch105500主步、Task8/9迁移7602/9103与保护2915/4186不变。507冻结hash及逐epoch/逐步/逐episode身份与attempt成功向量独立PASS。13批均243.30秒，剩余评估约28.4分钟加最终验收，预计02:30–02:55参考。正常无重启，无部分AUC。43与LIBERO50仍排队未GPU准入，容量watcher仍ARMED未接管，data普通9.205GB/root全部14.075GB、system普通5.902GB。下一轮若完成先package_final.py 41和audit_final.py <包> 41真实独立验收，并核43自动启动；若仍评估则继续原进程，禁止盲重启。
