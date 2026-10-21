# 当前授权与服务器交接

用户2026-10-21最新明确启动CAET-v2 OBJECT41、43和RA+IFA LIBERO50 41、42、43。此范围覆盖历史只seed42限制。先读本目录PROTOCOL、IMPLEMENTATION_NOTES、RUN_STATUS、REPORT、gate1两份LOCAL_AUDIT、launch证据。源219文件与已验收42完全相同，各自绑定父RA，不重训重评基线。COVR全部永久取消，旧controller/A2不可恢复，CAET-v1 GOAL科学阻塞仍未解决，不纳入本队列。

新Q=/data/mlr_ifa/experiment_workspaces/CAET_v2_OBJECT_extension_LIBERO50_20261021，controller PID71536。顺序CAET41→CAET43→L50共享预训练+41→42→43。controller、QUEUE、ORCHESTRATION合同只读，严禁盲重启；失败先核对真实进程/exit/hash，保留失败并版本化恢复，正常失败不重试。不得在活跃worker运行时重复启动。SSH root@jq1.9gpu.com:11920，凭据仅内存/getpass。

CAET W=Q/object_seed{s}，O=/data/mlr_ifa/experiments/CAET_v2_OBJECT_seed{s}_20261021，P=/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed{s}_20261019。每seed400epoch105500主步4CP200records20vector；GMM/global RNG s/envs,s+1/0–19/600步；BC+.1meanIFA gamma.3，原RA有序成员781/740/780/660，sampler0与距离标定42。效果模型支持/留出门、迁移.05与负BC梯度冲突门、旧组NLL epsilon.05 dualrate.1/mu界[0,1]全不变。Task7零支持整阶段RA回退；8/9需实际留出才启用，包括保护。两seed的第一关已PASS，禁止重跑；41已8隔离GPU更新0环境PASS，勿重复。43仅排队，不可称GPU通过。当前41已9epoch1413步，507source/input hash与本地audit_progress PASS，0正式，任务6最终tensor待100epoch。

检查：python check_queue.py只读快照队列和实际GPU进程；python check_progress.py 41或43保存完整该seed快照，再python audit_progress.py <目录> <seed>。后者必须真实GPU_PREFLIGHT存在才能用，不能把尚未产生文件误判训练故障。只检查活跃seed，不重复已完成seed42。正常无异常不重启。每次保存报告并publish.py发布，禁止二进制payload/BUNDLE归档发布。

完成：package_final.py <seed>只在该supervisor和pipeline均0、PROCESS_EXIT_ACCEPTANCE PASS无worker时下载白名单轻量；audit_final.py <包目录> <seed>独立重算（当前仅语法检查，41/43尚未实证）。新seed是干净启动，不要求42旧失败recovery_environment01文件。四CP100键严格加载、Task6实际tensor/初始化RNG/IFA与父一致、原成员载荷/全部逐epoch主步/支持门和迁移保护账目/200记录20attempt全验收；完整后另做三seed41/42/43汇总、样本SD和逐seed负结果，seed42引用原final_20261021_231144，不重训重评。历史v1分别报告，因果时序缺陷保留。收集效果模型额外更新曝光及耗时。

L50 entry是本Q/LIBERO50_seed{s}.json与.sh；引用旧Q独立worker_entrypoint_libero50.py和原冻结bundle，不调用旧总controller。source=本Q/libero50/seed{s}/source，shared W=本Q/libero50/shared_pretraining，shared O=/data/mlr_ifa/experiments/LIBERO50_reproduced_pretraining_seed42_20261021。41 O=/root/mlr_ifa/experiments/RA_Coreset_LIBERO50_seed41_20261021（系统盘），42/43 O=/data/mlr_ifa/experiments/RA_Coreset_LIBERO50_seed{s}_20261021。官方50数据已50/50完整审计不重复下载，shared1941000+下游各859400合计4519200更新不减。共享真实4隔离GPU更新0环境，下游各4隔离+1零动作原生接口；此刻均未通过GPU，必须真实执行。5阶段各5任务100epoch，gamma.1，5ep99/1500records150vector每seed；4500正式总计。FP32/原RA/IFA/stateTrue/原GMM、全局RNG原协议，不引入CAET或残差。共享pretrain可能很长，真实吞吐后估时。

资源待办必须主动解决：当前数据约9.13GiB（41初始恢复已占）系统5.50GiB。每seed45388feature文件约2.82GiB，加检查点/恢复/共享预训与两CAET，总累计存储可能不足。队列每入口至少4.5GB（L50）、1.6GB（CAET）空闲保护，不保证整个未来容量。依据真实产物做工程存储方案；可以在新、尚未冻结入口采用版本化路径分配，或审核仅本轮新产物的逐字节验证无损归档，禁止删旧实验/更改冻结源/默改合同或减少预算。不得等满盘才处理；不新付费资源。当前科学源未改。若更改调度需保留旧队列/退出证据，安全边界交接，不杀活跃训练或评估来省磁盘。原L50轻量包装/审计位于execution/20261020_three_methods/package_libero50_light.py、audit_libero50_light.py，需核对新路径后使用，保持payload禁下载。

本地publication通过publish_mlr_branch远端MLR，精确stage/类型体积secret/hash审查、非force、remoteSHA核验并存receipt。当前是PARTIAL启动证明，不算任何新seed完成。正常监控每小时，近阶段完成可30分钟；实际有异常自主处理，不能绕过科学失败或盲重启部分global RNG评估。全部完成后交付OBJECT三seed和L50三seed完整结果，不自动扩更多seed/GOAL。总监控保留其他科学阻塞的真实状态。
