# 最新交接：2026-10-23 00:16 CST，PARTIAL

用户最新纠正的扩展门槛只适用于新模块：每个TACR/TACRST的seed41完整AUC > 同seed RA41 701/960，分别扩自己的42/43。不是703/960三seed平均。已有ALIGN OBJECT三seed→GOAL的原门槛不改。

本目录PROTOCOL/IMPLEMENTATION_NOTES/REPORT/RUN_STATUS/USER_METHOD/GATE1_INDEX、launch_20261023_001410、snapshot_readonly_20261023_001540/LOCAL_AUDIT完整读后行动。远端Q=/data/mlr_ifa/experiment_workspaces/TACR_TACST_OBJECT_conditional_20261023，controller95924，合同3403553fc5bd26db0045a80c9e7db156fde52f0f633bd85833d520a7d397da2b，1462源码调度hash。状态WAITING_PRIOR_QUEUE_NATURAL_EXIT，六输出均未创建，所有GPU准入待各自执行。不能用CPU通过代替实际8调用（5主3表征）/0环境。

旧队列A=../20261022_align_object_extension_goal_conditional仍controller93501正常，最近本目录快照00:15 GPU是旧evaluate94045；旧完整23:47快照OBJECT42主训练400/105500、正式10/200，后续计数需A/check_progress再核，不能沿用当作当前读数。先按A正常检查/最终包装验收/门槛/发布；同时本目录check_progress.py→audit_progress.py <快照>只读核新controller等待、1462hash/进程GPU磁盘/已启动run账目。两队列都不能盲重启。新controller服务器自动等旧exit0+实际完成receipt/hash/GPU空闲/锁释放后才顺序TACR41→条件42/43→TACRST41→条件42/43。旧队列工程失败则留证停止，不绕过。当前等待无新模块ETA，实际GPU吞吐后估算。

控制残差R/U/gate必须正式加载；主100键与部署14键分别严格，E/D不部署；训练学生20键。新学生跨任务继承，优化器每阶段原协议重建，恢复含所有主/表征optimizer/scheduler/grads/modes/requires/RNG/游标/teachercache身份。ST键为自身上一阶段单一T/R/E，全阶段固定，真正RA成员、无未来旧全demo数据。Task7没有ST对也训练BC与effect。新增图41的8/9=156/207，42=152/205，43=180/200。原RA/ALIGN基线只读，不重复评估；原TAC已完成、EFTR用户提前终止、七步/AUX8/RSET/COVR/LIBERO50均取消，禁止恢复或清理。

完成KEY用package_final.py KEY生成manifest、resume_light_download.py <包>白名单并行全SHA、audit_final.py <包>本地独立原矩阵与双更新/教师/部署SHA/退出receipt。aggregate_local.py --variant TACR或TACRST --suite OBJECT --seed41 <包> --seed42 <包> --seed43 <包> --output <新目录>。未扩展组只报告41完整结果及门未过，不伪造三seed。TACR对ALIGN、RST对TACR、各自对RA，后续全部负结果保留，不作自动显著性。新包装/最终分支目前无真实完成run，须真实验收，外置修复不能动冻结源/科学阈值或重跑正常失败rollout。

容量按launch：新6run三system三data，每run1.6GiB；旧待运行四run亦计。system只f_bavail留1GiB、data可f_bfree留2GiB。每run前复核剩余界。保留旧资产，不付费扩容、不降精度batch预算、不下载模型/latent/buffer，不等满盘才处理。工程失败确认停进程/hash一致/证据保留后新版本恢复，不能重贴冻结源或盲重启部分globalRNG评估。

每检查/审计按AGENTS本目录publish.py经publish_mlr_branch审查类型大小secret/hash、精确stage非force推MLR并核remoteSHA/本地receipt；只轻量证据。A发生新检查也按A/publish.py发布。共用同一align-object-seed-goal监控，每30分钟，只有实质阶段变化/恢复失败/完成/需行动通知，健康等待安静。一轮保存发布后结束不守候。监控须两队列实际所选分支全部本地独立验收/比较/发布后才删，不能A完成就删除新任务。整体仍PARTIAL。

补充：import_readonly_20261023_002022六环境导入PASS；先前两次外置EGL配置导入失败保留于import_readonly_20261023_001655及import_readonly_20261023_001919，未改冻结源或训练。AUTOMATION_UPDATE_RECEIPT实际保存带前缀30分钟规则，提示末尾换行由工具trim，比较strip后相同。


2026-10-23 00:51 CST：snapshot_readonly_20261023_005155/LOCAL_AUDIT与CAPACITY_LOCAL_AUDIT PASS。controller95924仍正常等待旧队列，1462冻结hash不变、六输出不存在、未GPU准入；GPU仍为旧evaluate94045，无新失败/恢复/重启。data普通10.129GB/root14.999GB、system普通7.979GB；旧四待运行与新六run的保守剩余上界仍通过，data保留2GiB后余0.826GB、system保留1GiB后余1.752GB；这不是最终容量验收。旧ALIGN42正式计数见同轮A全快照，不沿用旧10/200。新队列无可用吞吐与ETA，正常等待不通知。


2026-10-23 01:25 CST：snapshot_readonly_20261023_012140/LOCAL_AUDIT PASS，新95924仍WAITING、1462hash不变、六输出不存在，无GPU准入。旧ALIGN42已在A/final_OBJECT_42_20261023_012252完整本地独立PASS，AUC83.020833%对同seedRA71.5625%；旧43已自动GPU准入并训练中，本N不越过旧GOAL条件流程。capacity_readonly_20261023_012323按剩余新增分配计当前43已写稳定私有1.009GB的信用，data root13.990GB，对最坏剩余11.017GB＋2GiB后仍余0.826GB；system普通7.979GB仍余1.752GB。不能拿部署时全额四run界重复扣当前已耗的空间而误报满盘；此仍是规划界，不是最终容量验收。

外置resume_light_download.py发现原复制工具仅允许/data元数据路径，已加明确的/root/mlr_ifa/experiments/前缀以匹配冻结TACR41/43与TACRST42输出；仍只.json/.jsonl/.exit、包目录内路径、<8MiB、manifest全SHA，不下载pth或payload。仅外置包装修改/语法验证，无服务器源码或合同修改；实际新组最终下载仍待真实验收。N无新性能结果或可用吞吐ETA。


2026-10-23 01:51–01:52 CST：snapshot_readonly_20261023_015144/LOCAL_AUDIT PASS，95924仍正常WAITING、1462hash不变、六输出未创建、无GPU准入/新性能结果。旧A的43正常389完整epoch101518步/0正式，Task9已89轮；GOAL门槛仍待完整43。capacity_readonly_20261023_015201按43稳定私有已分配1.184GB抵扣后，data root13.8146GB，对最坏剩余新增10.8415GB＋2GiB保留后余0.8256GB；system普通7.9794GB保留及新三run界后余1.7517GB。无旧资产删除、活动payload操作、付费扩容或预算改变；仍非全程最终容量验收。新组继续等A全部所选分支有效退出及互斥锁，无可用新GPU吞吐/总ETA。正常保存发布后结束，不守候。
