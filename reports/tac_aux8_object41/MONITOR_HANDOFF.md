# TAC-AUX8：八步输入与独立训练用下一动作头

观察时间2026-10-22 13:20 CST，整体PARTIAL。用户修订要求已落实并实际重新开始训练。

旧七步TAC在297个完整epoch/68373主步、0正式episode处由用户要求中断，旧调度与实验树已停止并保留；不是完成或NO_GAIN。证据见../20261022_caet_modules_object_seed41/tac_user_stop_20261022_131004，旧689hash通过。新部署前再次确认旧进程全无、旧合同未变。

新实验CAET-v2+TAC_AUX8 OBJECT41，从官方原始OBJECT权重/空本run缓冲开始。主八步输入、BC、Temporal Transformer、部署GMM及闭环源码未改；独立H_aux预测下一动作，F预测对应合法下一转移。教师NLL只更新H_aux；效果和辅助头保护的梯度仅经H_aux输入更新共享Temporal，部署GMM没有直接辅助梯度。保护出口同步从部署GMM移到H_aux，以免留下下一动作目标冲突。主state_dict仍100键，训练用辅助头独立保存，正式评估不加载。完整设计及探索性超参数在PROTOCOL.md预先冻结。

真实GPU隔离准入PASS：10次主优化器调用+5次独立辅助头调用，0环境/0正式更新。关闭迁移时主模型/optimizer/scheduler/grads/modes/RNG与原RA精确一致，主+辅助完整状态恢复和主动效果分支重复恢复一致。共享Temporal效果梯度范数平方3.14798509e-8，部署GMM直接梯度缺省；构造辅助NLL恶化-14.371857→-14.270255后，保护响应降低至-14.290219，非全局或闭环保证。冲突门两项检查均保留BC下降。

progress_20261022_132018：新controller83788、pipeline83907、train6 PID83920。691个源/入口/调度hash通过；Task6已2完整epoch/314主步，独立H_aux教师314更新/2652窗口曝光；迁移与保护0符合Task6设计。prepare/labels6/gpu_preflight真实exit0，无失败。本地audit_progress逐数量及GPU证据独立PASS。Task6最终100键张量与RA精确比较尚待ep99，不提前宣称通过。

Task6前两epoch6.04/5.22秒，早期仅可估本阶段余下约8–10分钟；Task8/9新辅助头和保护开销未实测，完整训练/闭环ETA暂不可靠。正式预算400epoch105500主步、4ep99、200episode20vector未减。H_aux更新与效果拟合更新独立记账。data普通6.107GB/root10.977GB、system普通5.902GB，data保留2GiB，不删旧资产。

新有限队列TAC_AUX8→原冻结RSET→原冻结EFTR，旧queue与新queue双锁。B/C没有混入新的语义LODO原型，仍各需自己的GPU准入。COVR和LIBERO50预训练保持用户取消。新版本最终本地轻量打包/验收需核验H_aux状态链和教师账目，不得直接套用旧七步TAC的预检数目或输入差异检查；旧基线只读。

## 下一步

Q=/data/mlr_ifa/experiment_workspaces/CAETv2_TAC_AUX8_OBJECT41_20261022，A worker Q/TAC_AUX8，O=/data/mlr_ifa/experiments/CAETv2_TAC_AUX8_OBJECT_seed41_20261022。B/C仍使用原Q=/data/mlr_ifa/experiment_workspaces/CAETv2_TAC_RSET_EFTR_OBJECT41_20261022各worker，源合同不变；只能由新有限queue继续，禁止复活旧controller82742。

在本目录check_progress.py只读保存，再audit_progress.py <目录>，正常保存报告发布后结束，不守候。失败必须查真实退出/停止/hash/保留证据才版本化修复，禁止盲重启部分globalRNG评估。主训练完整epoch恢复已包括H_aux/optimizer/grads/modes/requires_grad/steps与全部原状态；H_aux教师只在合法非效果留出样本更新，可能小于105500。Task7主RA回退但H_aux仍学教师，须分清主迁移关闭与独立头优化。Task8/9自身F留出通过才开迁移与保护。

最终主4CP严格100键/成员载荷/IFA/初始化与Task6tensor/200正式20attempt/各有效exit0无残留，再独立辅助头4CP及历史引用hash、教师连续账目、原episode指标复算；与RA和原CAET-v2分开报告，不混旧7步TAC。现有旧任务package_final.py/audit_final.py尚不适用AUX8，完成前需外置版本适配，不改冻结源。

按AGENTS本目录publish.py审查白名单代码和元数据推publish_mlr_branch远端MLR，核remoteSHA与receipt，不发布原始payload/模型或bundle。监控caet-v2-object41每30分钟，实质变化才通知，不建新任务。


## 2026-10-22T13:25:08.579931+08:00 常规进度

证据progress_20261022_132501/SNAPSHOT.json、LOCAL_AUDIT.json、ETA.json。Task6正常47完整epoch/7379主步，H_aux教师7379更新/62322样本曝光，正式0。691冻结hash、逐epoch主步及独立头计数、GPU准入绑定独立PASS；已结束prepare/labels6/preflight均exit0，controller83788/train83920仍活跃，无失败/无恢复。Task6迁移与保护仍0，最终tensor同父须等ep99。RSET/EFTR未启动。

最近10epoch平均5.471秒，Task6余53epoch约4.83分钟（仅训练，入库另计），整组ETA仍未实测。data普通5.765GB/root10.635GB，system普通5.902GB。正常继续原进程，没有重复预检或改源。整体PARTIAL，健康常规进度不发通知。


## 2026-10-22T13:52:37.786541+08:00 用户停止所有训练（覆盖此前继续执行指令）

用户要求停止训练并清理废用数据。所有当前训练与调度已停止，GPU 空闲；caet-v2-object41 自动化已删除。不得恢复 TAC_AUX8、旧 TAC、RSET 或 EFTR 队列。TAC_AUX8 182 完整 epoch/34314 主更新/0 正式；旧 TAC 297 完整 epoch/68373 主更新/0 正式；RSET/EFTR 未启动。状态 USER_STOPPED，不是实验完成或工程失败。

已清理 37 个通过预检的临时序列化副本，实测释放 11.460 GB，6 个合同引用候选保留。正式检查点、实际训练恢复点、数据、特征、buffer 与审计记录保留。清单和独立核对见 ../20261022_stop_cleanup/REPORT.md、LOCAL_CLEANUP_AUDIT.json。历史进度和协议保留但不再是继续运行授权。
