# TAC-AUX8：八步输入与独立训练用下一动作头

观察时间2026-10-22 13:20 CST，整体PARTIAL。用户修订要求已落实并实际重新开始训练。

旧七步TAC在297个完整epoch/68373主步、0正式episode处由用户要求中断，旧调度与实验树已停止并保留；不是完成或NO_GAIN。证据见../20261022_caet_modules_object_seed41/tac_user_stop_20261022_131004，旧689hash通过。新部署前再次确认旧进程全无、旧合同未变。

新实验CAET-v2+TAC_AUX8 OBJECT41，从官方原始OBJECT权重/空本run缓冲开始。主八步输入、BC、Temporal Transformer、部署GMM及闭环源码未改；独立H_aux预测下一动作，F预测对应合法下一转移。教师NLL只更新H_aux；效果和辅助头保护的梯度仅经H_aux输入更新共享Temporal，部署GMM没有直接辅助梯度。保护出口同步从部署GMM移到H_aux，以免留下下一动作目标冲突。主state_dict仍100键，训练用辅助头独立保存，正式评估不加载。完整设计及探索性超参数在PROTOCOL.md预先冻结。

真实GPU隔离准入PASS：10次主优化器调用+5次独立辅助头调用，0环境/0正式更新。关闭迁移时主模型/optimizer/scheduler/grads/modes/RNG与原RA精确一致，主+辅助完整状态恢复和主动效果分支重复恢复一致。共享Temporal效果梯度范数平方3.14798509e-8，部署GMM直接梯度缺省；构造辅助NLL恶化-14.371857→-14.270255后，保护响应降低至-14.290219，非全局或闭环保证。冲突门两项检查均保留BC下降。

progress_20261022_132018：新controller83788、pipeline83907、train6 PID83920。691个源/入口/调度hash通过；Task6已2完整epoch/314主步，独立H_aux教师314更新/2652窗口曝光；迁移与保护0符合Task6设计。prepare/labels6/gpu_preflight真实exit0，无失败。本地audit_progress逐数量及GPU证据独立PASS。Task6最终100键张量与RA精确比较尚待ep99，不提前宣称通过。

Task6前两epoch6.04/5.22秒，早期仅可估本阶段余下约8–10分钟；Task8/9新辅助头和保护开销未实测，完整训练/闭环ETA暂不可靠。正式预算400epoch105500主步、4ep99、200episode20vector未减。H_aux更新与效果拟合更新独立记账。data普通6.107GB/root10.977GB、system普通5.902GB，data保留2GiB，不删旧资产。

新有限队列TAC_AUX8→原冻结RSET→原冻结EFTR，旧queue与新queue双锁。B/C没有混入新的语义LODO原型，仍各需自己的GPU准入。COVR和LIBERO50预训练保持用户取消。新版本最终本地轻量打包/验收需核验H_aux状态链和教师账目，不得直接套用旧七步TAC的预检数目或输入差异检查；旧基线只读。
