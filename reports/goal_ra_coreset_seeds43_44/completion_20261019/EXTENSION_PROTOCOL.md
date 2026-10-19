# RA-Coreset GOAL seed43/44 扩展合同

用户2026-10-19明确授权新增43和44，保持已完成seed42的方案与协议；旧no_auto_extend仅禁止未经授权扩展，本轮已获授权。无论43指标如何，完整执行预定44，不按中途结果挑seed或调参。只复用各自已验收Random/Coreset基线及state=True特征，旧42和其他实验只读，不重训或重评基线。

两组各官方GOAL六任务权重起步、独立空buffer、任务6→9每任务100完整epoch、batch10，400epoch/76100step/4ep99CP/200episode/20vector attempts。新增合计800epoch/152200step/8CP/400episode；三seed最终1200epoch/228300step/12CP/600条RA记录，加原Random/Coreset各600条用于只读比较。

RA算法、模型、损失、训练和评估源文件与42逐字节相同。仅审计/合同中的seed参数化、离线Random名单校验使用当前seed，以及从各自实际8×7专家动作窗口可靠重建demo/target/length身份。身份有歧义则停止；不得拿文件编号当时间、跨seed复用旧标签或旧feature哈希。身份只用于校验和诊断，不影响选样。原参考距离标定seed42、FP64、0.5视觉/0.5动作、参考集、tie容差1e-12与DistributedSampler seed0固定。

Random硬锚点仍是同seed原Random完整q无放回名单的前floor(q/2)，q499/445/452/985，锚点249/222/226/492；原K-Center S0有序名单必须与对应seed父Coreset一致；缺失锚点按Random顺序补入，只删除非锚点，以全候选最大最近距离、平均最近距离、固定文件ID顺序作最少交换。无新比例、SC/Event约束、Loss选样、几何硬底线或stable-slot。

训练seed/数据打乱/随机锚点/每stage全局动作RNG分别43或44，环境两组seed[s,s+1]。同seed原stateTrue缓存原字节及modal合同不改；新训练合同单独绑定来源。原BC+.1meanIFA、gamma.7、AdamW、warmup500逐optimizer-step线性、路径排序concat/shuffle/原buffer管理不变。IFA按本组实际记忆原算法计算，记录其与父差异。Task6入库前实际模型tensor和初始化全RNG必须与对应父Coreset相同，失败停止，不能调seed凑一致。

每seed依次8软件测试、父输入和4759真实窗口绑定、离线选样及所有删除候选独立校验、冻结、恰好4次隔离GPU原生更新及恢复测试，0环境/0策略rollout。通过后正式重新seed/官方起点，不接续预检状态，不重复已过预检。正式评估原0–19初态，每有效cell20条、10lane、600步、5零动作、argmax分量Normal，非独立轨迹Generator，不加680或9000预算。仅评估本seed新RA四个最终模型。

Q=/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_extension_20261019，W=Q/seed{s}，O=/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed{s}_20261019。沿用GOAL系统盘新产物授权，旧资产保持原位置只读。队列43准入/训练/评估/验收→44同流程→三seed聚合→退出后最终验收。工程失败停止队列，保留退出码日志；正常评估失败不重试。

恢复前确认进程停止且合同一致；只能本seed完整epoch恢复model/optimizer/scheduler/grads/modes/allRNG/cursor，存储按预提交清单幂等补齐，不能重复入库/计步。不能盲重启整个queue或部分globalRNG评估；不能保证原动作RNG和环境顺序则BLOCKED。禁止改冻结科学源码或重新贴合同。只读汇总故障可保留原错误后外置修复，不重跑环境。

完成须每seed所有数量、4CP严格加载、Task6tensor/RNG、真实入库与离线一致/锚点/配额/载荷/hash、全部矩阵/精确AUC/FWT/NBT/OldFinal/FinalAll、20attempt与退出无残留通过；三seed汇总报告逐seed配对差、均值样本SD(ddof1)、全部任务和负结果。queue.exit0及FINAL_QUEUE_ACCEPTANCE PASS后本地独立复算。完成前只能PARTIAL。

这次是在观察42后授权的探索性扩展，不能声称预注册三seed或统计显著；披露global动作RNG未隔离、原排序影响实际batch、IFA间接变化、官方pretrain state元数据unknown及调度补全。仅三seedGOAL，不扩OBJECT或其他seed。每次检查/实验完成按AGENTS推送相关代码/轻量证据至MLR，不推权重/演示/latent/buffer/原动作/RNG二进制/凭据。按已有授权两小时检查，遇工程阻塞处理，完成全部验收后删除自动化。
