# SC-v2 GOAL seed42 正式协议

2026-10-18。离线验收先完成，四任务合法交换5/2/4/6条，Task6–8共11条实际成员变化。用户随后明确授权：离线通过后先跑GOAL单seed，确认有效才扩3seed；评估“按上次实验的计划来”，扩展采用AUC严格提高且OldFinal不降低双门槛。

解释与落实：沿用上一轮已执行SC-Coreset的原评估计划，仅新SC-v2组200条、20向量attempt，原0–19初态、每cell20、10lane/env42,43、600步、5零动作、每stage重置global RNG42、argmax分量Normal。复用已有同seed Random和Coreset评估，不启用先前提出但未选定的600条逐轨迹Generator共同重评。三方法是同一配置协议，不是逐轨迹匹配的实际随机流；明确保留这一限制。后续若改变评估协议必须另定，不能混入当前结果。

官方GOAL权重起步、seed42、state=True、原stateTrue特征只读复用、空buffer；Task6–9各100完整epoch、batch10，共400epoch/76100step。BC+0.1平均IFA、gamma0.7、AdamW1e-4、warmup500、逐step线性调度、concat/shuffle/DistributedSampler seed0、网络跨任务继承不变。任务配额499/445/452/985。前端/模型/损失/训练器/原评估器字节保持父SC-v1相同。仅当前任务结束的入库规则改为SC-v2。内部arm标识仍为sc_coreset，独立SCv2目录、合同与报告区分版本，不覆盖旧SC。

SC-v2精确定义采用本轮离线协议和selection.py。参考Ω为候选所代表的不同(demo,target)，时间四分区×末端命令符号，终点并入末区；选中仍按窗口数计E。从原S0开始，只加尚未保存的监督身份；保护原示范与已支持终点；q不变；R和mean不超过S0加原1e-12容差；E严格降低。穷举合法交换按E、R、mean、加入/删除ID排序，无合法交换即停。原距离/全任务尺度、标定42、FP64不变。正式每次选样结果必须与离线有序ID完全一致。新元数据另存，载荷原样复制；旧任务不重选。独立监督数不减，非额外监督或频率权重。

IFA按本轮真实buffer重新选对，公式/算法不变，保存与父差异；不能声称实际IFA输入不变。Task6首次入库前实际模型tensor和初始化全RNG须与父Coreset相同，否则停止。仅ep99四个最终模型用于评估，不按loss或成功率挑选。

目标端五项采样性质测试；四任务实际storage集成结果与离线证据一致、选择器不消耗训练全局RNG；恰好4次隔离GPU原生BC+IFA更新和完整边界恢复检查，0环境测试/0策略rollout。正式训练独立进程重设seed，丢弃smoke状态。复用父已通过的模态/加载/原评估器资源审计，不把CPU测试冒称GPU通过。

工程恢复先确认进程停、核对合同与输入、留存失败；仅从本run完整epoch恢复model/optimizer/scheduler/grads/modes/allRNG/cursor；存储按已提交名单幂等补齐，不重抽、重复计步或入库。部分global RNG评估禁止盲重启；正常失败不重试，不能保证RNG和环境顺序则BLOCKED。冻结代码不悄改、不重贴合同。

正式后台train→evaluate→audit，退出后独立process acceptance。完成需400epoch76100step、四CP严格加载、200真实记录20attempt、原输入/载荷/配额/选择约束/IFA/Task6tensor一致、全部矩阵与精确AUC/FWT/NBT/OldFinal/FinalAll、pipeline/supervisor退出0且无残留。双门槛为AUC>63/80（78.75%）且OldFinal>=41/60（68.3333%），Random参考AUC80.9375%。全部工程与门槛通过后才准备43/44独立合同并复用对应基线；失败正常交付停止，不调阈值。

用户既有GOAL系统盘授权适用。新W=/root/mlr_ifa/experiment_workspaces/SCv2_Coreset_GOAL_seed42_20261018；新O=/root/mlr_ifa/experiments/SCv2_Coreset_GOAL_seed42_20261018。旧资产只读，不启动新付费资源。每轮按AGENTS向MLR发布代码、报告、脱敏轻量证据；不发布权重、载荷或原绑定动作。

解释边界：单seed探索性、后续条件扩展；离线只保证指定几何/构成约束、不保证行为；修改11条旧记忆属于很小干预；官方pretrain state元数据unknown、调度/ep99补全、全局动作RNG未逐轨迹隔离、IFA间接变化均披露。负结果也完成验收。
