# RA-Coreset GOAL seed42：沿用上次实验协议

2026-10-18。用户要求执行RA压缩包和补充附件，并明确“实验协议和上次实验一样（此处以我说的为准）”。本实际协议优先于包内设计；原包原样保留，SHA256=0ce33e6d238fd5545b142f8b2dd33fac5edf7a744c91c1fa952799d288f4a805。

## 明确覆盖的包内条款

沿用SC-v2已经执行的原训练加载器和原评估协议：按真实buffer文件名字典序、原random.Random(seed).shuffle、DistributedSampler(seed0)。不启用stable-slot adapter。保留每stage重置global动作RNG，原argmax分量后Normal采样，不启用E1 epsilon bank/逐轨迹Generator。本轮仅RA新组200条，原Coreset和Random模型/指标直接复用，不重训、不重评基线；包内400条新E1预算取消。共同配置协议不等于实际逐轨迹噪声相同；不得宣称已隔离共同旧样本重排、IFA间接变化或动作RNG混杂。

参考代码中的stable_slots及其合成测试仅保留原交付来源，不在实际加载路径调用。实际模型/BC/IFA/优化/加载/评估计算保持上轮相同；新namespace为ra_exp，内部arm为ra_coreset。train/evaluate只改arm、存储模块import和父路径变量，不改训练/评估语义。

## RA算法

完整当前任务U按原文件名排序；q=ceil(N/2)。重现同seed原Random无放回sample(U,q)的完整有序名单，取前floor(q/2)为硬锚点A；重现原K-Center有序S0。按Random锚点原顺序将A中缺失成员加入，只删当前S中非A成员；删后全候选最大最近距离最小，再比较均值、字典序文件ID，使用参考代码原1e-12并列规则。恰好|A-S0|次交换，q不变、锚点全包含，无合成或重复成员。无几何硬底线、无命令/时间/事件配额、无Loss梯度评分。

原视觉token1/2×8步与末端7维专家动作距离、0.5/0.5权重、全任务动作尺度、4096标定对、256参考、FP64、标定seed42保持。完整身份绑定只用于校验与描述性统计，不用于RA决策。真实入库结果必须等于离线有序名单；载荷逐字节复制，旧任务不重选。Task9自己的入库不影响已完成模型。

## 首轮预算与输入

GOAL seed42，官方六任务goal_multitask_model_ep96.pth，state=True，父原stateTrue缓存只读，各任务10演示，Task6→9各100完整epoch、batch10，共400epoch/76100optimizer steps。N=997/889/904/1969；q=499/445/452/985；锚点249/222/226/492。原BC+.1meanIFA、gamma.7、AdamW1e-4、warmup500逐step线性、阶段重建optimizer/scheduler、网络连续继承、空buffer起步。实际IFA依原算法对RA记忆计算，完整报告与父差异。4个ep99最终模型，不按loss挑模型或早停。

评估只新RA：原初态0–19，每有效cell20，10lane×2，env42/43，600步上限、5零动作、global RNG每stage42、原GMM动作分布，共200条20vector attempts。正常失败不重试；技术异常保留全部attempt，不能保证原全局RNG和环境顺序时BLOCKED。

W=/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed42_20261018；O=/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed42_20261018。沿用已有GOAL新产物系统盘授权。父P=/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed42_20261017。旧SC/OBJECT/其他GOAL资产只读，禁止覆盖或重新贴合同。无新付费资源。

## 测试与验收

8项包内合成CPU测试仅软件证据；真实4759窗口和原C/Random名单、所有原始输入/源/config/modal哈希核验。独立计算每次所有合法删除对象的最近距离，验证最小损害决策，不只检查输出shape。完整保护选择器全局RNG。四任务实际storage集成选择与离线一致。恰好4次隔离GPU原BC+IFA优化器调用及完整epoch状态恢复、0环境/0策略rollout，正式重新seed和官方权重起步，丢弃预检状态。

首次入库前Task6实际模型tensor与父完全一致，初始化全RNG相同。完成需400epoch76100步、4CP严格加载、200条20attempt、10有效矩阵单元、全部精确AUC/FWT/NBT/OldFinal/FinalAll、入库配额/载荷/锚点/最少交换/离线身份、原输入哈希、退出0无残留和独立验收通过。训练完不代表完成。

## 推进规则与恢复

保留RA附件更严格的三条件门槛：AUC>原Coreset63/80；FWT>=13/16；OldFinal>=41/60；全部工程验收PASS。Random原AUC259/320同时报告，不作为必要门槛。按附件no_auto_extend，首轮结束先报告实际门槛和跨套件后续资格，不能因CPU覆盖指标启动下一组。若未来执行OBJECT或扩seed，必须独立套件/seed合同，不将GOAL配置移植为OBJECT。

工程恢复先确认进程退出与合同一致、留存原失败；从本arm完整epoch恢复model/optimizer/scheduler/grads/modes/所有RNG/cursor，按预提交清单幂等入库，不重抽/重复计步。不能盲重启部分评估、修改冻结代码或重贴合同。只读汇总故障可外置修复并留证，不重跑环境。

每轮依AGENTS向MLR发布代码、报告、轻量证据，禁止权重/latent/buffer/演示/原绑定动作/凭据。正常运行保存进度和实测ETA即结束不持续等待。单seed探索性；官方pretrain模态元数据unknown、父调度/ep99补全、全局动作RNG、顺序放大与IFA间接变化均披露。负结果正常验收停止。
