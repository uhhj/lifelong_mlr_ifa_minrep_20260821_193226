# GOAL Random MLR+IFA vs Coreset MLR+IFA，seeds42/43/44

用户于2026-10-17授权GOAL两方法比较，明确三seed42/43/44，并允许所有新产物放系统盘。先等待现有OBJECT ZR40/41队列退出且成功，不占用其GPU；失败则等待处理，不绕过前置队列。旧实验全部只读，不push、不新建付费资源。

## 科学合同

固定原始公开源码a721807f2f7fa7f9833dcc85c592d78e9accd00b，沿用已验收OBJECT Coreset比较的共同工程补全，仅更换套件、官方GOAL权重、gamma、seed与路径。论文正文明确三随机seed但未给数值；42来自官方脚本，43/44为用户指定。不是作者原始轨迹精确恢复。

官方GOAL六任务预训练权重goal_multitask_model_ep96.pth；不重建预训练。每seed重新导出GOAL任务6–9的state=True特征，同seed两组共用。从相同官方权重、独立空buffer及重置训练RNG开始，两组分别连续训练四任务，每任务10演示、100完整epoch、batch10、AdamW lr1e-4、BC+0.1平均IFA、GOAL gamma=0.7。逐step线性调度、warmup500，任务边界重建optimizer/scheduler，模型跨任务继承；这些调度细节与ep99固定取模规则为披露的实现补全，非正文精确指定。

三入口resolved config的vision/language/state必须为布尔True。语言、agentview、eye_in_hand、state四token；joint7后gripper2；window8/embedding512。冻结空间前端含预训练LoRA。官方cfg.policy缺use_state时记录unknown/DISCLOSED_GAP，不伪造作者设置；真实状态参数缺失或不完整加载则STOP。

唯一组间变量为当前任务结束后入库成员。K=ceil(0.5N)，0.5是存储比例，非batch旧样本比例。保留原拼接、shuffle、DistributedSampler默认seed0和set_epoch。Random保持原独立Random(training_seed)抽样；Coreset保持已冻结visual/action Greedy K-Center：8步×两视觉token平均cos距离与末步7维专家动作距离各0.5；动作MAD/IQR/range/1稳健缩放；4096不重复无序标定对、正距离中位数尺度；最多256参考中心；独立选择器seed42；FP64、块256、字典序平局1e-12。只拷贝原payload，不加入语言/state距离、事件、残差、Loss选样或动态配额。旧任务只能使用实际buffer。IFA按各自实际buffer计算，可能间接变化，必须记录。

训练、导出、数据集、Random入库、评估global RNG为对应seed。Coreset标定42不随训练seed改；同seed两组初始化实际tensor/RNG以及首次入库前Task6 ep99 tensor要求完全一致。不得混用旧state=False或来源不明缓存。

## 评估与验收

每组四个epoch_index99最终checkpoint，独立进程评估各阶段已见任务，10cell×20=200条；三seed两组共1200条。使用原初态0–19、10lane×2、env seed[s,s+1]、5零动作稳定步、600步上限、window8、原argmax GMM分量后Normal采样。沿用已验收OBJECT实现：每stage重置global动作RNG，cell之间连续；无逐轨迹独立Generator，不增加重复、680口径或9000评估。正常失败不重试，技术异常单独记账。

GOAL数据预计窗口997/889/904/1969，K499/445/452/985，实际资源及导出必须验证；每任务完整epoch batch100/139/185/337，因此每arm76100step，六arm共456600step/2400epoch/24最终CP。以真实数据验证这些数，不能为凑预算增删样本。

每seed：18原软件测试加GOAL配置/预算检查，真实GPU模态语义/完整权重/GMM一致性及最多4次隔离optimizer恢复检查，一次单环境单零动作接口检查（无策略闭环）。不重复通过预检、不用CPU冒充GPU、不接续smoke状态。

所有原始输入、代码、三入口配置、modal、特征、buffer及CP合同可追踪；实际episode和attempt账目齐全；每seed8CP实际严格加载、800epoch/152200step/400record、Task6一致、全部矩阵/精确FWT NBT AUC通过。pipeline/supervisor及queue真实退出0、无残留和外置独立验收通过后才能COMPLETED。三seed报告均值、样本SD(ddof1)、逐seed C−R和全部任务，负结果不影响验收。

## 历史GOAL基线复用审计

旧state_v2 seed42完成是真实历史结果，但不作为本轮配对Random：其评估每cell重置RNG，本轮沿用OBJECT每stage重置；旧IFA使用样本当前语言、旧task0.pt负参考及sim_language矩阵角，本轮共同实现使用验证稳定的各任务语言参考直接计算角；两者均取pair均值，不能误称旧版未平均。旧导出worker12，本轮0。旧代码/modal合同与导出随机轨迹也不同。故新跑三seed Random，旧结果保持不变且仅作历史参考。

## 工程恢复与边界

先确认相关进程停止、保存失败证据并核对冻结输入，才从本seed/arm完整epoch checkpoint恢复模型optimizer/scheduler/grads/module modes/Python NumPy CPU CUDA RNG/cursor。存储用预提交清单幂等补齐，不重抽、重复入库或重复计步。部分评估主动拒绝盲重启，必须保证原global RNG顺序和全attempt账目，不能保证则BLOCKED。不得悄改科学代码或重贴合同。只读汇总故障可留证外置修复，不重跑环境。

披露三seed探索性、论文A100对现有3090、预训练state元数据未知、调度/最终CP补全、global动作RNG未隔离、IFA随buffer间接作用。不把描述性均值当作显著性或普遍因果。正常监控保存快照即结束；完成后下载轻量证据不下载权重数据，删除监控，不自动扩seed或启动下一轮。
