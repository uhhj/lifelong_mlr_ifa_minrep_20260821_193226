# GOAL seed42 state_v2 协议审计

本轮按用户明确决定启用状态模态：policy.use_vision/use_language/use_state 三项均为布尔True。运行标签 goal_seed42_mlr_ifa_paper_aligned_state_v2。固定仓库 yfqi/lifelong_mlr_ifa，commit a721807f2f7fa7f9833dcc85c592d78e9accd00b。此文件取代v1阻断决定；v1完整审计另存 protocol_audit_v1_retained.md，不回滚旧修改和旧结果。

“按论文描述启用状态模态；官方checkpoint原始模态配置若缺失，仍记为未知。” 实际检查完整checkpoint.cfg.policy，use_state缺失，记 pretrain_use_state=unknown、effective_use_state=true、DISCLOSED_GAP。官方cfg的seed=42仅为实际元数据；论文正文未列出42。未修改官方checkpoint，未把当前True写为作者设置。公开仓库没有完整预训练流程；六任务来源按论文和官方权重入口解释，不能凭cfg独立还原未公开的任务选择和预训练轨迹。

## 证据与参数

固定源码行号及论文对照保留在v1审计的参数表。用户论文第4页式(4)/冻结和状态模态、第5页6+4任务/10演示、第6页100epoch/batch10/lr1e-4/linear scheduler/IFA.1/GOAL.7/三指标、第7页存储比例为主要证据。arXiv v1附录A.1/B.1仅为明确标注的补充版本，不替代附件论文。

| 项目 | 发布代码 | 本轮实际决定 / 影响 |
|---|---|---|
| 模态 | complete_policy.yaml:18–20 为True/True/False；complete_policy.py:209–213有两条不同4-token分支 | 三个入口共享++True override；使用现有state_mod分支。影响特征与优化轨迹，用户v2明确授权 |
| 状态 | complete_policy.py:185附近joint_states再gripper_states | 保持7+2维，language/agentview/eye_in_hand/state，4×512，不改模型 |
| 冻结 | CompletePolicy先冻结空间前端再建decoder/head；BriefPolicy读取latent | CLIP、已有LoRA、projection、state encoder、FiLM均冻结，仅6层decoder与GMM更新 |
| 来源 | README:33未发布预训练流程，:39官方Drive | 同一官方GOAL ep96；完整615键strict加载；状态编码/调制12个参数形状核对；下游100键覆盖必须完整 |
| seed / 任务 | GOAL sbatch seed42，start6/end10；benchmark order0 | 单训练seed42，LIBERO-GOAL任务6→9；不扩大实验组 |
| 演示/窗口 | datasets.py取前10；data.seq_len8 | 每任务demo_0..demo_9，无8/2拆分。索引、cache、padding真实验证 |
| batch / eval | 默认18 / 2 | 训练显式10，导出1，eval.n_eval显式20。属于论文参数对齐 |
| gamma/IFA | 启动脚本.72、loss系数.1 | .7/.1常数；当前样本各自语言正参考；按样本、任务对平均；无pair为0。优化路径修复已披露 |
| IFA选择 | 语言、agent-view top ceil(50%)交集，再筛当前任务对 | 保持；旧任务统计只读buffer，当前读全缓存，无未来特征学习；旧0.pt语言负参考沿用原代码 |
| 相似度 | 大型两两内积矩阵均值 | 换用归一化均值内积的恒等式，FP32；保留torch.topk近tie行为。数值舍入差异已披露 |
| 存储 | past_data_rate=.5，capacity59660000个文件 | 原seed42随机窗口ceil(.5N)；不是batch旧样本50%，不是五条完整演示；不设置30%配额 |
| 训练遍历 | 当前所有窗口+旧buffer，预先shuffle，DistributedSampler.set_epoch | 每阶段100完整遍历，0..99；新旧实际次数、重复数和步数逐epoch记录 |
| 优化器 | AdamW配置lr1e-4、(.9,.999)、eps1e-8、wd.1，clip100 | 保持；每阶段重建optimizer/scheduler但不重置网络 |
| LR | 文件名cosine实为LambdaLR，warmup500；训练错误地epoch推进 | optimizer.step后一次scheduler.step；总步数100×len(loader)，warmup→峰值1e-4→0；完整曲线存CSV。source_gap和optimization_affecting_patch，不声称作者轨迹 |
| 读取进程 | train.num_workers=12，persistent_workers=False | 导出12；纯缓存训练0，避免每epoch重复spawn。无随机变换、样本顺序/父进程torch RNG对照测试，不改sampler和batch规则 |
| 日志与最终模型 | Loss在batch循环内反复除长度；只存70–95 | 修正日志；固定99最终模型，滚动epoch恢复点；不以Loss或闭环挑模型 |
| 闭环 | metric.py:84–175，每批10、5零动作、8步历史、600策略步 | 保持2批×10；仅四阶段已见增量任务，200条。每单元RNG42；batch42/43，worker=batch_seed+id |
| 动作 | GMMHead选最高logit组件的Normal并采样 | 保持随机采样，不改为均值；一致性测试比较means/scales/logits |
| 成功 | bddl_base_domain.py:800–808将done设为_check_success | 超时不是成功；基础设施异常另存并中止该单元；resume只能同单元、同初态重跑 |
| 指标 | 用户论文与Runbook第8节 | FWT对角均值；NBT/AUC先按任务列平均；负NBT不截断；FinalAvgSR仅辅助 |

## 真实路径与隔离

旧原仓库 /data/mlr_ifa/src 保持；v1工作区和产物保留。v2工作区在 /data/mlr_ifa/experiments/goal_seed42_mlr_ifa_paper_aligned_state_v2/source；先保存原status/diff/untracked代码归档，再在隔离副本接续补丁。无reset、远程push、权重/数据上传或付费资源启动。

官方权重 /data/mlr_ifa/checkpoints/goal_multitask_model_ep96.pth，SHA256=241d3370acfc04d694df04256956a97aa13cfdc2e651cf258d338f8e72988be7。GOAL演示/data/mlr_ifa/datasets/libero_goal；BDDL/init在/data/mlr_ifa/src/libero/libero/{bddl_files,init_files}。环境/data/mlr_ifa/env，CLIP缓存revision57c216476eefef5ab752ec549e440a49ae4ae5f3，本轮离线读取。

正式新缓存为本轮run/features，初始run/buffer为空；隔离smoke不进入正式buffer/模型链。旧False或未知模态缓存不能补标签复用。modal_contract.json包含前端与官方权重哈希、状态维度/顺序、token顺序、时窗、导出模式/增广、固定commit和source.patch哈希。feature/buffer/checkpoint均携带完整合同及同一fingerprint，读取前检查内容及fingerprint。合同未知或不符拒绝，源码快照变化也不能覆盖既有运行。

## 实际入口、恢复与验收

scripts/run_goal_seed42.sh调用goal_v2_driver；由其把同一MODALITY_OVERRIDES拼入原save_feature.py、lifelong_adaptation.py、test.py的实际Hydra命令。前两入口仍单GPUtorch.distributed/NCCL。实际resolved_export/adaptation/eval分别落盘；BriefPolicy字段仅声明，不重编码旧latent。

模型主体CompletePolicy未改：forward继续导出空间latent，闭环继续get_action。runbook_v2.py包装原Basic.observe导出、MLR.observe训练、原数据集/采样器/相似度/随机buffer选择以及完整策略闭环。新增事务buffer generation和完成manifest；重复stage提交校验后返回已有generation，不追加；原buffer不改写。

每epoch恢复点保存网络、AdamW、scheduler、全部RNG、sampler epoch/seed、任务/步数、父checkpoint hash、配置、feature/buffer manifest和合同。每阶段保留epoch99最终恢复点。保证epoch边界恢复；不声称batch中途或掉电精确恢复。阶段最终权重严格重载检查后登记manifest。

评估在全部训练完成后单独进程执行。每单元第一批真实在线观测再次验证状态token与两条策略路径分布一致，保存并恢复RNG，避免影响正式动作随机数。每20条构成原子cell结果；发生基础设施错误则保留异常journal并停止，恢复时从同一单元RNG42和原20初态重跑，权威rollouts只汇总完成cell，未完成不生成完整指标。

测试是否通过及正式进度只以对应JSON/日志为准。合成矩阵和False分支仅软件负对照，不属于额外正式组。模态语义测试通过不证明官方状态编码器当初得到有效预训练，也不证明本轮闭环收益。实验完成仍须400epochs、4个正确最终checkpoint、200正式记录和10个有效矩阵单元；仅训练完成状态为TRAINING_COMPLETED_EVAL_PENDING。

工程修复记录：首次正式导出尚未产生窗口时，12个spawn读取器因get_dataset重新打开的h5py句柄不可pickle而失败（TypeError）。在get_dataset返回前关闭句柄，由每个worker按原属性机制按需重开；不改变demo、索引、缓存数值、采样或增广。补做真实HDF5的12-worker两窗口读取测试；失败目录完整保留，不把该失败算作正式训练结果。

补充读取器修复：spawn子进程不会继承父进程的robomimic ObsUtils模态映射。测试实际暴露OBS_KEYS_TO_MODALITIES=None；新增worker_init_fn，在每个worker中按同一obs modality配置初始化映射。保持已有worker随机seed、不添加随机变换。原失败栈及已通过的模型/闭环子测试均保留。
