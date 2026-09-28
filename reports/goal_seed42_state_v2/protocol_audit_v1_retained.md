# GOAL / seed 42 协议审计

状态：BLOCKED。用户已明确选择“暂不决定，完成审计与测试后保持 BLOCKED”。不启动正式特征导出、400 epoch 训练或200次评估。本文件中的采用值，除隔离测试外，均为预注册待执行值。

实验标签：goal_seed42_mlr_ifa_paper_aligned_v1。固定来源 https://github.com/yfqi/lifelong_mlr_ifa ，commit a721807f2f7fa7f9833dcc85c592d78e9accd00b。下列源码行号均指未修改的该 commit；原文件存于 source_audit.tar.gz，修改另存 source.patch。

## 工作区保护

本地 D:/论文课题研究/MLR+IFA 不是 Git checkout；现有 deployment 文件哈希存 local_workspace_hashes.json。服务器原仓库 /data/mlr_ifa/src 的 status、staged/unstaged diff、untracked 清单、commit 已保存在本次 evidence 目录。检查时原仓库干净。另行本地克隆至本次 source 目录并 detached checkout 固定 commit；未 reset 原仓库，未覆盖原 buffer 或实验。全部新远程产物位于 /data 数据盘。未推送、未上传权重/演示、未创建付费资源。

## 必须解决的新科学协议冲突

libero/configs/policy/complete_policy.yaml:18–20 为 use_vision=True、use_language=True、use_state=False。save_feature_goal.sbatch 和 test.sbatch 均未覆盖 use_state。

complete_policy.py:183–202 虽计算9维关节/夹爪状态，209–211 的 False 分支却使用最后一个视觉 v_feat，通过 state_scale/state_shift 再生成第三个视觉 token。实际4个 token 为语言、agent 相机、手眼相机、额外手眼视觉；机器人状态 token 未进入 temporal decoder。这不是关闭数据读取，也不是简单的命名差异。

用户论文第4页描述多模态潜表示包含机器人状态；另一个公开版本 arXiv v1 的附录 A.1 也描述空间前端和时间解码结构。官方 GOAL 权重含完整 state_encoder、state_scale、state_shift 张量，cfg 的 policy_type=TailPolicy，但 cfg 无 use_state/use_language 字段。参数存在和严格 shape 兼容均不能证明预训练采用哪个分支。

因此无法把默认 False 无声认定为原方法，也不能未经决定改 True 后声称恢复作者轨迹。此项影响导出特征、训练与闭环动作。用户已决定暂缓选择。smoke 仅按源码 False 验证接口，同时扰动真实样本状态以验证该分支是否确实忽略状态。该测试不用于比较两个科学方案。

## 参数与实现对照

| 项目 | 论文 / 发布源码证据 | 本轮处理 | 轨迹影响 |
|---|---|---|---|
| 来源、预训练 | README:20–39，预训练流程未发布，官方 Drive 权重入口 | 仅官方 GOAL ep96；不重建预训练 | 保留来源；预训练完整轨迹未知 |
| seed | 两个 GOAL sbatch:13；论文第6页仅称3 seeds | 训练42来自脚本，不宣称正文列42 | 固定随机数 |
| 官方权重 seed | 实际 checkpoint.cfg.seed=42，benchmark_name=LIBERO_GOAL，experiment_dir 含 GOAL | 原样存 pretrained_manifest；这只是元数据，不独立证明原预训练轨迹 | 无 |
| 任务顺序 | benchmark/__init__.py:83–84、204–207；论文第5页6旧+4新 | order0；前6预训练，增量6–9；名称落 data_manifest | 无 |
| 演示 | 论文第5页10条；data/default.yaml:6；datasets.py:63–65 | 前10完整demo，无8/2划分；检查索引、cache、窗口、最后动作padding | 清除不可访问旧cache不改有效样本 |
| 时序 | data/default.yaml:5；complete_policy.py:222–228 | 8步×4模态×512；仍有未决定状态分支 | 除状态待决外不改 |
| epoch | 论文第6页100；train/adaption.yaml:2；mlr.py:711 | 四阶段各100完整数据遍历，0–99 | 不改为固定steps |
| batch | 论文第6页10；train/adaption.yaml:3 默认18 | 显式10；单GPU/world_size1；导出batch1单独记录 | 是，论文参数对齐 |
| AdamW | 论文lr1e-4；optimizer/adam_w.yaml:1–6 | lr1e-4、betas(.9,.999)、eps1e-8、wd.1 | 沿用 |
| clip、loss scale | train/adaption.yaml:5–6 | 100、1 | 沿用 |
| LR | scheduler/cosine_annealing.yaml:2–6 实为LambdaLR；utils.py:236–244；base.py:109–124；mlr.py:754–755 | 500优化器步warmup；每次optimizer.step后仅一次scheduler.step；100×len(loader)衰减至0；完整序列存csv | source_gap + optimization_affecting_patch；不声称作者原曲线 |
| 冻结、结构 | 论文第4页；complete_policy.py:109–127；brief.py:25–43 | 空间前端含LoRA冻结；6层GPT2+5模式GMM可更新；保留官方权重 | 状态token选择未决 |
| BC | base_policy.py:153–155；policy_head.py:114–123 | 混合batch原GMM NLL，监督最后一步动作 | 不变 |
| IFA权重 | 论文第6页.1；base.py:268 | 常数.1，不使用未调用的 hinge_weight_sigmoid | 不变 |
| GOAL gamma | 论文第6页.7；adaptation_goal.sbatch:22 为.72 | 显式.7 | 是，论文参数对齐 |
| IFA任务对 | 论文式(4)；mlr.py:466–658 | 语言与agent全部任务对各ceil(top50%)交集，再筛当前任务；仅当前任务全缓存及旧buffer统计；无未来任务 | 保留选择 |
| IFA正参考 | base.py:172–186 取混合batch首例 | features[current_mask,0,0,:]逐当前样本自己的语言参考 | 是，修复已授权bug |
| IFA负参考 | base.py:221–222 旧任务0.pt语言 | 保留同一reference，仅内存cache减少IO；不把旧完整特征库加入BC | 不改参考选择 |
| IFA平均 | 论文第4页式(4)；base.py:216–268 每任务对直接相加 | 对当前样本和参与任务对两级平均，无pair/无当前样本为0 | 是，公式对齐 |
| 相似度 | mlr.py:334–368 构造两两大矩阵 | 归一化向量均值内积的恒等式，FP32；保留torch.topk和源顺序 | 数值舍入有微小差异，近tie可能改变选择；不抽样 |
| 存储 | 论文第7页.5；mlr.py:28、187–213；sbatch:25 | 原随机窗口ceil(.5N)，seed42，每task独立rng；59660000为文件容量 | 保留行为，不是5条完整demo |
| 回放batch | mlr.py:128–140、685–703 | 当前所有窗口+旧buffer合并，原shuffle和DistributedSampler，epoch set_epoch | 不加30%或50%batch配额 |
| Loss日志 | mlr.py:717–723 循环内部反复除以len(loader) | 修正到epoch循环后；observe提供BC/rawIFA/weightedIFA/total/LR/grad记录 | 日志本身不改变优化；不作选样/早停 |
| 最终模型 | mlr.py:731–740 保存70–95 | 固定epoch99最终模型，不挑best | checkpoint规则为本轮补充 |
| 权重前缀 | utils.py:59–64 无前缀无return；test.py:231 无条件k[7:] | 接受全有/全无前缀，混合前缀拒绝 | 避免错误加载 |
| 权重覆盖 | brief.py:43、test.py:233 strict=False | 官方full strict=True；下游键集合/shape严格匹配后覆盖，报告coverage；缺失拒绝 | 不留随机可训练张量 |
| 导出 | CompletePolicy.forward:133–136 返回空间潜特征；save_feature.py:171–187；base.py:303–308 | 保留forward；训练模式导出+原增广；无优化器更新；正式未导出 | 不静默改变train/eval |
| 闭环 | metric.py:84–175 | 原8步滑窗、5零动作、600策略步；每单元20=2×10；首次20不同官方状态 | 本轮明确约定 |
| 动作随机性 | complete_policy.py:230–253；GMMHead:103–110 | 选logit最高高斯组件后按其std采样；不是改为确定性均值，也不是重新采样混合组件 | 保留发布实现 |
| 成功定义 | bddl_base_domain.py:800–808 done覆盖为_check_success；env_wrapper.py:87–88、103–104 | 环境任务成功即done；600步超时失败；smoke逐步核对done==check_success | 无 |
| 评估RNG | 源metric.py:111–115；执行文档第7节 | 单元重置42；批次42/43，workers按venv.seed的seed+i；正式未运行 | 本轮补充约定 |
| 评估控制 | test.py:215、223–224 | 原busy-wait改明确FileNotFoundError；主CLI在BLOCKED下拒绝正式评估 | 工程控制 |
| 指标 | 论文第6页；执行文档第8节 | 对角FWT、按列分别平均NBT/AUC；FinalAvgSR仅辅助；上三角NaN | 单元测试用合成矩阵，绝非实验结果 |

## 实际软件环境与资源

复用已安装且版本与requirements相符的 /data/mlr_ifa/env（Python3.8），不升级依赖。GPU、驱动、CUDA、MuJoCo、robosuite、robomimic、transformers、peft实际值在 environment.json，完整依赖在 requirements.lock.txt。

官方 GOAL 权重 SHA256：241d3370acfc04d694df04256956a97aa13cfdc2e651cf258d338f8e72988be7。Drive下载入口由固定README核对；来源安装记录及本轮checkpoint.cfg共同留档。CLIP缓存来自 openai/clip-vit-base-patch16，revision 57c216476eefef5ab752ec549e440a49ae4ae5f3；本轮工具设置offline，避免动态下载改变来源。

数据 /data/mlr_ifa/datasets/libero_goal；BDDL和初态 /data/mlr_ifa/src/libero/libero/{bddl_files,init_files}/libero_goal。所有10任务文件哈希、名称、状态文件哈希和前20初态哈希入data_manifest；增量4任务验证前10演示的完整有效索引。data.train_dataset_ratio=.8 是未被本路径使用的遗留配置，未调用 get_train_test_loader。

## 测试范围与仍未放行事项

隔离测试包含真实窗口、CUDA/NCCL、两次优化器更新、冻结前端哈希、混合batch的IFA回归、相似度恒等式、严格权重覆盖/缺失拒绝、相同输入前向一致性、模型+优化器+调度+RNG重载后下一更新一致性、一个完整真实闭环episode、指标定义及不完整结果拒绝。运行结果以smoke/*.json和日志为准，不把“写好了测试”记为通过。

此时提供的full/resume/eval接口是明确阻断入口，不是假装已经可运行的正式完整编排。正式特征溯源manifest、事务式幂等buffer与正式链的集成、四阶段恢复链、逐epoch实际新旧样本计数、10个单元的独立正式评估及重试规则尚未执行验收。用户决定状态模态后仍须补全并验收这些运行控制，再用全新RUN_DIR重跑smoke、从官方权重和42开始。

隔离恢复测试只证明保存边界后下一个优化器更新一致，不宣称完整DataLoader/buffer阶段恢复已验证，更不宣称batch中途精确恢复。发布版其他不可达旧函数、其他套件、其他算法不在本轮执行路径。

## 来源边界

S1 用户提供的CVPR论文PDF：第4页式(4)/冻结，第5页任务协议，第6页Setup/指标，第7页存储。S2 固定commit真实源码及实际权重cfg。S3 https://arxiv.org/html/2603.10929v1 是另一个明确版本，仅补充结构与系数核查，未代替S1。500 warmup、checkpoint规则、RNG隔离和恢复保证等均未写成论文原话。由于状态模态未决及公开细节缺口，目前不存在可以声称“精确复现作者轨迹”的正式结果。

补充：事务式buffer helper已实现，使用原随机选择函数，在独立smoke目录检查同任务重复调用不追加、源buffer不变、seed42选择一致；不把该helper单元测试当作四阶段恢复系统已验收。首次测试发现内存tuple与JSON list返回类型不同，已统一返回已提交manifest的JSON形式，保留失败日志与目录。
