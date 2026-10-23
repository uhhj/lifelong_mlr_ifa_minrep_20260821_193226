# CAET-EM-v1 — OBJECT seed41

本协议与逐字保留的 USER_METHOD.txt 一同冻结。新的可执行效果路线由用户明确授权，不沿用物理位置探针的 10mm/25mm 准入门。原失败及取消任务保持原结论，不重启。此版本只运行 B0 READOUT_BC4、B1 EM、B2 EM_TF 各 Task6→9；不自动扩 seed，不扫描权重或其他模块。

## 固定比较与预算

三组共享官方初始化 Temporal、原训练人口/RA-Coreset 成员、8×4×512 原生输入、四步合法未来动作、GMM mixture NLL、最新帧 modality3 控制读出及 modality0 IFA 读出。部署保持原 32 token 上限，观察历史长度1–8由原 evaluator 唯一管理；每次仅采样最高 logit 分量的第一个未来动作，执行后重新观测。不会采样被丢弃的其余三步以免额外推进部署 RNG。

每组 Task6/7/8/9 人口1562/2261/3081/3620，每轮157/227/309/362个batch10，100epoch/阶段，400epoch、105500主更新；三组共1200epoch、316500主更新。原RA成员781/740/780/660按已验证的原选取清单原字节复制，不重新选择。每组四个ep99策略CP，10三角cell×20初态=200正式episode，三组600；10环境、seed41/42、初态0–19、5个zero warmup、600policy steps。正式评估不重跑正常失败rollout，不盲重启部分globalRNG评估。

## 训练前固定的实现补充

- 原始时间标识绑定窗口u，动作 a[u+1+k]，实际观测机器人响应 r_obs[u+1+k]−r_obs[u+k]，视觉差 Phi[u+1+k]−Phi[u+k]；k=0..3。来源身份不唯一或历史填充时掩码全false；尾部逐步保留独立 action/response/visual mask，不删DataLoader成员、不重复末动作。第一步BC合法性精确沿用ALIGN身份规则。
- Phi 为官方冻结空间视觉编码器经官方768→512 image_proj 后、语言FiLM前的每相机512维特征；完全使用部署eval预处理、无增广。不能差分原随机增强feature cache。PCA每相机32维，仅Task6 identity排序前8演示的全部当前帧拟合；double SVD，最大绝对载荷坐标符号固定，最终投影FP32。后两演示验证及Task7–9均不得拟合投影。当前观测支持向量用两个pre-FiLM Phi和9维实际机器人观测，不含未来标签或专家动作。
- Tokenizer输入三支线性7/9/64→64，逐分支mask后加模态embedding，64→256；每个未来位置用一个learned query作1层4头cross attention并LayerNorm，随后4步位置embedding和2层Transformer(d256,h4,FFN512,dropout0)，线性→4×64。GR/GV分别64→128→9/64 ReLU。全部证据分支的均值/总体标准差仅Task6训练8演示合法样本拟合，std下限1e-4，冻结。动作监督仍原动作单位，不改变GMM输出尺度。
- Context为最近帧原生camera1/camera2/state三token拼接1536→LayerNorm→256 ReLU→128。Decoder输入E64+context128+learned future position16，208→512 ReLU→512→原GMMHead；原GMM权重在Stage I前严格来自官方。没有h_control到decoder的旁路。
- Stage I训练 tokenizer/GR/GV/context/decoder/GMM，AdamW3e-4,wd.01,batch64,FP32,clip100,100epoch，不设额外学习率调度。第一8演示train、末2validate，按完整验证合法token聚合的 NLL+.1按位置/旋转/夹爪三组均值Huber+.2标准化R Huber+.2标准化V Huber 选最早严格最小值。训练损失不作性能门。effect标准化统计来自选中codec对训练8演示完整合法effect位置的输出，std下限1e-4。冻结全部codec、GR/GV、统计与投影。B1/B2共用该初始decoder，后续独立学习。
- B0为 h_control512→512 ReLU→2048 reshape4×512→原官方GMMHead。B0全部可训练参数lr1e-4。B1/B2 predictor512→512 ReLU→256，Temporal/predictor lr1e-4；Stage I初始化的全部decoder/context/GMM lr2.5e-5。主AdamW betas(.9,.999),eps1e-8,wd.1，clip100，原每阶段500warmup、原线性schedule逐主步更新，IFA .1 / gamma .3，所有参数有明确optimizer白名单。额外初始化使用独立RNG流；原Temporal和原初始GMM tensor逐键严格官方加载。
- 主BC始终解码预测effect，不输入真值effect。LE=maskedHuber(Zhat,Z*)；effect合法mask为三个分支mask交集。B1=BC+.1IFA+.1LE。B2再加.02LT+.05(LF+.1Lroute)。T/F在Task6严格关闭，B1/B2Task6必须完全一致。IFA计算保持项目原selected_pairs与语言参照，不改RA人口或数据顺序。
- B2教师是自身上一阶段单一完整策略冻结eval/no_grad。教师缓存固定整阶段，绑定前一CP、codec、dataset身份和合同SHA。支持特征用旧RA均值/总体std标准化，各camera512/state9分组L2归一化等权1/3；欧氏距离。半径为旧RA每成员跨(task,demo)最近邻距离Q95；当前半径内必须至少3个不同旧(task,demo)。tauA=max(旧RA有效样本分组Huber Q90,.02)，tauF=max(旧RA标准化R/V重建均值Q90,.1)。rA/rF各在各自有效未来位置均值；样本至少一个三分支同时合法未来位置才可入T。T仅当前样本，逐有效未来位置平均；零支持为可微零，不根据闭环调整阈值。
- F只使用当前原DataLoader batch内合法旧RA样本，不另增采样人口。对固定真值effect解码和预测effect解码各取.5，比较双方各自最高logit分量的mu/sigma：.5前六维均值平方差+.5夹爪平方差；routing softmax加权stopgrad分量行为距离。对合法action位置归一化，不按混合分量编号对齐，不除方差。额外decoder/teacher/tokenizer计算独立记账，不称等算力或等总内存。
- 科学诊断按固定阶段运行：真实/预测effect动作与响应重建，action/response/visual逐支跨batch固定roll1置换、预测effect零化/roll1置换、动作偏移和argmax分量变化；只能说明接口使用与离线一致性，不当作真实闭环收益。所有负结果保留。

## 资产边界与恢复

每次证据导出只读当前任务合法10演示；同任务跨arm可只读复用同一冻结证据，后续训练访问旧任务仅原RA成员与sidecar。旧任务完整labels文件虽留存供审计，不可作为后续训练人口；代码路径强制只载current与自身old sidecars。不删除旧资产，不下载权重/特征/原action/state。

shared与B2置于/data，B0/B1置于/root普通空间。每组1.8GiB保守上界包括4主CP、完整恢复及原子临时文件、原RA字节及sidecar/teacher；shared .3GiB。实际模型字节预检，不满足就工程停止，不删资产或降预算。保留/data root f_bfree≥2GiB，system普通f_bavail≥1GiB，启动前及每阶段检查。

正式前自己的GPU隔离准入验证完整下一步恢复、最新camera/state依赖、未来标签不进部署、长度1–8、空mask可微零、mixture训练/argmaxNormal部署、T/F梯度及codec冻结。原CP只作strict_base_init；所有新部署参数以strict_policy_restore全键/shape/dtype/tensor验证，不用旧100键冒充完整加载。

每完整epoch保存model/optimizer/scheduler/grads/modes/requires_grad/allRNG/下一epoch/cursor/selectedbest/数据与teacher与codec SHA。失败保留原证据，确认进程停和hash一致后仅版本化工程恢复；无盲重启controller、无科学负结果重试。服务器科学源一旦冻结不可覆盖。

## 验收与解释

全部三组400epoch与完整200episode/20vector/10cell及自然有效退出、无残留、本地独立矩阵/账目/所有新参数严格加载、receipt后才接受。AUC/FWT/NBT/OldFinal/FinalAll完全沿用原项目公式。B1−B0检验效果接口，B2−B1检验T+F组合，逐任务完整矩阵和遗忘轨迹同时披露；旧ALIGN/RA只读外部参考。单seed探索，无显著性/无遗忘保证；不自动扩seed或启动额外消融。

每轮精确审核轻量代码/报告/hash/outcomes，经publish_mlr_branch非force推MLR并核远端receipt。运行PARTIAL，按实际阶段ETA调整同一监控；健康未变安静，阶段切换/失败/完成通知。
