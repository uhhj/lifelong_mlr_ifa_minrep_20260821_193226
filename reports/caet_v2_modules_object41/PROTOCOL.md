# CAET-v2 独立模块 OBJECT seed41 探索性实验

2026-10-22 用户明确授权：TAC、RSET、EFTR 分别独立叠加 CAET-v2，OBJECT seed41。用户另已确认 B 不叠加 TAC。本实验不是组合 A+B+C。执行 A→B→C，固定三个入口，不依赖前一入口性能高低，不扩 seed/GOAL，不重训重评 RA 或原 CAET-v2。LIBERO50 预训练、COVR 均保持取消。

每组独立从官方 OBJECT 初始化及空 buffer 开始；原 MLR+IFA 主 BC 完整8帧/末动作，4任务各100epoch、105500主步、4个ep99主评估CP、200正式20vector10cell；RA配额781/740/780/660及有序成员、源载荷不变；BC+.1meanIFA/gamma.3/warmup500、FP32、sampler0及shuffle41不变。原20初态/10lane/env41,42/600步5zero/global-stage-RNG41/argmax分量Normal不变。原辅助.05、效果拟合10epoch（最多64条/批）、原训练与整演示留出分离；效果额外更新与候选优化器试算完整独立记账，不冒充105500主步。

以下是看过v1/v2及GOAL后的探索性设计，不宣称独立事前验证。未指定的工程/阈值选择在新图/闭环前冻结，不根据结果调优。

## A：CAET-v2+TAC

保留原BC，辅助使用原缓存前7个真实观测obs[u-7:u-1]，与a[u]及states[u]→states[u+1]物体平移/SO3/目标进展绑定。拒绝歧义、padding及不存在转换，主BC样本不删。相对位置和夹爪来自obs[u-1]与state[u]。原效果组/距离/互近邻/跨demo支持规则不改。效果模型允许7步均值聚合，不引入B的注意力。旧v2功能保护也以相同7步和a[u]计分，避免保留下一动作监督矛盾；其历史逐组最优界限和双对偶规则不变。

先以真实缓存核时序/物理效果/原RA成员/float32 sidecar图及实际batch覆盖。隔离GPU检查原生关闭更新完全一致、全状态恢复和原功能测试；另在固定合法样本上报告7vs8输入的执行分布KL、分量切换率、动作均值和夹爪差异及7步梯度有限性。此分布审计不将差异默认为0，也不以无依据阈值宣称等价；部署仍8步，差异是需闭环检验的科学限制。

## B：CAET-v2+RSET（独立于A）

保留v2合法因果next-action标签obs[u]→a[u+1]及8步输入，不混入TAC。效果模型将均值聚合替换为每帧1536→128投影、固定正弦位置编码、单层4头时序自注意力（FF256、dropout0）、注意力池化，再结合7维动作MLP输出7维效果；不改主策略。

动作支持仅使用合法当前训练窗口及实际RA旧成员，排除效果留出。状态描述为最后时刻非语言三token向量归一化；余弦距离选最多20近邻，要求cos>=.90；动作按训练动作逐维总体std归一化（floor.05），与候选动作的RMS距离<=.50；至少3个独立(task,demo)邻居满足。训练数据自查询排除自身及同demo，防止自证。全局跨任务支持仍须原图PASS。零训练报告专家动作下潜在剩余支持；未知的未来策略动作支持和拟合可靠性必须标待实测，不能冒称零训练可证明。

可靠性分层按已冻结物理组归并成STATIC/TRANSLATION/ROTATION与open/closed，不使用肉眼臆测接触标签。每层合法整演示留出>=12窗口且>=2个(task,demo)，MAE须优于训练均值和该层固定循环动作置换，动作梯度有限非零，才开启该层。全局原留出门仍保留。实际batch权重=cross-task×action×reliability；无可信样本则迁移0，BC+IFA照常，原v2保护保持。报告每层与三个筛选环节剩余量、demo覆盖及拒绝原因，不因稀疏放宽门槛。

## C：CAET-v2+EFTR（独立于A、B）

v2效果标签/均值预测器/支持门/迁移.05保留；替换next-action NLL双对偶保护为上一阶段单一完整策略的实际执行分布KL，输入为原部署8步。在所有旧RA样本上适用，不要求该样本效果标签有效；t>=7可独立启用保护，即使迁移因原科学门回退。参考是本run前一阶段ep99，不取逐组历史最优，不用未来模型。

qexec为原最大logit分量的对角Normal；KL方向reference→candidate，按动作维求和后样本平均，保护系数.1。监测模式索引切换、最后夹爪维均值变化；KL不宣称控制argmax不可微边界。

每batch从完全相同模型/optimizer/scheduler/梯度/modes/allRNG/step状态试算native和native+.05effect+.1KL两个AdamW候选。两支native/dropout与辅助RNG分离；最终只提交一个主更新，候选试算另计。固定当前batch上eval模式原BC NLL candidate<=native+.01nats/sample；同一旧batch reference→candidate KL<=.05nats/sample；相对reference分量切换比例<=.10，夹爪均值绝对差均值<=.05（原动作单位）。current或old batch为空的判据记NA，而非造0；旧为空则无KL检查。任一失败则恢复并提交完整native候选，包括optimizer/moments/scheduler/RNG/梯度，不做再加权或重试。原始梯度点积另记诊断但C不以它代替候选门。每完整epoch在全部旧RA成员报告KL/模式切换/夹爪偏差。局部batch约束不保证闭环或全记忆硬约束，native回退自身也可能超预算，须如实报告。

准入必须真实GPU：关闭完全一致、候选重放/全状态恢复一致、构造劣化候选被拒绝、零附加候选等于native且可接受、接受/拒绝均不多计scheduler/主步。全部试算计账。冻结阈值不可后改。原小型功能通过不当闭环成功。

## 执行与验收

新Q=/data/mlr_ifa/experiment_workspaces/CAETv2_TAC_RSET_EFTR_OBJECT41_20261022，各variant独立源码、合同、输出。冻结后不覆盖源码/失败码。正式前须零训练审计、真实隔离GPU准入；不适用阶段科学回退，整实验无有效机制不得冒称方法有效。工程失败先查停进程与hash，保留失败后只从完整epoch全状态版本化恢复；部分正式globalRNG评估不盲重启。数据盘仅使用既有root空闲且留2GiB，系统仅普通空闲；不删旧资产、不新付费资源。

最终所有退出、主步、4CP严格加载、原成员载荷/IFA/初始化Task6张量、机制/候选账目及200正式/20attempt、完整矩阵均验收后本地独立复算AUC/FWT/NBT/OldFinal/FinalAll，与同seed RA及原v2分别比较。三方法单seed不做显著性或普适结论。只下载/发布代码、报告、哈希、轻量逐episode结果，不发布原动作状态latentbuffer权重RNG二进制。每轮按AGENTS经publish_mlr_branch推MLR并核SHA。长期运行由本任务监控，未完PARTIAL。
