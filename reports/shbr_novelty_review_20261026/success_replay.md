# SHBR查新分报告：成功筛选、自模仿与自主机器人经验回放

检索截止：2026-10-26（Asia/Shanghai）。范围：成功轨迹筛选、自主采集数据用于机器人模仿学习、成功行为回放、与持续/终身机器人学习的交叉。仅以原论文、作者项目/代码为结论依据；搜索摘要和第三方页面仅作线索。不是穷尽数据库的法定查新证明。

## 结论

1. “保存机器人自身成功经验，再拿来训练”在机器人学习、甚至不使用强化学习的机器人模仿学习中已有明确先例，不能作为SHBR独立创新点。
2. “成功经验”不是只有SIL才有。2020年的MILI已经自主采集、过滤无用/失败试次并纳入模仿训练；2026年的TSIL在机器人长时操作中显式缓存快速成功轨迹以避免好行为在后续更新中丢失。
3. 但这些论文不能直接等同SHBR：MILI是元/多任务模仿自改进，TSIL是PPO+优势加权动作似然；都不是已核实的“有限任务获得时采集→成功历史特征→冻结执行高斯参数→跨任务蒸馏+GMM路由”完整算法。
4. 最新直接持续机器人先例RIFAR已使用可靠性筛选与同历史动作漂移回放。它是世界模型生成回放，而非真实成功采集、固定教师高斯蒸馏。应纳入邻近工作，不能声称其与SHBR相同。
5. 对SHBR更准确的评价是：成功筛选部分已有先例；与历史策略蒸馏组合是否构成有意义的、足够区别于直接基线的算法增量，仍需和负责持续蒸馏/GMM的分报告一起判断。此检索没有找到完整同构方法，不等于证明不存在。

## 实际检索

广搜采用robot continual/lifelong imitation、successful trajectories、success-filtered replay、historical policy、self-imitation、distillation、autonomous improvement等组合；以查到文章的相关工作反向追踪SIL/GASIL/MILI、机器人SILP、RoboCat、SOAR和TSIL。核对年份/版本，明确“持续自改进”不自动等于任务序列抗遗忘。

## 深读条目

### SIL — Self-Imitation Learning，ICML 2018

原论文：https://proceedings.mlr.press/v80/oh18b/oh18b.pdf
出版页：https://proceedings.mlr.press/v80/oh18b.html
证据：§3、Algorithm 1、式(1)–(3)；论文PDF第2–3页。

保存过去的状态、实际采样动作和回报，以正回报—价值差加权动作负对数似然，并训练价值函数。好经验由优势而非简单终局成功标签定义。实验主要Atari及MuJoCo；这是RL自模仿，不是机器人任务序列抗遗忘蒸馏。对SHBR的影响：自主经验回放/模仿良好决策已有先例，但不能把SIL的样本动作BC称为缓存策略分布蒸馏。

### GASIL — Generative Adversarial Self-Imitation Learning，2018预印本

原文：https://arxiv.org/html/1812.00950
元数据：https://arxiv.org/abs/1812.00950
版本：v1，2018-12-03；不得将“Under review ICLR 2019”当已录用。
证据：§4.1、Algorithm 1、式(8)–(10)、§5.1。

按回报保留top-K整条经验轨迹，将其视为生成对抗模仿的目标；判别器学习好的轨迹分布，策略梯度使用相应奖励。另设PPO+BC(top-K)对照。点质量与MuJoCo任务，不是机器人跨任务固定教师保留。对SHBR的影响：优质/成功样本筛选+额外保持训练大方向不新；但GASIL没有SHBR的固定执行高斯监督与分量选择。

### MILI — Scalable Multi-Task Imitation Learning with Autonomous Improvement，2020

原文：https://arxiv.org/html/2003.02636
元数据：https://arxiv.org/abs/2003.02636
版本：v1，2020-02-25。
证据：§III（filtering function）、§IV Algorithm 1第12–24行、§IV-B。

已训练机器人自主执行，只将完成某种有用行为的试次加入数据池；以任务嵌入匹配成任务数据，再进行元模仿训练。原文明确不使用RL，且一次任务失败可作为另一任务成功。对SHBR：不能再把“从自身成功轨迹提取样本用于机器人模仿学习”当跨领域首次迁移。区别是MILI扩大/组织模仿数据进行多任务自改进，没有已核实的任务结束固定策略参数缓存与顺序抗遗忘蒸馏。

### TSIL — Temporal Self-Imitation Learning，2026预印本

原文：https://arxiv.org/html/2606.19752v1
元数据：https://arxiv.org/abs/2606.19752
作者代码：https://github.com/generalroboticslab/TSIL
版本：本次深读v1，2026-06-18。
证据：§3.2–3.3、式(1)–(3)、Appendix A。

各任务配置维护快速成功轨迹的小缓存，按效率选择并回放；重新标注当前时间目标/回报，采用正return-value gap门控和效率权重的动作NLL+value loss，加入PPO。明确动机包括防止稀有成功行为被后续更新遗忘。v1在15个长时操作任务上实验，但不是顺序任务持续学习保持协议。区别：sampled-action replay及价值学习，而非历史执行高斯蒸馏；并未逐任务固定获得时教师输出。

### RIFAR — Reliability-Informed and Forgetting-Aware Replay for Continual Robot Learning with World-Action Models，2026预印本

原文：https://arxiv.org/html/2610.03079v3
元数据：https://arxiv.org/abs/2610.03079
版本：v1 2026-10-02；本次深读v3 2026-10-19。
证据：§3.1–3.4、Appendix B.4/C.2/C.3。

直接针对持续机器人学习、LIBERO与真机。保存短历史示范前缀，世界模型生成旧轨迹，先按值预测完成条件及逆动力学一致性过滤，再按相同历史输入上适应前后的动作漂移选择回放。训练目标保留世界模型生成动作。重要区别：所谓生成成功来自模型终止预测，不等于真实环境成功；模拟器执行仅用于另行分析验证。没有核实到SHBR式真实成功样本+固定执行高斯蒸馏。它削弱“选择可靠历史来防遗忘”这一宽泛新颖性主张。

### ReDOA — Real-World Distillation via Observation Alignment for Image-Based Manipulation Policy Learning，2026预印本

原文：https://arxiv.org/html/2610.25691
版本：v1，2026-10-19。
证据：§III-B/III-D、§IV-B。

将模拟器训练的状态教师放到真机执行，收集成功轨迹和真实图像，训练图像学生；失败会重试/部分初态被排除。任务是分别训练的真机操作，不是连续任务抗遗忘。监督为动作轨迹的离线模仿，不能仅因文中称distillation就解读为缓存高斯分布蒸馏。对SHBR：真实成功执行数据供机器人学生学习已有先例；额外采集方式、预算、蒸馏目标、序列保持与SHBR明显不同。

## 相关但需避免误引的条目

### RoboCat，TMLR 2023

主论文：https://openreview.net/pdf/2a72d6fcd3472bdf5725a3e011fe30f7ad1c46e0.pdf
作者报告：https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/robocat-a-self-improving-robotic-agent/robocat-a-self-improving-foundation-agent-for-robotic-manipulation.pdf
证据：§2.1.2、Appendix G.1.1。

专门微调的策略自主产生额外数据，与先前数据合并训练下一版通用机器人模型。使用目标重标注，原文明确使用失败自主轨迹，因此不能简单写成“RoboCat只保留成功轨迹”。它说明自主数据回灌、历史输入和扩大技能集已有大规模机器人先例，但不等于固定容量顺序蒸馏保持。

### SOAR，2024预印本

作者项目：https://auto-improvement.github.io/
原文：https://auto-improvement.github.io/static/paper.pdf
arXiv：https://arxiv.org/abs/2407.20635
证据：方法部分的自主采集/成功检测与hindsight-relabeled goal-conditioned learning。

自主多任务采集、VLM成功检测，并通过目标重标注利用自主经验；不是仅成功筛选，也不是任务获得时高斯教师抗遗忘。属于自改进机器人系统的邻近数据利用先例。

### Self-Improving Embodied Foundation Models，NeurIPS 2025

原文：https://arxiv.org/html/2509.15155
证据：§2.2、Appendix A.3。

利用steps-to-go预测产生奖励与成功检测，自主练习并以RL自改进。不能把成功检测直接等同成功样本过滤蒸馏；此文不是SHBR式跨任务保持。其意义是自主执行与成功信号已是机器人学习常见设计材料。

### Self-Imitation Learning by Planning / SILP+，2021 / 2023

2021作者机构全文：https://pure.rug.nl/ws/portalfiles/portal/192059744/Self_Imitation_Learning_by_Planning.pdf
2023扩展：https://arxiv.org/abs/2306.06754

机器人运动规划利用已访问/成功状态规划生成示范，示范与交互使用分离buffer，并加BC训练。不是直接冻结过去完整策略输出的抗遗忘方法。可作机器人自模仿先例，不能冒充成功筛选持续蒸馏。

## 对SHBR可以与不可以说什么

不能主张首次：
- 机器人自身执行得到训练样本；
- 仅使用成功/优质轨迹再学习；
- 成功行为在后续训练中复习；
- 在模仿学习中融入自主执行样本；
- 可靠历史回放帮助持续机器人学习。

本次没有找到完全同构的单篇实现，但这不是首次性证据。值得作为明确待验的窄区别：
- 独立成功历史缓存与原示范回放并行；
- 任务获得时固定执行分布参数，而非采样动作或滚动教师；
- 面向argmax-component部署的高斯参数保持+编号不绑定的选择约束；
- 固定采集/内存/使用次数下的成功筛选效应。

这些窄区别需由其他分报告核对（尤其RECALL/CIDER/mixture蒸馏），并与同样自主数据量的普通蒸馏、公平缓存数的成功/未筛选，以及去路由/全分布保持对照比较。不要因组合未搜索命中即声称新范式。
## 追加核查：CEL/FERT与未验证线索

### CEL/FERT — Learning from Fleet Experience: A Collective Learning Paradigm for Robots，2026预印本

原文：https://arxiv.org/html/2610.25736v1
作者项目：https://cel-robot-learning.github.io/
版本：v1，2026-10-19。
证据：§3.1.1–3.1.4、式(10)–(13)、§4.1、Appendix F。

成功自主轨迹进入rolling buffer；失败轨迹提供检索上下文，检索相关成功片段并提高它们的训练优先级。目标写成记录动作chunk的NLL，flow策略以flow-matching实现；无需critic。冻结视觉编码器缓存的是检索嵌入，不是历史策略输出。五个任务分别训练task-specific policy，不能称为连续五任务抗遗忘实验。此文直接表明“成功自主数据＋冻结特征缓存＋额外监督更新”不是新材料，但没有SHBR的跨任务固定执行高斯保持或GMM选择约束。

### DMC-VLA：仅列未验证高相关线索，不用于结论

查询命中标题《Dual-Memory Consolidation with Success-Failure Contrastive Margins for Continual Vision-Language-Action Deployment》：
https://www.researchgate.net/publication/404051087_Dual-Memory_Consolidation_with_Success-Failure_Contrastive_Margins_for_Continual_Vision-Language-Action_Deployment

截至本次，精确标题和DMC-VLA查询仅找到该ResearchGate页面，未核验可靠的正式出版记录、arXiv版本、DOI或作者项目。摘要声称长期成功行为记忆和持续VLA，因此是高相关待核线索；但不应把其声明当已确证的发表先例，更不能用来宣布SHBR重复。已将此不确定性列入来源记录。

新增查询式：robot CEL continual experience learning success replay；robot continual success filtering policy；robot continual success-filtered replay policy；successful trajectories continual imitation robot；Dual-Memory Consolidation Success-Failure；DMC-VLA。
