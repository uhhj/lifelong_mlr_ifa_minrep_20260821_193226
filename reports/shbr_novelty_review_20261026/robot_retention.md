# SHBR查新子报告：机器人持续学习中的历史行为保持

检索截止：2026-10-26（Asia/Shanghai）。本文件仅为文献核验，不运行实验、不修改科学源码。原始论文与作者代码为证据；第三方摘要只用于发现线索。检索不能证明全球文献中不存在同一组合。

## 结论

SHBR的“旧执行样本＋任务结束时策略输出缓存＋后续行为保持”已存在非常直接的机器人持续学习先例。尤其ClonEx-SAC（2022）和RECALL（2023），不能把这条主线称为首次引入机器人领域。没有在本次核验的这些论文里找到“成功轨迹短历史＋argmax执行高斯W2＋最近分量margin”完整相同组合；这是限定检索范围的结果，不是独创性证明。

## 逐项原文核验

### ClonEx-SAC / Disentangling Transfer in Continual Reinforcement Learning

- 日期/身份：NeurIPS 2022；arXiv v1 2022-09-28。
- 核验位置：§5.2、§6、附录B.1和Algorithm 1。
- 每任务末从SAC缓冲随机选子集，以已训练网络输出标注，存入独立expert buffer。后续actor对缓存分布使用reverse KL；默认不正则critic。原始输入是Continual World机器人状态；仿真持续RL。
- 与SHBR最直接的重合：独立旧输入库、固定任务结束时输出、后续辅助行为保持。未采用成功轨迹专选、短视觉历史、GMM选中分量与margin。旧任务头用于新任务初始探索是其另一机制。
- [原文](https://arxiv.org/html/2209.13900#A2.SS1)；[NeurIPS](https://papers.neurips.cc/paper_files/paper/2022/hash/2938ad0434a6506b125d8adaff084a4a-Abstract-Conference.html)。可信度：高。

### RECALL / Replay-enhanced Continual Reinforcement Learning

- 日期/身份：TMLR 11/2023；arXiv 2023-11-20。
- 原文§4/Algorithm 1及官方代码双重核验。Continual World仿真、多任务头SAC；旧数据也参与RL，并以PopArt处理目标尺度。
- 代码`on_task_end`缓存当前任务状态上的`policy_mu/std`；`ExpertReplayBuffer.store`追加，不刷新此前目标；`auxiliary_loss`为KL(当前Normal‖缓存Normal)。因此任务获得时固定高斯参考、无需持续运行旧网络，都已存在。
- 与SHBR差异：不按成功整轨迹筛选，不是视觉短历史，不是5分量GMM执行分量W2和margin。监督主干为RL，而SHBR为IL；场景不同不能单独证明方法新颖。
- [论文§4](https://arxiv.org/html/2311.11557#S4)；[代码](https://github.com/Sweety-dm/RECALL/blob/55770cc2d711353d78c8b476a32acf0298570d7c/myrecall/methods/recall.py#L44)；[缓存](https://github.com/Sweety-dm/RECALL/blob/55770cc2d711353d78c8b476a32acf0298570d7c/myrecall/sac/replay_buffers.py#L82)。可信度：高。

### CIDER / Continual Interactive Distillation for Embodied Reinforcement Learning

- 日期/身份：2026-08-22 arXiv v1，真实存在，截止日前。
- 核验§4.1–4.2、§5、附录B/C。真实机器人六任务HIL-RL；历史transition的状态用于蒸馏，历史数据不用于critic Bellman更新。
- 每阶段开始冻结“上一阶段累计策略”，重新给全部历史状态缓存高斯均值/标准差；任务均衡reverse KL。教师是滚动刷新，区别于SHBR按任务获得时永久目标。
- “gradient routing”是adapter梯度流控制，不是GMM动作分量路由。没有在已读原文找到success-only archive规定；Archive具体选择细节未充分展开，不能断言成功/失败构成。单高斯策略，不是选中GMM分量保持。
- [原文§4](https://arxiv.org/html/2608.21899v1#S4)。可信度：高（方法与日期）；成功筛选细节未核明。

### ACCRUE / Continual Off-Policy Reinforcement Learning for Robot Foundation Models

- 日期：2026-10-13 arXiv v1；本轮之前已公开。
- 核验§4.1–4.3。冻结外部视觉特征；跨任务继承轻量actor/critic；每旧任务留最后N训练episodes；新旧actor样本1:1，critic仅当前任务。
- 每阶段用上一阶段actor重新生成确定性evaluation动作chunk并缓存，阶段内固定，缓存后丢弃teacher；目标每阶段刷新，参考action chunk保持采集时值。历史损失为输出MSE，不是方差蒸馏/GMM routing。仿真及真机RL。
- 这直接否定“固定视觉特征＋少量旧执行缓存＋无在线旧教师”可单独作为SHBR独有贡献。其末N采样不等于success-only。
- [原文§4.3](https://arxiv.org/html/2610.18596#S4.SS3)；[官方项目](https://accrue-rl.github.io/)。可信度：高。

### CPC / Continual Policy Consolidation for Lifelong Robot Learning

- 首版2026-01-30；核验版本v2 2026-09-06，均截止前。
- §4.1–4.3、Appendix A/B：独立RL教师产生轨迹，中央学生离线回归教师动作；DPP选历史轨迹，Transformer-MoE动态扩容及分阶段冻结。教师数据固定，不是滚动重打标签。
- MoE路由用于网络专家选择/负载均衡，不是GMM动作分量选择。未见success-only筛选；回归采样动作，不是SHBR缓存真实标准差。主结果Meta-World/Continual World，附加LIBERO；不是本文中已核验的真实机器人结果。
- [原文](https://arxiv.org/html/2601.22475v2#S4)；[作者代码入口](https://github.com/Agentic-Intelligence-Lab/Continual-Policy-Distillation/tree/cpc)。可信度：高（论文）；未逐行审计其代码。

### Zhou等：Forgetting and Imbalance in Robot Lifelong Learning with Off-policy Data

- CoLLAs/PMLR 2022；核验§5–7、Appendix B。
- 仿真双足机器人动力学连续变化；先用MPO在线采集，最终把累计经验作为离线RL数据，通过CRR重建通用策略，不依赖任务边界。其“distillation”明确是从数据到策略，不是冻结教师网络输出蒸馏。
- 因此只能作为机器人终身学习的经验质量/分布不均衡和离线重学习相关工作，不能误称为SHBR式任务末固定策略缓存。无success-only/GMM执行分量保持。
- [正式论文](https://proceedings.mlr.press/v199/zhou22a/zhou22a.pdf)。可信度：高。

### CPD / Continual Policy Distillation of Reinforcement Learning-based Controllers for Soft Robotic In-Hand Manipulation

- arXiv 2024-04-05；核验§III-C/D、§IV。
- 软体手物体旋转仿真；对象专用PPO教师产生状态动作示范，学生持续监督学习，比较回放策略。KL版本把采样教师动作视作均值，教师标准差固定1e-6，而非缓存原教师不确定性。
- 已在机器人持续模仿/蒸馏中结合有限样本回放，不能泛称“把策略保持从RL迁到机器人IL”即首次创新。与SHBR区别是外部独立专家与其数据、非success-only真实执行分布缓存、无执行分量路由。
- [原文§III](https://arxiv.org/html/2404.04219#S3)。可信度：高；真机未在已核原文中确认，按仿真表述。

### Memory Anchors for Continual Robot Learning

- 2026-08-27 arXiv v1；核验§3–5。
- 机器人持续模仿学习，LIBERO及真机。利用新旧观测表示重叠、动作冲突定位旧示范锚点，改变经验回放选样；不以整轨迹成功筛选，不缓存历史执行高斯。
- 这不是SHBR的相同保持目标，但已直接研究“哪些旧输入值得保护”，因此不能把改变保护位置本身视为新方向。它也与RA选样贡献相关。
- [原文](https://arxiv.org/html/2608.26545#S4)；[官方项目](https://robot-adaptation.github.io/MemoryAnchors/)。可信度：高。

### Dream to Remember（扩展线索，仅摘要核验）

- 2026-10-09 arXiv；[官方记录](https://arxiv.org/abs/2610.13816)。冻结世界模型生成旧任务想象轨迹以保持共享steering模块，涉及CALVIN、LIBERO、真机。
- HTML返回404、PDF工具受大小限制；未把第三方声称的“replay KL”细节当已核方法。只能据摘要判断其为相关的想象重放保持路线，不能据此判定具体损失重合。

## 创新边界与应做对照（本报告判断）

1. “终身机器人上首次旧数据回放/首次固定教师输出/首次策略蒸馏”不成立；最少引用ClonEx-SAC、RECALL，并说明与CIDER/ACCRUE的教师刷新差异。
2. “成功轨迹的部署历史＋选中GMM分布W2＋最近分量margin”是本次范围内尚未见完全同构的实施组合。不能把未找到等同首次，也不能把组合差异等同有效创新。
3. 强对照应包含适配后的任务末固定Normal/KL缓存、随机未筛执行历史、相同样本数量/使用次数/环境预算。W2与KL、是否保持路由需分别控制，避免把资源与损失差异捆绑。
4. O/S自身产生不同轨迹，比较的是全流程策略效果；要归因成功筛选本身，应使用共同采集池与匹配预算。无此实验不能宣称成功标记是已证实机制。
5. 目前可守住的定位是“面向时序GMM模仿策略的历史行为保持适配/组合”；强方法创新要靠区别的独立必要性和稳定闭环收益。

## 引用纠错

- 之前RECALL、CIDER的标题和arXiv号码真实，并未查到身份错配；CIDER仅以预印本标注，不能冒称已录用。
- 之前对“这是跨领域首次迁移”的描述过强：同领域ClonEx/RECALL以及CPD已明显覆盖基本框架。
- Zhou2022不属于在线新任务训练期间固定教师输出蒸馏，需避免概念混同。
- CPC/CIDER的“routing”不能作为与SHBR GMM routing逐项相同的证据。
