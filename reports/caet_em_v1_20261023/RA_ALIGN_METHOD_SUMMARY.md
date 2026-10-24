# 当前RA-Coreset与ALIGN的具体思想

本摘要以实际执行协议为准，不将早期设计稿或后续CAET-EM模块混入已完成RA/ALIGN。仅核对本地已有协议、时序审计说明与三seed汇总，无新前向、训练或环境。

## RA-Coreset：随机总体支持与几何覆盖结合的回放选样

RA命名为Reservoir-Anchored Coreset，但当前实际实现是每任务训练结束后，在完整候选窗口上无放回随机抽样作为硬锚点，并非新增在线reservoir替换训练器。纯K-Center优先覆盖远离现有样本的位置，可能偏向几何边缘而改变常见行为的回放分布；随机锚点旨在补充来自原总体的均匀随机支持，不保证完整buffer严格保持原分布或关键行为。

对当前任务N个窗口，配额q=ceil(N/2)。先复现原K-Center集合S0；再复现同seed原Random完整q条名单，取前floor(q/2)为硬锚点A。这里锚点约占buffer的一半、原候选的四分之一，不是原数据的一半。按原Random顺序加入缺失锚点，每加入一条只删除一个非锚点：优先使全候选到保留集合的最大最近距离最小，再比较平均最近距离和固定文件ID。恰好|A−S0|次交换，最终数量q且包含A。不是先随机取一半再重新运行一套K-Center。

距离沿用8帧双相机视觉token及末端7维专家动作，视觉/动作权重各0.5、原稳健尺度标定与selector_seed42。选中真实窗口原字节入库；之后训练使用完整当前任务加旧RA记忆，IFA沿原算法按实际buffer计算。模型、BC、IFA公式和部署GMM不增加新模块；距离与覆盖不是闭环收益保证。

实际保留原文件排序、concatshuffle和sampler、每stage全局动作RNG与argmax-component Normal部署；未启用初始附件中的stable-slot或E1独立噪声。成员变化仍可能间接改变数据顺序、优化路径、IFA及评估随机流，不能声称纯成员因果已完全隔离。不使用损失/梯度/闭环失败选样，不增事件配额或样本复制。

## ALIGN：将行为克隆监督对齐到当前观测之后的下一动作

在本项目实际导出的post-action观测时序中，窗口末观测已经包含a[u]执行后的结果。原RA用a[u]监督；ALIGN将目标改为a[u+1]，让策略从当前历史预测接下来要执行的动作。这里的索引结论来自当前缓存与真实demo的时序核验，不宣称所有数据集均应统一加1。

严格核验sample/demo/u唯一身份、八个真实未填充帧及u+1存在；合法样本计算下一步GMM NLL，无效样本只屏蔽BC。原DataLoader人口与优化步不变，IFA照常；全无效batch使用可微零BC。旧RA成员保留与其身份绑定的合法下一动作侧车，而非用不确定身份猜标签。

固定原8×4×512输入、Temporal、原控制读出、原单步五分量七维GMM、官方初始化、RA成员/配额、BC+.1IFA及各suite自身超参。无辅助效果预测、迁移、历史解码保护；没有B0的modality3完整最新帧控制读出、随机四步动作映射或四步监督。ALIGN是在RA回放底座上的时序监督修正，两者作用轴分别为记忆成员选择和监督目标时序。

## 已有结果的边界

相同41/42/43三seed等权AUC：OBJECT RA73.229167%→ALIGN80.173611%，+6.944444pp；GOAL RA81.979167%→ALIGN81.111111%，−0.868056pp。OBJECT OldFinal仍−0.555556pp，GOAL OldFinal−4.444444pp；不能把AUC总体改善称每个旧任务均被保护，也不能宣称跨套件一致改善或显著性。

RA解决有限记忆中覆盖与总体支持的取舍；ALIGN解决观测和下一动作的时间对应。二者都未直接解决闭环访问状态泛化、GMM分量选择或长期控制函数保护。本摘要不启动实验、不修改冻结协议、不恢复监控。

依据：20261018_ra_coreset_goal_seed42/PROTOCOL.md（优先覆盖其attachment设计稿）；20261019_ra_object_conditional/PROTOCOL.md；20261022_align_object_extension_goal_conditional/PROTOCOL.md及IMPLEMENTATION_NOTES.md；同目录OBJECT/GOAL三seedaggregate报告。文件SHA见RA_ALIGN_METHOD_SOURCES.json。
