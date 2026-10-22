# CAET-v3 独立模块 OBJECT seed41

用户于2026-10-22明确确认三组：TAC、EFTR、RA-TAC-Align，顺序 TAC→EFTR→ALIGN。旧 TAC、AUX8、RSET、EFTR、COVR、LIBERO50 均不恢复。本轮是独立新版本，不续训旧模型，不扩 seed。

三组均从官方 OBJECT ep99/stateTrue 初始化和空本run buffer开始，复用原RA seed41已冻结特征及选样输入，原始文件只读。原八步输入、Temporal Transformer、五模式七维GMM、batch10、FP32、四阶段各100epoch/105500正式更新、warmup500/逐步scheduler、IFA系数.1/gamma.3、RA有序成员781/740/780/660及标定42、sampler0/shuffle41保持。正式执行沿用argmax分量Normal、阶段global RNG41、env41/42、初态0–19/10lane/600步/5zero；只评四个ep99模型，200正式/20vector/10cell。RA基线不重训重评。新增计算单独计账，不称等算力比较。

TAC与ALIGN：BC统一为a[u+1]。通过真实demo、possible_starts、八帧动作序列及填充规则核验唯一身份；bc_valid要求u+1<len(actions)，effect_valid另要求u+2<len(states)。原DataLoader人口与顺序不变，无效BC只屏蔽对应NLL并按有效数量取均值；全无效时可微零BC，IFA照常，正式步数不跳过。保留每步有效/无效BC曝光与样本侧车身份。原RA的BC为a[u]，此处是明确算法改动，不宣称监督不变。ALIGN无辅助模块或候选更新。

TAC：原Temporal末时刻512维隐藏表示+真实7维下一专家动作，EffectHead 519→256→128→7/ReLU；物理平移、SO3、篮子坐标系目标进展与旧科学定义一致。固定scale=[.01,.01,.01,pi/18,pi/18,pi/18,.01]，SmoothL1系数.03。效果回归可含静止样本，所有合法当前/保留历史样本参加；不另做效果拟合epoch或抽留出调参。128维L2投影、温度.1、跨任务对比系数.02。效果和投影头只训练使用，独立AdamW lr3e-4/wd.01，每有效主batch至多一次；梯度不通过采样动作进入部署GMM。

跨任务正图沿用v2规则：同物理组、非静止、固定效果/相对位置/夹爪兼容阈值、互top10、两侧至少3demo及12独立时刻。不再需要独立预测器留出门，因为本版是联合专家监督，不输出反事实控制梯度。图为隔离旧设计的可比性，仍排除原数值最后demo作为图留出（不排除主BC和效果回归）；不利用已看过正式结果调整。实际图用float32标签构造并与零训练图比对。对每批支持当前锚点，确定性读取最多一个不同旧demo的正伙伴（按step轮换）；负伙伴最多四个，来自合法旧RA，非静止、物理主方向组不同且固定尺度效果距离>=2，按身份排序轮换。去重同(task,demo,u)，无可信负对则该锚点对比为零并计账。伙伴只辅助前向，不改变原BC采样或增加正式optimizer步。TAC独立实验不混入EFTR候选机制，直接联合梯度；共享表示冲突仍是实验风险，不能声称已被消除。

EFTR：BC仍原RA a[u]，没有TAC对齐/EffectHead/对比或v2采样效果损失。每个旧任务使用本方法自己刚学完该任务的ep99为教师；在该任务RA入库成员确定后eval/no_grad缓存全部5分量means/std/logits及argmax，与成员sha及教师checkpoint sha绑定。后来仅读取保留成员和侧车，不访问旧丢弃示范。教师定义为原部署模式的实际标准差，模式固定属于连续模型锚定，不声称置换不变。

EFTR训练正则：教师选中kT到当前同kT的7维对角KL样本均值*.05，加relu(.1+max_other_logit-logit[kT])样本均值*.01。与跨任务图无关，Task7即可启用。每步从旧RA各任务身份排序、轮流交织的列表轮换最多32条做历史功能检查与正则，cursor完整恢复。候选0/.5/1从同一主参数/AdamW/梯度起点，由真实PyTorch AdamW与原grad_clip生成；只提交一个状态、scheduler只推进一次。确定性eval评分C为当前batch原BC NLL，R为教师执行分量到候选实际argmax分量KL/7 + .1*模式切换率。先C<=C0+.02；其中若有R<=.10优先该集合，否则保留预算失败记录；按(R,C,候选编号)排序。当前batch无当前样本时C为NA且不设虚假0；无旧记忆则严格走原RA更新。候选模拟恢复全部原随机状态、模式、梯度与requires_grad；不消耗额外正式步。局部规则不保证闭环遗忘或所有旧记忆硬预算。

准入先零训练身份/因果/RA成员/边界与图核查，再每入口自己的真实GPU隔离测试：关闭新增机制与RA精确一致；masked NLL与直接有效子集相同；TAC效果/对比可向Temporal及aux传播且不直接向GMM；EFTR真实三候选起点/优化器动量/一次提交/scheduler与全状态恢复精确；构造退化/模式切换检验约束响应。准入0环境，全部额外更新计账。未通过不得正式启动。TAC/ALIGN Task6改变BC，不能要求等于RA Task6 tensor；应记录差异，初始化仍必须逐tensor及RNG同父。EFTR Task6必须同父。

恢复只从完整epoch，全主+aux模型/优化器/梯度/modes/requires_grad/RNG/步数/旧检查cursor/teacher hash/support hash；存储幂等。正式评估不盲重启，技术失败留证版本化处理。科学负结果不重跑。所有正式退出、四CP严格100键、原成员/IFA、辅助与候选账目以及200/20完整记录验收后本地独立复算AUC/FWT/NBT/OldFinal/FinalAll，与RA和三组分别比较。当前均为探索性：已看过v1/v2/GOAL/此前诊断，参数不宣称独立事前验证。

仅当前新队列有启动授权。数据盘root空闲保留2GiB，系统仅普通空闲；不删其他资产、不新增付费资源。按AGENTS发布代码/报告/轻量证据并核MLR远端SHA，禁止模型/动作/状态/latent/buffer/RNG二进制上传。
