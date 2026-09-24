# 功能子空间更新归因实验：文献复核与 Codex 执行方案 v1

日期：2026-09-24。
状态：本文件是待执行的新实验协议，不代表实验已运行。

## 结论与范围

保留“旧任务 Mean-GN 敏感参数子空间 + 新任务条件更新增量 + 有限更新反事实”的主线。研究对象是**当前更新的局部作用**，不是永久知识标签，也不是每个旧窗口一个约束的联合 QP。

本轮分清三个问题：

1. 新任务的加入使更新有多少分量进入旧任务所选的敏感参数子空间？
2. 放行这个分量，相对于移除它，对独立轨迹的新旧 GMM NLL 有什么边际作用？
3. 利用这种结构做离线候选选择，是否比相同干预幅度、相同评价预算的廉价对照更有用？

闭环成功率是独立的行为评价；NLL 结论不替代它。

## 文献依据与不可外推的边界

以下以用户上传版本为准，不判断其他版本或发表状态。

- **Interference and Retention in Continual Learning**：图2（第3页）、§3.1–3.3、§4、§S15。保留输出敏感结构、局部几何与目标相关影响的区分；严格恒等式受线性/平方损失/收敛等条件约束，不能把论文的二次能量直接当成当前 GMM NLL。§S15 明确讨论有限步误差和评价锚点的分布。
- **Disentangling and mitigating the impact of task similarity for continual learning**：§3–5、§8。输入特征关系与输出要求关系需要分开；教师—学生模型中的 readout correlation 不能等同于机器人窗口动作余弦或参数投影率。
- **HReDiff**：图1（第2页）、§II-C、§III。损失信号可能遗漏关键行为退化，任务与状态层面需要区分。本文不移植其扩散熵估计和回放调度。
- **TSN-Affinity**：§3.3–3.4、图1（第9页）。其复用是旧权重冻结时的子网络复用，不等于沿旧敏感方向更新；本实验不能声称验证了该论文的复用机制。
- **FAGR**：§III-B/D、§V。特征统计之外仍需要动作对应与质量检查；统计压缩不自动证明知识共享或前向迁移。本轮不生成伪样本、不修改 buffer。
- **REGEN**：§6、图5（第9页）。模型内部或生成结果的成功不等于实际动作执行成功；本轮不引入世界模型。
- **SafeDiff**：图1、§5.1。成功与安全应分别评价；物理力方向不是本实验的参数方向，不照搬其力阈值。
- **DLO-Lab**：§5.2。动力学梯度/动作到物理状态的导数与策略输出对网络参数的 Jacobian 不是同一个对象；本轮不迁移模拟器。

补充工程依据：PyTorch AdamW 的更新依赖历史一二阶矩；梯度为零和梯度为 None 的执行语义不同。实际执行使用仓库安装版本，不因最新文档而升级环境。

---

# 可直接交给 Codex 的完整任务指令

## 0. 任务与禁止事项

实现并执行 `functional_component_counterfactual_v1`。

这是独立影子状态上的单步干预实验，不是正式训练，也不是只看相关性的观察性实验。

禁止：
- 恢复正式 lifelong/sample-QP 训练；
- 修改正式 MLR、IFA、buffer 采样、优化器、loss 或恢复槽；
- 调 KKT/物理写回阈值、做 correction inflation、量化修复；
- 构造逐样本/逐任务 QP；
- 新增 Fisher/GGN 变体、功能指纹聚类、生成式回放；
- 按评价结果更改 rank、步长、任务、数据划分或起点；
- 为得到明显效应，把小投影放大到整步更新范数。

仅新增独立实验代码、协议和报告。始终记录：
`formal_training_resumed=false`。

## 1. 先读取现有代码和资产，不从文字猜 API

仓库：`uhhj/lifelong_mlr_ifa_minrep_20260821_193226`。
预期研究分支：`codex/sample-qp-lifelong-v1`，以实际本地分支为准。

检查并复用：
- `reports/actual_update_fidelity_v1/`，历史参考提交 `4e89ae7`；
- `reports/gn_functional_geometry_v1/`，历史参考提交 `17e1a9f`；
- `reports/gn_h3_subspace_closure_v1/`，历史参考提交 `a1d5e97`；
- 对应候选 AdamW 更新、状态恢复、参数展平、Mean-GN 线性算子、GMM NLL、LIBERO 评价代码。

记录实际 HEAD、相关导入路径、Python/PyTorch/CUDA、GPU 型号、参数块和可用磁盘。
只做必要的定向版本/身份检查，不扫描整个历史实验目录，不反复计算全库 hash。

租用 GPU 可以更换型号/UUID。配对分支必须在同一运行环境执行；不要要求跨 GPU 逐位一致，不以 UUID 改变阻塞。

已有120起点来自 seeds45/46/47、阶段6→7和7→8、epoch30–99。不能写成 task8→9，也不能推断早期 epoch0–29。
只有 model-only checkpoint 不足以生成真实 AdamW 候选，禁止重新初始化 optimizer 冒充原状态。

## 2. 固定有限预算

### 离线主实验
- 从原120起点 manifest 的六个 seed×transition 单元中，各取6点，共36点。
- 每个单元按实际 epoch、batch_index、global_step 排序；当有20点时取零基位置 `[0,4,8,11,15,19]`。
- 起点不足20但不少于6时，使用固定等距索引规则，先写明实际索引；不能按结果补点。
- 每个单元第3个选中点用于技术 smoke，共6点。这6点仍包含在主样本中。
- smoke 只决定代码/数据链是否正确，不按效应正负决定继续；代码正确后完成剩余30点。
- 这些仍是开发种子与相关训练载体，不宣称36次独立确认。

### 行为核验
- 固定上述6个 smoke 起点，不按离线收益重新挑选。
- 每点只评估 KEEP 与 GN-REMOVE，两任务（该阶段前一个旧任务和当前新任务），各20个相同初始状态/种子。
- 合计最多480次完整 rollout；不追加训练步，不因结果接近显著而加试次。
- 若环境或历史观测重建不支持，完成离线部分并将行为核验标为 `NOT_RUN_ENVIRONMENT_UNAVAILABLE`，不能用离线预测替代。

## 3. 参数、任务与数据角色

### 参数范围
- θ：原生 MLR 当前实际训练的全部参数。
- θ_P：既有 Mean-GN 实验的 policy-head 参数，按实际代码映射，不猜层名。
- 构造空间、投影及删除只发生在 θ_P；非 head 参数在 KEEP/REMOVE/HALF 和所有方向对照中保持为相同的原生候选值。
- 不冻结原生解码器更新以伪造“真实 MLR 候选”；只是不对它做本轮干预。

### 任务范围
- 6→7 阶段：主旧任务为6，主新任务为7。
- 7→8 阶段：主旧任务为7，主新任务为8。
- task 数字与语义任务名称都由真实 manifest 确认。
- 其他已见旧任务作旁观测任务，不各建一个空间、不新增约束；报告它们的损伤，不声称保护了全部旧任务。

### 数据角色
对每个起点，先按身份分配并冻结：
- U：原生当前 new/replay 更新 batch，原样保留；
- G_old：构造 Mean-GN，目标128个现有 buffer 窗口、至少4条轨迹；
- D_old / D_new：离线判断或候选选择数据，各目标64窗口、至少2条轨迹；
- T_old / T_new：最终评价，各目标128窗口、至少4条轨迹。

G、D、T 的相同任务轨迹互不重叠；T 还不得与本步 U 的相同任务演示轨迹重叠。禁止以“窗口索引不同”冒充轨迹隔离。

G_old 与 D_old 必须来自当前阶段可访问的合法旧任务记忆；不得悄悄从已丢弃旧演示补充控制器信息。
T 可以由独立 evaluator 读取额外保留的原始演示，但其内容不得流入 G、D、候选构造或候选选择；必须记录这是离线评估特权，不是免存储。

若当前数据无法满足分配：
- 先完成可用性清单；
- 不能混用轨迹填满；
- 不伪造 trajectory ID；
- 报 `INSUFFICIENT_TRAJECTORY_SEPARATION`，相应任务对不作跨轨迹裁决。
不能按后续测量结果决定删哪个起点。

“隔离”指不参与本轮更新/空间估计/选择。还要注明这些演示是否可能在历史训练中出现；不能无依据称为模型从未见过的数据。

## 4. 新任务增量的唯一主定义

在同一 θ、相同 optimizer/scheduler 状态、相同本步数据与随机上下文中，分别计算：

- d_M：完整原生 MLR 候选参数位移；
- d_R：仅移除本批新任务 loss 项，保留 replay 原权重、原 reduction 分母和共同项后的候选位移；
- Δ_N = d_M - d_R。

名称固定为“给定回放与优化器状态的新任务条件增量”。
禁止称为历史无关的纯新任务知识，也不假设 AdamW 可线性分解。

实现要点：
1. 从同一次混合前向或完全相同随机前向取得按样本 source 可分解的原 loss；不能删掉新样本后把 replay 重新按较小 batch 归一化。
2. 在 clipping 前验证原 scalar loss 与梯度的分项重建。若有无法归属的 new-old 交互 loss，先报告定义缺口，不自行归类。
3. 两分支各自走原有 clipping、AdamW、LR 和必要 scheduler 顺序，不能共用已经被更新的 optimizer 对象。
4. 记录两支 clipping 系数、学习率、一二阶矩及 step；不得重置动量或改成 SGD。
5. 对原生会更新的参数，counterfactual 的缺失贡献使用显式零梯度，而非随意使用 None 跳过动量/衰减；原生未参与的参数则按原生语义处理。
6. d_M、d_R 均由实际候选参数读回，使用双精度相减得到位移。
7. d_M 必须与本机同状态的原生一步参数结果一致。历史跨硬件只比较数值容差与作用方向，不要求 bitwise。

## 5. 沿用 Mean-GN，不改造空间定义

对 G_old，在起点 θ 处：

    mean_a(x) = sum_m pi_m(x) * mu_m(x)
    J_i = d mean_a(x_i) / d theta_P
    C = mean_i[J_i^T W_a J_i]

- W_a 沿用已有动作分组/尺度规则，仅由 G_old 确定；不使用 D/T 调权重。
- 只改变起点时重新估计 J 和 U，不把另一个 seed/task/checkpoint 的旧 U 直接套用。
- 完全同参数、同 G、同 W、同参数映射时可以复用旧 U。
- matrix-free JVP/VJP 与已有特征求解器复用；禁止形成 P×P 矩阵。
- 计算 top32；主干预固定使用 top16。
- top8/top32 仅用于谱、截断与数值稳定性描述，本轮不追加干预网格，也不挑最优 rank。
- 精确正交基用 P=UU^T；保存基有小误差时用 QR，或 P=U(U^TU)^(-1)U^T。
- 不假定 top16 达到90%；报告真实 trace 覆盖与 eigengap。
- 若少于16个可分辨非零方向，记 r_eff 并单列，不人为补方向或隐藏退化。
- 数值检查复用已有标准；不足时使用相对特征残差、正交误差、toy 显式算子对照，允许预先限定的一次数值重算，不开启无限精修。

Mean-GN 只描述期望动作。其正交补不是整个 GMM 的安全空间；方差等变化仍需通过完整 NLL/分布检查。

## 6. 投影与“更新介入”指标

设 x=Δ_N 在 head 上的分量：

    v_P = P x
    v = embed_head(v_P)       # 其他参数坐标为0
    w = Δ_N - v

记录：

    rho_head = ||v_P||^2 / ||x||^2
    rho_all  = ||v||^2 / ||Δ_N||^2
    E_full   = x^T C x
    E_r      = v_P^T C v_P
    E_tail   = (x-v_P)^T C (x-v_P)
    E_cross  = 2 v_P^T C (x-v_P)

E_full 用完整 matrix-free 算子或 JVP 求，不得以 top16谱能量代替。
核验 E_full ≈ E_r+E_tail+E_cross。sum(lambda*z^2)只能称截断近似。

x或Δ_N近零时相关比例记 NA，同时保留起点与绝对范数。不能因rho>0就标记“有共享”，也不能把近零/空间外更新标为安全或没有复用。

## 7. 核心2×2反事实，加一个HALF剂量

从相同起点构造以下实际模型参数：

    BASE: theta
    X00:  theta + d_R
    X10:  theta + d_R + v
    X01:  theta + d_R + w = theta + d_M - v
    X11:  theta + d_R + v + w = theta + d_M
    HALF: theta + d_M - 0.5*v

主比较是 KEEP=X11 与 REMOVE=X01。
X00/X10 只为分析空间内外交互，不是新的训练路线。

实现：
- KEEP 直接复制真实 native candidate 参数，不经重复重建；
- REMOVE/HALF 从同一 native candidate 临时减去指定 head 分量；
- 全部候选从固定源状态构造，不累计相减；
- 读取实际 FP32 参数并用 float64 差分记录 realized displacement；
- 保存 KEEP-REMOVE 的真实差，与理想 v 比较；
- 若量化使二者相同，记 `BELOW_PHYSICAL_RESOLUTION`，不放大、不修复、不使用原QP的5e-6门限；
- 对 non-head 参数要求主分支一致；对 optimizer/scheduler/RNG 要求实验后恢复。

若缓存 decoder 输出：BASE、d_R、d_M 必须使用各自非head状态产生的正确缓存。禁止把原θ的decoder缓存用于已经更新decoder的候选。

## 8. 三类对照，区分结构价值与“少改一点”

全部在同一个 head 参数块干预，其他参数等于KEEP。
每种方向都有 λ={0,0.5,1} 三档，统一写成 d(λ)=d_M-(1-λ)*v_method。

### C1：自然随机等秩投影，3个固定种子

生成不看D/T标签的 Haar/高斯QR随机 r_eff 维正交基 R：
    v_random = embed_head(R R^T x)

保留自然投影范数，不把小投影放大。记录随机rho及r_eff/P的几何参照。
用途：观察实际更新与随机空间的自然介入程度，不单凭此对照宣称等干预优势。

### C2：同投影范数的随机方向/子空间对照，3个固定种子

避免把幅度差当成方向收益。令 a=rho_head；生成单位z且z与x正交：

    t = a*x + sqrt(a*(1-a))*||x||*z

它满足：
    ||t||=||v_P||
    x^T t=||t||^2

故t可以是某个正交子空间对x的投影。需要显式r维基时，用t/||t||为第一列，其余r-1列从同时正交于x和t的空间补齐。

不要对自然随机小投影进行暴力缩放。a≈0或1的退化情况单列。
本对照使用GN测得的投影幅度，是干预归因对照，不是声称零GN成本的可部署算法。

### C3：head新任务增量整体衰减，廉价主对照

    kappa = ||v_P|| / ||x||
    v_shrink = embed_head(kappa*x)

同样有 ||v_shrink||=||v||。删除量、参数块、候选数与GN一致。
这避免了匹配整步d_M时可能出现||v||>||d_M||、负缩放或改变其他参数块的问题。

随机种子、构造顺序先冻结。不得从3次随机重复中挑最好或最差一次；主比较使用预先固定的平均，保留每次结果。

## 9. 评价：完整GMM NLL为局部代理，不能冒充知识共享

对 D/T 的每条完整窗口按官方terminal-frame GMM NLL计算；窗口长度、mask、reduction和动作目标与原训练保持一致。

每个任务先按轨迹平均，再对轨迹等权。报告原始逐窗口值，但不把相邻窗口当独立实验。

每个任务j计算：

    marginal_with_rest_j = L_j(X11)-L_j(X01)
    marginal_without_rest_j = L_j(X10)-L_j(X00)
    interaction_j = marginal_with_rest_j-marginal_without_rest_j
    absolute_keep_j = L_j(X11)-L_j(BASE)
    absolute_remove_j = L_j(X01)-L_j(BASE)

interaction明显时只称“在其余更新存在时的条件贡献”，不把效应唯一归属于某份知识。

分类主标签：
- 两任务marginal均可分辨为负：`JOINT_NLL_BENEFIT`；
- 旧正、新负：`OLD_NEW_NLL_TRADEOFF`；
- 旧负、新正：`OLD_FAVORED`；
- 均正：`BOTH_WORSE`；
- 新负、旧在等价区间内：`NEW_BENEFIT_OLD_NEUTRAL`；
- 其他或不确定：`UNRESOLVED`。

不能命名为“已证明共享知识”“已证明成功率提升”。
负的边际效应只表示比REMOVE好；不代表比BASE好，必须同时报告absolute结果。

## 10. 数值与抽样不确定性分开

- deterministic forward重复用于核对随机性，但重复值相同不等于没有舍入误差。
- 在6个smoke点的D数据上，对固定FP32物理参数做原评估与float64-head评估的损失差比较；只用于数值校准，不改变模型参数候选。
- 记录每任务数值差阈值tau_num = max(1e-7, 5*所测配对效应数值差的最大值)。
- 描述性NLL等价带固定 tau=max(tau_num,1e-4)，单位为每terminal-window NLL。1e-4仅是本轮预先选定的解释带，不是行为安全标准。
- raw连续效应是主结果；同时报告tau_num与tau的分类，不能只选较好一档。
- T上使用成对轨迹bootstrap区间：区间完全低于-tau才标负，完全高于tau才标正；完全落入[-tau,tau]才标近似中性；跨边界为UNRESOLVED。
- 小轨迹数导致的宽区间如实保留，不追加窗口冒充独立轨迹。

不要求每个起点都出现显著共享或冲突。弱投影、无效应、不确定起点保留在分母并报告覆盖率。

## 11. 关键状态和分布检查，不改变采样/训练

从既有评价窗口元数据标记：
- 演示相对时间前/中/后三分之一；
- 若有可核对专家夹爪命令，命令开闭切换附近±2帧窗口。

后者只称“夹爪指令变化窗口”，没有接触/抓持标注时不称成功抓取关键帧。
对这些分层报告NLL变化、正损伤均值、p90和最大值及样本/轨迹数。
不根据本轮loss增量选择所谓关键状态。

另外报告KEEP与REMOVE之间的期望动作变化和完整GMM分布漂移：
- 从KEEP分布按固定seed每窗口采64个动作；
- 用完整GMM log_prob估计KL(KEEP || candidate)，所有候选共用KEEP样本；
- 记录Monte Carlo标准误，有限采样的负估计不能静默裁成0；
- KL只衡量分布变化，不作为改善/损伤标签，不用于新建Fisher。

若实际分布接口无法可靠采样/log_prob，则明确NOT_AVAILABLE，不以均值距离冒充完整分布。

其他旧任务的T-monitor：每任务目标16窗口；若有数据，报告各任务与最差任务，不纳入选取主空间，不新增QP。

## 12. 离线“可利用性”检查，与机制效应分开

这是离线候选选择评估，不训练门控网络、不写入正式优化器。

对每种方法，统一候选λ={0,0.5,1}，在D上固定选择：

    J_D(lambda) = 0.5*L_old_D(lambda)+0.5*L_new_D(lambda)
    lambda_D = argmin J_D(lambda)

数值并列时优先λ=1，其次0.5。权重不根据T调整。
这个标量只定义本轮共同NLL目标，不声称代表机器人总体效用；旧、新、最差旧任务结果单独报告。

冻结选择后，才打开T结果计算：
    gain_method = J_T(KEEP)-J_T(lambda_D)

主对比：
- GN vs C3整体head增量衰减；
- GN vs C2三次等干预随机对照的平均。

C1自然随机结果辅助报告。
同时报告GN固定λ=0和0.5，以区分几何效应与动态选择效应。

禁止在T上选λ再当算法结果。T上事后最优如需计算，只能另列`ORACLE_DIAGNOSTIC`，默认不算。

本轮D选择可以检验有限更新反馈是否跨轨迹迁移；它不证明未来无需forward probes的预测门控已经成立。

## 13. 统计分析与裁决

### 必须报告
1. 每起点rho、E_full/E_r/E_tail、物理可分辨性；
2. D与T的边际效应及类别一致性、UNRESOLVED覆盖率；
3. 条件交互项及旧/新绝对损失变化；
4. 三档λ的效应曲线，不能预设单调；
5. 等干预方向对照和固定/动态选择对照；
6. 逐任务/轨迹/关键状态尾部损伤；
7. 每个seed×transition单元以及每个seed的结果。

### 重采样口径
- 不能把36起点或成千窗口假设成完全独立。
- 成对bootstrap对各分支使用相同采样权重。
- 对同一seed/task中重复出现的trajectory_id共享bootstrap权重，保持跨起点的轨迹依赖。
- 对每个载体内6个时间有序起点采用长度2的循环块bootstrap，并报告整载体汇总与leave-one-seed-out描述。
- 3个开发seed不足以证明跨种子普遍性；CI标为本固定载体集合上的条件区间。
- 主方向对照只有GN-vs-C2、GN-vs-C3两项，使用5000次成对块重采样，给各自97.5%区间作为两项比较的保守同时控制；其他比较描述性报告。

### 状态必须分开
- `MEASUREMENT_VALIDITY`：实现/配对/数据角色是否成立；
- `PROJECTED_UPDATE_ENGAGEMENT`：报告大小和参照分布，不把非零当创新门禁；
- `LOCAL_EFFECT_REPLICATION`：可分辨局部作用是否在D/T复现；
- `GEOMETRY_ADDED_VALUE`：GN相对等干预对照的离线选择增益；
- `BEHAVIORAL_EFFECT`：单独来自rollout；
- `RESOURCE_COST`：开销，不和统计显著性合成一个分数。

`GEOMETRY_ADDED_VALUE=SUPPORTED_ON_DEVELOPMENT_SET`必须至少满足：
- GN自身T上共同NLL增益区间下界>0；
- 对C2与C3的两项配对增益差同时区间下界>0；
- 三个seed平均方向均不反向；
- 旧/新目标的代价与旁观测旧任务损伤全部披露。

未满足时是INCONCLUSIVE或NOT_ESTABLISHED，不通过调rank/改阈值补救。
这不是最终方法或行为安全PASS。

## 14. 6点小规模闭环核验

离线测量有效且环境可用后，在预先固定6点执行，不按收益挑点，也不要求离线指标先显著。

- 每点KEEP与GN-REMOVE；旧/新两任务各20次；共480次rollout上限。
- 使用仓库原生评估动作规则，不能为了本实验改成GMM期望动作。
- 两分支使用相同初始状态、环境种子、policy sampling seeds和horizon。
- 每个episode完整重置状态与策略时序上下文；不能仅替换权重而保留上个分支的隐状态。
- 测成功率、已有可验证的任务进展/违规指标和执行长度；没有力传感/标注时不造SafeDiff式力安全指标。
- 仅冻结候选权重做评估，不继续训练一步。
- 不更新子空间，不重选λ，不追加trial寻找显著性。
- 给配对成功/失败转移表和不确定性；20次/任务通常只能作为低功效行为检查，没有显著差异不等于等效。

本轮不执行5/20步续训或完整lifelong；这些需下一独立协议。

## 15. 只保留必要工程门禁

必要门禁：
A. 起点、参数映射、原生候选parity和状态回滚正确；
B. 数据身份、轨迹隔离与D/T使用边界明确；
C. 空间算子/投影正确，候选值有限，KEEP/REMOVE非head相同。

模型状态、optimizer或数据角色真正错误时停止对应起点并落盘。
以下不能当停止理由：
- GPU UUID变化；
- 新旧效应不显著或方向不如预期；
- 小投影被FP32量化成零；
- 不满足以前sample-QP的物理约束阈值。

环境启动失败应一次性检查配置/本地CLIP缓存/模型加载，不建立无限审计循环。不能长replay重建已存在的合法完整状态。

## 16. 测试、资源和落盘

至少测试：
1. 原生loss分项重建与counterfactual reduction；
2. AdamW同状态两影子分支、None/zero语义和完全回滚；
3. projector幂等、自伴及固定span内换基不变；
4. x=v+(x-v)，能量分解；
5. 2×2候选参数与交互恒等式；
6. 等幅随机构造的norm、dot与可投影性；
7. KEEP与原生candidate一致，REMOVE只改head；
8. toy模型中“无参数更新仍能复用旧函数”与“相同敏感度、相反专家残差”的反例；
9. D选择不读取T；
10. 分支评价结束模型、optimizer、scheduler、RNG、mode均恢复。

记录每起点：两候选生成、GN建基、投影、D/T前向、KL采样、rollout的时间/显存。
不根据前次118秒直接承诺本轮耗时。完成smoke后用实际吞吐估算剩余工作；不在低价值统计上无限增加计算。

数据与基顺序处理，不同时驻留36个完整副本；基可按固定规则保存/释放并保留可再现信息。写文件原子化、可按start_id续跑，禁止重复已经完成的起点。

目录建议：
    functional_component_counterfactual_v1/       # 新实验代码
    reports/functional_component_counterfactual_v1/

必须有：
    PROTOCOL.json
    SOURCE_AND_CONTEXT.md
    START_MANIFEST.json
    DATA_ROLES.json
    PROPOSAL_PARITY.json
    GEOMETRY_CHECKS.csv
    ENGAGEMENT.csv
    CANDIDATE_EFFECTS.csv
    INTERACTIONS.csv
    SELECTIONS_D_ONLY.csv
    T_EVALUATION.csv
    STATE_AND_TRAJECTORY_STRATA.csv
    NUMERICAL_RESOLUTION.json
    CELL_AND_SEED_SUMMARY.csv
    STATISTICAL_REPORT.json
    RESOURCE_COST.json
    BEHAVIORAL_REPORT.json
    TEST_REPORT.json
    SUMMARY.json
    FINAL_REPORT.md

图只需：介入比例、能量覆盖、旧/新边际效应散点、KEEP/HALF/REMOVE响应、交互项、等幅对照、行为配对结果。不要自动再生成研究路线图或扩展候选。

## 17. CLI与执行顺序

为本新模块实现以下CLI（这些是本任务要求的新接口，不是声称仓库已有API）：

    python -m functional_component_counterfactual_v1.run --stage inventory
    python -m functional_component_counterfactual_v1.run --stage smoke
    python -m functional_component_counterfactual_v1.run --stage offline --resume
    python -m functional_component_counterfactual_v1.run --stage behavior --resume
    python -m functional_component_counterfactual_v1.run --stage report

inventory结束后，在任何D/T效应测量前冻结协议和manifest。
smoke通过技术检查后继续offline，不用重新请求解释性确认。
behavior仅在真实环境可用时执行；缺失则给出具体缺项并生成NOT_RUN报告，不阻止offline报告完成。
只定向提交新实验代码和报告，不覆盖历史结论或正式资产；远程push沿用本次执行环境中用户明确授权的流程。

## 18. FINAL_REPORT必须回答

1. 我们测到的是新任务条件更新介入，还是错误地叫成了知识复用？
2. Mean-GN top16覆盖多少本次更新能量，截断外还剩什么？
3. 新任务条件增量在head/全参数中的占比及原生进展是什么？
4. 保留空间内分量的边际好坏能否跨轨迹复现？
5. 条件交互是否改变结论，边际受益与整步受益是否一致？
6. 同干预幅度、同候选数、同D数据下，GN是否胜过廉价对照？
7. 共同NLL收益是否掩盖其他旧任务或夹爪指令变化窗口的损伤？
8. 6点rollout支持什么、不支持什么，NLL与行为是否一致？
9. 算力、显存、历史数据访问成本是多少？
10. 下一步是有限规模续训，还是不再推进该几何；依据是什么？

最终不得只输出一个PASS。必须保留各层结论和全部未运行/不确定项。
`formal_training_resumed=false`。
