**下一轮固定为“G1-OBJECT-v1”：关闭E1和P1，只在原MLR＋IFA中加入BC主导的IFA二值门控，先完成OBJECT的seed 42四阶段训练与200次主闭环评估，以总体AUC是否超过历史同seed基线决定是否补跑另外两个seed。**

本轮不新增旧知识保持损失，而是检验：**IFA与当前批次的新旧专家行为监督出现梯度方向冲突时，暂时关闭IFA，是否能够改善总体学习表现。**原论文将BC和IFA联合优化，本实验只改变IFA在每一步是否参与，不改变BC与潜在回放本身。Yu 等 - Lifelong imitation learn…

以下为固定执行方案。算法与新增配置是本次实验设计，不是论文已有结论。

# 一、实验范围、假设与比较对象

## 1. 本轮只新增一个模块

\[
\boxed{
G1=\mathrm{MLR+IFA}+\text{BC主导的IFA二值门控}
}
\]

| 项目 | 执行安排 |
|---|---|
| 实验名称 | `G1-OBJECT-v1` |
| 数据集 | 仅LIBERO-OBJECT |
| 训练序列 | 任务6、7、8、9，完整四阶段 |
| 首轮seed | 42 |
| 初始化 | 已有OBJECT预训练检查点 |
| 参考结果 | 复用历史MLR＋IFA，不重跑 |
| 预训练、潜在特征 | 直接复用，不重新生成 |
| E1、P1 | 完全关闭 |
| 首轮成功后 | 冻结配置，只追加G1的43、44 |
| 本轮不加入 | 行为分组、梯度投影、教师蒸馏、回放配平、动态阈值、参数扩展 |

G1必须从原预训练检查点开始，**不能从E1、P1或历史基线已经完成持续学习的模型上接着训练**。同一次G1运行内部，四个阶段连续继承模型参数。

## 2. 验证问题

本轮的唯一主要问题是：

> **在其他设置不变时，对原IFA进行逐步选择性启用，是否比始终按原规则使用IFA获得更高的OBJECT总体闭环AUC？**

这里“BC主导”中的BC，指**原训练混合批次上当前样本与旧回放样本的共同BC损失**，不是只优先考虑新任务。

梯度相似性用于调整辅助损失已有研究基础，但它提供的是方法启发，不是当前AdamW机器人策略的收益保证。[arXiv](https://arxiv.org/abs/1812.02224)

## 3. 固定比较基准

已有冻结协议记录的历史seed 42主结果如下：

| 总体指标 | 历史MLR＋IFA |
|---|---:|
| AUC ↑ | 77.708333… |
| FWT ↑ | 77.500000 |
| NBT ↓ | −0.277778… |

历史协议同时保存了seed 43、44的对应结果，后续可直接复用。**不能拿G1与失败的E1或P1比较后，就宣布获得收益。**

最终决策只依据总体指标：AUC为主要指标，NBT和FWT解释收益与代价。单任务表现没有独立否决权。

---

# 二、冻结原实验协议，不同时改变其他设置

## 1. 数据与训练设置

| 项目 | 固定设置 |
|---|---|
| 预训练基础 | 原6任务预训练模型，不重新训练 |
| 增量阶段 | 4阶段，每阶段新增1个任务 |
| 新任务示范 | 每任务10条 |
| 每阶段训练 | 100 epochs |
| 基础batch size | 10 |
| 优化器 | AdamW |
| 初始学习率 | \(10^{-4}\) |
| AdamW参数 | betas＝(0.9, 0.999)，eps＝\(10^{-8}\)，weight decay＝0.1 |
| 调度 | 已冻结的`batch_step`线性调度，warmup＝500步 |
| 原IFA权重 | 0.1 |
| OBJECT角度缩放系数 | 0.3 |
| 缓存 | 原保存规则、容量和`past_data_rate=0.5` |
| 主评估检查点 | 各阶段最终`ep99` |
| 主评估次数 | 每有效任务—阶段组合20次 |

四阶段任务划分、每任务10条示范，以及100 epochs、batch size、初始学习率和IFA系数具有论文依据。AdamW其余参数、warmup、调度调用单位和`ep99`主端点则是**已经冻结的项目实现选择**，不能全部描述为论文明确规定。Yu 等 - Lifelong imitation learn… Yu 等 - Lifelong imitation learn… 

## 2. 必须沿用的实现细节

从已冻结基线配置读取并保存：预训练检查点校验值、特征清单、示范选择、任务顺序、正常回放采样、梯度裁剪、训练精度、每阶段优化器初始化规则、环境初态、动作采样方式和最大执行步数。

尤其注意两个边界。

**不重新解释IFA实现。**本次读取的`base.py`在原循环中逐项累加IFA，并保留了`ifa_own_reference_fix`等实现选项。G1必须继承现有选项、任务对顺序和累加方式，不能同时修改参考选取、配对平均方式或角度计算。

**不沿用P1的教师目录或E1的检查缓存。**建立G1独立的可写缓存目录，按原增量起点初始化为空；已有潜在特征只读复用，不能把上一条实验结束时的缓存直接作为G1初始缓存。

---

# 三、G1算法：只决定原IFA是否参与本次更新

## 1. 主损失与辅助贡献分别保存

设本次原训练批次为 \(B\)。定义：

\[
L_{\mathrm{BC}}(B)
=
\text{原实现对整个混合批次计算的BC损失}.
\]

保留原IFA各项，记：

\[
A_{\mathrm{IFA}}
=
\text{原实现本次准备加入的全部加权IFA贡献}.
\]

这里的 \(A_{\mathrm{IFA}}\) **已经包含原权重0.1**。于是，数学上：

\[
L_{\mathrm{native}}
=
L_{\mathrm{BC}}+A_{\mathrm{IFA}}.
\]

使用加权贡献求门控梯度，与使用未加权IFA具有相同的内积符号，因为原权重为正；这样也能减少实现中重复乘0.1的风险。

**实现时必须分别保存BC与IFA项，不能通过`总损失－BC损失`反推出IFA。**原BC可能是较大的负NLL，相减会引入不必要的数值抵消，也可能掩盖损失构造错误。

原模型的BC仍使用原GMM似然与原目标位置；门控不改变专家动作监督定义。

## 2. 门控梯度范围：整个可训练时序解码器

定义共享参数集合：

\[
\phi=
\{\text{时序解码器中所有原本可训练的参数}\}.
\]

在当前代码结构中，应从底层策略模型的`temporal_transformer`模块获取这些参数，保存其名称、形状和数量用于核验，而不是依赖任意字符串包含关系筛选。

计算：

\[
g_{\mathrm{BC}}=\nabla_\phi L_{\mathrm{BC}},
\qquad
g_{\mathrm{IFA}}=\nabla_\phi A_{\mathrm{IFA}}.
\]

**不把策略头纳入门控内积，但策略头仍正常训练。**原因是IFA作用于解码器输出表示，策略头的直接监督来自BC；原论文也将解码器与策略头作为持续学习阶段的可训练部分。Yu 等 - Lifelong imitation learn…

本轮不改成最后一层梯度、表示梯度、逐层投票或逐任务梯度。这些是不同的门控信号，不能混入G1-v1。

## 3. 一个批次只计算一个门控

计算整个解码器参数空间上的内积：

\[
d_t
=
g_{\mathrm{BC}}^\top g_{\mathrm{IFA}}
=
\sum_{p\in\phi}
\left\langle
\nabla_p L_{\mathrm{BC}},
\nabla_p A_{\mathrm{IFA}}
\right\rangle.
\]

门控固定为：

\[
\boxed{
w_t=
\begin{cases}
1,&d_t\ge0,\\
0,&d_t<0.
\end{cases}
}
\]

最终目标：

\[
\boxed{
L_t
=
L_{\mathrm{BC}}
+
\operatorname{stopgrad}(w_t)A_{\mathrm{IFA}}
}
\]

含义是：方向不冲突时保持原IFA；方向冲突时，本步不采用IFA，但**BC和旧回放继续正常训练**。

即使本批次包含多个原IFA任务对，也对其总贡献计算**一个门控**，不能逐对开关。门控同样不能拆成新任务门、旧任务门或行为组门。

## 4. 每次有有效IFA贡献时判断，不继承E1的检查间隔

G1没有“每10步检查一次”的规则，也没有额外检查batch。

**每个原训练步骤，只要原IFA确实产生非零贡献，就根据该步梯度重新判断。**不复用上一批次的门控，不使用移动平均，不累计历史投票。

本批次即使没有旧回放样本，只要原规则允许产生IFA，仍按本批次BC与IFA判断。不能偷偷增加“必须包含旧样本才能门控”的新条件。

## 5. 边界处理固定如下

| 情况 | 固定处理 |
|---|---|
| 原规则本来不计算IFA | 执行原更新，不调用门控梯度探测 |
| 原IFA所有hinge均为零 | 保留原损失路径，记录`HINGE_ZERO`，无需探测 |
| 有效梯度内积为正或恰好为零 | 保留原IFA |
| 有效梯度内积为负 | 本步关闭IFA |
| 某一整体梯度范数为零 | 按 \(d_t=0\) 保留原IFA；余弦记为不可定义 |
| 部分参数结构性未连接 | 对应内积分量按零处理，并记录未连接数量 |
| 正IFA贡献却与整个解码器完全断开 | 视为实现错误，停止排查 |
| 损失、梯度或内积非有限 | 保存现场并停止，不静默关门继续 |

不增加“内积足够负才关闭”的阈值。本版阈值就是零；小梯度和接近零的内积可以记录，但不因此动态改变规则。

**门控开启率高或低都不构成提前停止的依据。**

---

# 四、一次训练步骤如何实现

## 1. 两种梯度必须来自同一次原训练前向

每个batch只执行原BC＋IFA需要的前向，保持原训练模式和原dropout。

然后，在**同一参数位置、同一批次、同一个前向计算图**上分别取得BC和IFA梯度。

本轮不使用E1或P1的确定性检查分支，不关闭基础训练dropout，也不重新前向一次计算门控。否则，比较的梯度可能来自不同随机实现，且会同时改变原训练过程。

门控是在当前训练随机样本上的局部信号，不是对总体任务关系的无噪声判断。

## 2. 使用两次梯度探测，再进行一次正式反向

建议使用：

```python
torch.autograd.grad(
    loss,
    shared_decoder_parameters,
    retain_graph=True,
    create_graph=False,
    allow_unused=True,
)
```

分别取得 \(g_{\mathrm{BC}}\) 和 \(g_{\mathrm{IFA}}\)。两次探测都保留原计算图，且不构建高阶梯度图。`autograd.grad`返回所请求的梯度，正式的`.grad`累积留给后面的总损失反向。[PyTorch Docs](https://docs.pytorch.org/docs/2.13/generated/torch.autograd.grad.html)

确定 \(w_t\) 后：

- 开门：对原样构造的`native_loss`反向。
- 关门：只对保存的`bc_loss`反向。

**第一版不手工拼接并写入参数梯度。**这样可以保留原梯度裁剪、AdamW和调度的更新路径，也更容易验收。

## 3. 保留原损失相加顺序

当前代码在IFA循环中执行：

```text
loss = loss + ifa_weight * current_hinge
```

G1应保留这条原总损失构造顺序，同时另外收集加权IFA项供梯度探测。**开门时直接使用原总损失张量，关门时直接使用原BC张量，不做总损失减法。**

这既避免重复加权，也避免为了“统一公式”重写原循环中的求和或平均方式。

## 4. 内积计算与优化器处理

基础模型和梯度精度继承原FP32设置。内积、平方范数采用**分块转换为FP64后累加**，不需要把模型改成FP64，也不必展平并复制整个解码器。

门控使用裁剪前、AdamW处理前的梯度。不要分别对BC与IFA梯度做归一化、逐层缩放或裁剪后再计算符号。

最终仍然执行：

\[
\text{选定损失}
\rightarrow
\text{原loss scale}
\rightarrow
\text{一次正式backward}
\rightarrow
\text{原全局梯度裁剪}
\rightarrow
\text{一次AdamW}
\rightarrow
\text{原调度调用}.
\]

**不额外执行优化器，不恢复参数，不修改AdamW矩状态。**关门只表示本步没有新的IFA梯度注入，历史IFA对AdamW动量和模型参数的影响不会被清除。

## 5. 分布式边界

已有训练启动代码使用DDP包装，原损失路径直接调用底层`module.compute_loss`。本版继承单GPU运行方式，并明确检查`world_size == 1`。 

PyTorch明确提醒，不能依赖DDP为`autograd.grad`提供其通常的梯度同步行为。因此，本轮门控对底层模型前向图求梯度，并在实际启动器下完成验收；**多卡全局梯度门控不属于G1-v1，不允许各卡独立决定后直接混用。**[PyTorch Docs](https://docs.pytorch.org/docs/2.13/generated/torch.nn.parallel.DistributedDataParallel.html)

## 6. 控制流程伪代码

下面是待接入项目的逻辑，不是仓库已有接口名称：

```text
加载原OBJECT预训练模型
关闭E1、P1
建立G1独立运行目录和合法初始缓存
固定共享解码器参数清单

依次学习task6、task7、task8、task9：
    沿用原数据加载器、优化器与调度器初始化

    对100个epoch中的每个原训练batch：
        按原方式清零梯度

        执行一次原训练前向
        保存bc_loss
        按原顺序构造native_loss
        另行收集本次原加权IFA项aux_terms

        如果G1关闭：
            selected_loss = native_loss

        否则如果原IFA没有贡献或hinge全零：
            selected_loss = native_loss
            记录不适用原因

        否则：
            aux_loss = 原加权IFA项的总贡献

            g_bc  = bc_loss对整个可训练解码器的梯度
            g_ifa = aux_loss对同一参数集合的梯度
            # retain_graph=True；create_graph=False
            # 不写入正常训练.grad

            检查梯度连接和有限性
            d = FP64分块累加的梯度内积
            w = 1，当d >= 0；否则w = 0

            selected_loss = native_loss，当w=1
                            bc_loss，当w=0

            记录门控、损失与梯度摘要
            释放不再需要的梯度探测张量

        对selected_loss执行一次原正式反向
        原梯度裁剪
        一次原AdamW更新
        一次原定调度调用
        保存日志，不保留跨batch计算图

    按原规则保存检查点、更新缓存和恢复状态

对四阶段ep99检查点执行原主闭环评估
从200次执行记录重建指标矩阵并汇总
```

---

# 五、代码验收：先排除实现偏差，再直接完整训练

本轮不增加独立离线研究实验，但需要完成少量批次和单元测试。这些是工程验收，**不是重新跑一条MLR＋IFA基线训练**。

## 1. 验收清单

| 验收项 | 通过标准 |
|---|---|
| **G1关闭** | 与原`observe`的损失、更新、优化器、调度和随机状态一致 |
| **强制全开** | 即使执行梯度探测，最终仍与同起点的原BC＋IFA更新一致 |
| **强制全关** | 与同起点、同优化器状态、同批次的BC-only更新一致 |
| **门控符号** | 人工构造同向、反向、正交及零梯度情况，开关符合公式 |
| **原损失不变** | 原BC、IFA筛选、参考、mask、归一化和累加顺序未改变 |
| **梯度隔离** | 两次探测前后，正常训练`.grad`未被额外累积 |
| **参数边界** | 门控只使用规定解码器参数；原冻结参数不更新 |
| **共享图** | 两类梯度和正式反向来自同一次原前向 |
| **更新次数** | 每基础步骤一次AdamW和原定调度，不重复 |
| **无IFA路径** | 不运行不必要探测，不残留上一批次门控 |
| **状态与显存** | 无随机流变化、模型状态污染或跨batch计算图累积 |
| **恢复与评估** | 断点恢复一致，评估加载的是实际G1主检查点 |
| **完整汇总入口** | 用固定测试矩阵验证结果写出、未舍入比较和表格生成 |

“强制全开／全关”只用于验收，不作为正式实验的额外训练分支。

对可以逐位比较的状态采用逐位校验；浮点核验需要容差时，在正式运行前固定并记录具体容差与最大误差，不能在发现不一致后临时放宽直到通过。

## 2. 第一阶段是自然检查点，但不是性能门槛

按原流程，task6没有需要施加的旧任务IFA，G1不应产生实际门控干预。可直接核对已有可用的初始化、输入和状态记录，不因此新增闭环试验。

**若第一阶段与历史结果不同，不得将差异解释为G1作用，也不得挑选一个成功率更接近历史结果的重复运行。**

## 3. 工程错误与负结果分开处理

缺参数、梯度断开、非有限数值、数据越界或汇总不一致属于工程问题，应保存失败记录并修复。

AUC不提高、门控关闭率高或新任务表现差属于实验结果，不能作为同版本中途改阈值、改任务对或挑检查点的理由。

---

# 六、正式运行、闭环评估与seed追加规则

## 1. 运行前固定输入

以提交`89e9936`所记录的原训练组件和历史协议为核对起点，创建G1独立实现与报告目录。新协议保存以下校验值：源代码、预训练检查点、特征清单、基础配置、共享参数清单及原指标程序。

如果服务器路径已迁移，通过文件校验值确认身份；不要仅凭同名文件认定是相同初始化。

**关闭P1不仅是把 \(\beta\) 设为零，还要确保不生成或加载教师目标；关闭E1则不再调用试更新控制器。**

缓存沿用原规则，模型结构、输入状态模态和推理逻辑均不改。

## 2. 首轮只运行G1的seed 42

完整执行四阶段，每阶段100 epochs，主模型均为`ep99`。

阶段间继续使用本次G1参数；各阶段优化器和调度器是否重新初始化，严格继承冻结基线规则，不自行决定。

允许保存原流程中的其他epoch检查点，但它们不进入主筛选，不进行事后最佳epoch选择。

## 3. 闭环评估保持200次

| 主检查点 | 评估范围 | 执行次数 |
|---|---|---:|
| task6／ep99 | task6 | 20 |
| task7／ep99 | task6、7 | 40 |
| task8／ep99 | task6、7、8 | 60 |
| task9／ep99 | task6、7、8、9 | 80 |
| **合计** | **10个有效矩阵单元** | **200** |

这是已冻结项目的主评估范围，不额外把预训练任务加入主指标，不追加680次辅助选检查点评估。

推理时不运行G1门控。模型前向与动作执行方法不变，任务ID也不因G1成为新增的推理输入。

使用已有修复后的评估启动方式，先做导入和检查点加载测试。环境执行初态、随机控制和终止规则沿用主协议，不根据结果更换初态或补做失败试验。

## 4. 仍使用原指标公式

以 \(R_{ij}\) 表示学习第 \(i\) 个增量任务后，对第 \(j\) 个任务的成功率，\(i,j=1,\ldots,4\)：

\[
\mathrm{FWT}=\frac14\sum_{j=1}^{4}R_{jj},
\]

\[
\mathrm{NBT}
=
\frac13
\sum_{j=1}^{3}
\frac1{4-j}\sum_{i=j+1}^{4}(R_{jj}-R_{ij}),
\]

\[
\mathrm{AUC}
=
\frac14
\sum_{j=1}^{4}
\frac1{5-j}\sum_{i=j}^{4}R_{ij}.
\]

这是附件1的指标定义。实现直接复用已经验证的汇总程序，不能把最终成功率替代AUC，或换用其他论文的NBT分母。Yu 等 - Lifelong imitation learn…

## 5. 追加seed条件

首轮计算：

\[
\Delta\mathrm{AUC}_{42}
=
\mathrm{AUC}_{G1,42}
-
\mathrm{AUC}_{\mathrm{历史IFA},42}.
\]

| 结果 | 处理 |
|---|---|
| 未舍入的 \(\Delta\mathrm{AUC}_{42}>0\) | 冻结G1-v1，追加43、44 |
| \(\Delta\mathrm{AUC}_{42}\le0\) | 当前配置不通过，不自动追加 |
| 只有NBT或FWT改善，AUC未提高 | 不通过总体收益筛选 |
| 单任务下降，但总体AUC提高 | 不单独否决 |
| 门控或损失诊断良好，闭环无改善 | 不认定有效 |

使用原始成功率矩阵重建的未舍入值比较，不用表格中的三位小数判断。

三seed阶段只增加G1运行，分别与历史对应seed比较；保留全部结果，报告均值、标准差和逐seed差值。**单seed正向仅是投入筛选，不是稳定收益或统计显著性的证明。**

---

# 七、日志与计算成本必须记录什么？

## 1. 门控日志区分“原IFA不适用”与“G1主动关闭”

每次更新记录以下信息；不适用时字段使用明确的`null`与原因，而不是填假零混入统计。

| 类别 | 字段 |
|---|---|
| 身份 | seed、task、epoch、更新编号 |
| 批次 | 当前／旧样本数量，原IFA参与样本数量 |
| 原目标 | BC、各原IFA加权项、原总损失 |
| 门控 | 是否有原IFA、hinge是否为零、\(w_t\)、决定原因 |
| 梯度 | \(d_t\)、两类解码器梯度范数、余弦、未连接参数数量 |
| 实际训练 | 实际采用的IFA贡献、最终损失、原裁剪前总梯度范数、学习率 |
| 成本 | 梯度探测、内积计算耗时及峰值显存 |

余弦仅作为诊断：

\[
c_t=\frac{d_t}{\|g_{\mathrm{BC}}\|\|g_{\mathrm{IFA}}\|}.
\]

零范数时记为不可定义；**门控使用内积符号，不用人为加分母常数后的余弦重新判断。**

统计时至少分别给出：全部更新数、原IFA构造次数、非零IFA次数、有效方向判断次数、主动关闭次数、零梯度次数。关门率必须标注分母，不能将“原来就没有IFA”算成G1成功过滤。

## 2. 记录平均IFA削弱程度，为以后解释结果留证据

可以报告：

\[
R_{\mathrm{IFA}}
=
\frac{\sum_t w_t A_{\mathrm{IFA},t}}
{\sum_t A_{\mathrm{IFA},t}},
\]

分母为零时记为不可定义。

该值描述**G1自己训练轨迹上保留了多少原拟施加的IFA标量贡献**，不等于相对基线减少了多少参数作用，也不是基线反事实。

如果G1有效，仍需要在后续区分“选择性关闭有效”与“整体IFA变弱有效”。本轮只保留这项诊断，不提前加入固定弱IFA或随机门控训练分支。

## 3. 计算成本：没有额外前向，不等于没有额外反向

本版在需要门控的步骤中增加：

> 一次BC到解码器的梯度探测，一次IFA到解码器的梯度探测，以及内积归约。

随后仍然进行一次正式反向。因此，虽然AdamW每步只更新一次，**计算图最多会经历三次相关反向遍历**，不能称为原训练成本不变。

设原步骤成本为 \(C_{\mathrm{native}}\)，需要探测的比例为 \(p\)，则附加成本可概括为：

\[
C_{\mathrm{G1}}
\approx
C_{\mathrm{native}}
+
p\left(
C_{\mathrm{BC\ probe}}
+
C_{\mathrm{IFA\ probe}}
+
C_{\mathrm{dot}}
\right).
\]

这是成本结构，不是实测倍率。记录完整训练墙钟时间、闭环时间、探测耗时和峰值显存；CUDA局部计时应使用能够正确处理异步执行的计时方式。

G1不新增持久教师或样本缓存，但需要暂存两组解码器梯度并保留计算图。及时释放临时张量，禁止把梯度向量或有计算图的loss保存在跨batch日志中。

已有可比基线耗时就复用；没有则只报告G1实测成本，不为测速重跑基线，也不套用E1或P1的开销比例。

---

# 八、配置、恢复与最终交付

## 1. 固定新增配置

以下是**待新增并接入项目的配置规范，不是仓库现成参数**：

```yaml
experiment:
  name: "G1-OBJECT-v1"
  benchmark: "LIBERO_OBJECT"
  tasks: [6, 7, 8, 9]
  seed_order: [42, 43, 44]
  first_seed_only: true
  baseline_rerun: false
  pretraining_rerun: false
  feature_regeneration: false
  initial_buffer: "empty"

e1:
  enabled: false

p1:
  enabled: false

g1:
  enabled: true

  main_objective: "native_mixed_batch_bc"
  auxiliary_objective: "native_weighted_ifa_contribution"
  shared_parameter_module: "temporal_transformer"
  shared_parameter_filter: "requires_grad"

  gate_type: "binary_global_batch"
  gate_rule: "dot_ge_zero_keep_ifa"
  dot_threshold: 0.0
  zero_gradient_action: "keep_native_ifa"
  nonfinite_action: "stop_and_save_evidence"

  decision_frequency: "every_nonzero_native_ifa_step"
  gate_history: false
  per_task_gate: false
  per_pair_gate: false
  per_layer_gate: false

  forward: "single_native_training_forward"
  probe_retain_graph: true
  probe_create_graph: false
  probe_allow_unused: true
  dot_reduction_dtype: "float64"

  write_probe_gradients_to_param_grad: false
  manual_gradient_mixing: false
  parameter_projection: false
  post_optimizer_correction: false
  optimizer_modification: false

  supported_world_size: 1

evaluation:
  primary_view: "PRIMARY_EP99"
  trials_per_task_stage_cell: 20
  expected_primary_trials_per_seed: 200
  secondary_checkpoint_selection: false
  primary_metric: "overall_auc"
  diagnostic_metrics: ["overall_nbt", "overall_fwt"]
  per_task_performance_gate: false
```

原IFA权重、gamma及其他基础参数从冻结配置继承，不在G1节点中重复维护另一个版本。

## 2. 断点恢复

恢复文件保存：模型、原优化器／调度器、阶段与epoch、更新计数、正常训练随机状态、合法缓存清单、门控统计累计量、代码与协议校验值。

G1没有教师库，也没有跨步门控状态。恢复后重新计算下一batch的门控，不能沿用中断前某个 \(w_t\)。

中断发生在未提交epoch时，使用原已验证恢复规则保留失败尝试记录；不能重复提交同一epoch或把不完整执行并入主结果。

## 3. 建议交付内容

新建独立的`g1_object_v1/`实现目录，以及`reports/g1_object_v1/`报告目录；原代码与历史结果只读保留。至少交付：

| 文件 | 内容 |
|---|---|
| `PROTOCOL.json` | 基础协议、门控定义、固定参数、seed追加条件 |
| `IMPLEMENTATION.json` | 实现提交与受保护原组件校验值 |
| `ENGINEERING_CHECK.json` | 全开／全关、符号、状态与梯度验收 |
| `PARAMETER_MANIFEST.json` | 门控参数集合及原可训练参数集合 |
| `RESULT_seed42.json` | 原始成功率矩阵、未舍入指标、历史参考与差值 |
| `MECHANISM_seed42.json` | 分阶段及总体门控统计 |
| `EXECUTION_COST.jsonl` | 训练、探测、评估及失败尝试成本 |
| `FINAL_VERIFICATION.json` | 完成性、数据合法性和独立汇总核验 |
| `FINAL_REPORT.md` | 总体结果、成本、机制及解释边界 |

最终验收确认：**400个完整阶段—epoch记录、4个主检查点、200次主评估、10个矩阵单元、指标独立重建一致，以及没有运行未获条件授权的额外seed。**

---

## 本轮结论应如何表述

G1即使有效，也只能先支持：

> **在当前OBJECT协议中，根据BC与IFA的局部梯度关系控制辅助损失，改善了总体闭环表现。**

不能直接升级为“识别了真正的共享知识”“每步避免遗忘”或“IFA负内积时必然有害”。AdamW、有限步长、随机批次与闭环状态变化仍然限制这些推论。

**最终执行顺序：冻结原基线输入 → 只实现G1门控并完成工程验收 → 跑OBJECT／seed 42完整四阶段 → 完成200次主闭环评估 → 总体AUC高于历史基线后，再补G1的43、44。**