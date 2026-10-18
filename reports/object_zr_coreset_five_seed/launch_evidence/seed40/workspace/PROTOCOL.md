# Coreset＋IFA＋指令零响应残差：LIBERO-OBJECT 单 seed 闭环执行规范

版本：v1；状态：执行设计，未在本交付环境运行新训练或 LIBERO 闭环。
首轮只新增 `coreset_ifa_zr` 一组，seed42；复用已完成的 `coreset`＋IFA 主基线，不重训、不重评基线，不自动扩展其他seed。

前提来自用户确认：真实指令与真实策略原型已通过。正式执行时必须绑定实际 `PROTO_FINAL.json`、原型源码、gate向量与输入哈希；本包中的设计文件和合成测试不能代替真实原型验收。已经通过的200步原型不需要重复。

## 1. 科学问题与协议边界

问题：保持相同 Coreset 样本身份、载荷、文件顺序、每任务配额、当前数据曝光、训练轮数和 BC＋IFA 规则，给每个增量任务加入一个在历史固定指令上零响应的新动作残差通道，是否改善闭环持续学习表现？

本轮恢复共享 Temporal Decoder 与公共 GMMHead 的训练。原型“共享网络冻结、仅训练残差”是结构隔离试验，不是本轮训练法。

新增残差不根据损失、梯度冲突、不确定性、事件标签或闭环结果决定优先级。梯度仅用于正常 BC＋IFA 学习。Coreset的既有视觉—动作距离完整保留，且本轮不更改成员。

本轮是同一父实现下、相同数据和优化预算的**模型模块对照**，不是“只替换入库选样”，也不是严格相同参数量或计算量。附加模型参数、门向量、控制时延必须计量。单seed即使提升，也不能独立证明零响应设计优于等参数量普通残差；这一机制对照留给后续，不纳入当前预算。

## 2. 固定来源、主基线与验收范围

仓库：`uhhj/lifelong_mlr_ifa_minrep_20260821_193226`。
读取时 `MLR` 分支：`1ed6e0dd25cb5748ee5e887444619483bd9bdc31`。
父源码：`experiments/object_coreset_seeds42_43_44/seed42/source`。
父运行：`/data/mlr_ifa/experiments/Coreset_paper_OBJECT_seed42_20261013`。
官方起点：父合同绑定的 `object_multitask_model_ep99.pth`，不能换成已学 Task6 或 Task9 的模型。

主基线仅使用该seed42首次Coreset＋IFA的200条已归档正式记录：

|指标|精确分数|百分数|
|---|---:|---:|
|AUC|109/160|68.125000|
|FWT|13/20|65.000000|
|NBT|-7/120|-5.833333...|
|OldFinal|11/15|73.333333...|
|FinalAll|53/80|66.250000|

成功率矩阵，行为完成阶段6–9、列为任务6–9：

```
30   —   —   —
70  85   —   —
20  95 100   —
30  90 100  45
```

Random＋IFA seed42只作已有辅助参照，不用于降低主门槛。不得使用Event+Coreset、IFA-off、state=False旧基线、seed43重复运行或五seed均值替换主基线。

执行前写 `REUSE_AUDIT.json`：核验父源文件/数据/模态/权重/缓存/初态/评估器/训练配置身份。当前GPU、驱动和软件版本必须实测，不把历史报告中的RTX3090当作当前机器事实。公共数值设置发生无法解释的变化时，不标严格配对复用；暂停正式准入并报告，不能擅自重训基线，也不能把跨环境探索包装成纯模块因果对照。

## 3. 不变的训练、数据和预算

|项目|正式设置|
|---|---|
|套件/顺序|LIBERO_OBJECT，task_order_index=0，6→7→8→9|
|训练seed|42；不是论文正文列出了具体seed列表|
|当前示范|父清单同10条；不重新选demo、不加入6个基任务动作训练|
|空间前端|官方冻结权重、原LoRA、vision/language/state均启用|
|缓存|原seed42、state=True的同一份空间latent；不重新导出|
|shape|feature [1,8,4,512]；actions [1,8,7]|
|训练样本|原当前全集＋合法旧Buffer；同文件名和拼接/打乱规则|
|BC监督|每条窗口只监督原末步专家动作；不增加多时刻或残差真值|
|epoch/batch|每任务100完整epoch，batch10，num_workers0|
|AdamW|lr1e-4，betas(0.9,0.999)，eps1e-8，weight_decay0.1|
|调度|原线性Lambda，warmup500，按optimizer step；每任务重置优化器/调度器|
|裁剪|对实际可训练的共享＋当前残差参数整体裁剪max_norm100；loss_scale1|
|IFA|权重0.1，gamma0.3；原任务对和平均角度损失；输入仍为共享last表示|
|checkpoint|固定epoch_index99训练完成状态，不按成功率选best|
|分布式|单GPU、单进程，保留父任务数据调度；模块注册见第7节|

新增残差用同一lr/调度/weight_decay，不继承原型的“无warmup、weight_decay0”。这是明确的正式实验选择，不搜索残差专用学习率、不加adapter warmup或残差预训练。

数量：

|任务|当前N|训练前旧Buffer|每epoch样本数|每epoch批次|100epoch更新|本任务入库|入库后总数|
|---|---:|---:|---:|---:|---:|---:|---:|
|6|1562|0|1562|157|15700|781|781|
|7|1480|781|2261|227|22700|740|1521|
|8|1560|1521|3081|309|30900|780|2301|
|9|1319|2301|3620|362|36200|660|2961|

累计400epoch、105500次更新。若实际长度不符，先报告，不通过修改drop_last、重复样本或删样本凑数。

0.5是入库存储比例，不是每个batch一半旧样本。原任务混合比例、尾批大小、DistributedSampler默认seed0与set_epoch(epoch)不变。

## 4. Coreset成员与顺序精确固定

只在当前任务训练结束后执行原Coreset入库。选择依赖冻结空间特征与当前真实动作，不依赖本轮已变化的时序网络，因而应与父Coreset得到相同成员。

可直接读取经过哈希验证的父 `coreset/storage/task{k}.json` 作为固定选择结果，在该任务结束时复制其原载荷到新Buffer。另在CPU只读核验原选择函数、输入和对应结果身份；这不属于基线训练或评测。也可重新运行同一个选择器后与父结果逐ID/顺序对齐，不能接受“数量相同但成员不同”。

约束：同源文件ID、同目标文件名、同tensor字节、同排序、同每任务数量；不从本轮训练后的z重新选样、不执行Event、不重排共同成员。选择结果不一致时停止，不强制替换为更有利的一版。

每阶段先保存入库预提交清单，后原子复制，再校验hash与任务计数。旧成员不得改变。总容量沿父文件数阈值59660000；本轮不应触发裁剪，意外触发即停止。

训练Task k只能读取当前任务全集和已提交的旧Buffer。父完整旧缓存仅供离线来源审计，不能进入后续梯度训练。未来任务门/选择清单即使已在磁盘上，也不能提前进入策略或优化。

## 5. 正式模型和学习规则

设缓存H提供8步多模态历史，

`z = T_theta(H)[:, -1, :]`，512维；
`e_raw = H[:, -1, 0, :]`，来自冻结语言前端而非时序z；

公共头原始输出为 `eta_base(z)`，75维，布局严格为35均值前激活、35尺度前激活、5 logits。

在完成到Task k时：

`eta = eta_base(z) + sum_{j=6..k} g_j(e_raw) * U_j @ V_j @ z`。

g按已验收方案只归一化语言输入，FP64内积，乘FP32残差先提升FP64再将贡献转回FP32。U[75,8]、V[8,512]，无bias；初始U=0，V~N(0,0.02²)，使用局部CPU generator seed42+k。不把整个模型 `.float()`/`.half()` 导致gate降精度。

所有已有通道参与前向，不只启用当前通道。旧通道在未来指令上可以非零、可以为负，其门值不必在[0,1]。不引入softmax、sigmoid、clip、eps置零或Task ID硬路由。

每个新门保护0..k-1，包括0–5六个基任务的固定语言指令。历史门永不重新拟合或重正交化；构造每个门只能使用历史＋当前指令。门不根据BC/IFA训练，不读取未来指令优化门向量。

官方encoder/tokenizer和实际语言输入与通过的原型完全一致时，可复用其门向量和几何证据。**不能加载原型50步/通道训练后的U/V，不导入原型优化器状态。** 否则会给实验额外训练数据曝光。

### 5.1 每阶段的可训练参数

- 共享Temporal Decoder、公共GMMHead：训练；保留父dropout及train模式。
- 新任务U_k/V_k：训练，加入同一AdamW与同一调度。
- 旧U_j/V_j：requires_grad=False，grad=None，完全排除优化器。
- 语言前端、gate_w、视觉/状态前端：冻结。

**旧分支冻结权重，不冻结其对输入z的雅可比。** 必须正常计算R_j(z)，禁止`torch.no_grad()`或detach(z)包围旧分支。否则共享网络丢失一条真实梯度路径，算法已经改变。

也不能复用原型预计算的z：本轮只能复用冻结空间H，z须每个batch在可训练T中重新计算。

原型参考 `ResidualReadout.__init__` 会冻结公共head并置eval。正式接入必须删除这一原型专用行为，或改用不改变原 `policy_head` 注册名的函数式读出。要用参数白名单确认公共head与T都确实可训练，不能只看cfg。

### 5.2 单个batch

```
H, action_sequence, task_labels = parent_batch
z = temporal_encode(H)[:, -1, :]                 # 保留梯度
raw_language = H[:, -1, 0, :]                  # gate内部detach
raw = original_head_raw(z)
for branch in existing_branches_in_birth_order:
    raw += branch(raw_language, z)              # 旧权重freeze，但不detach z
p = original_GMM_from_raw(raw)                  # 原tanh / scale activation / logits
L_BC = mean(-p.log_prob(action_sequence[:, -1, :]))
L_IFA = parent_angular_mean(z[current_mask], own_ref, selected_old_refs, gamma=.3)
L = L_BC + .1*L_IFA
zero_grad()
backward(L)
clip_grad_norm(shared_and_current_branch, 100)
optimizer.step()
scheduler.step()
```

每个epoch曝光和训练次数不变。正常损失用于训练与有限性诊断，不根据loss变化调整分支、样本、权重或早停。

IFA仍作用在共享时序表示；不对75维动作参数加IFA。由于Buffer成员/顺序与当前空间缓存相同，IFA配对及语言reference应和父C组相同；不匹配应查实现，不将其当算法新作用。IFA具体损失值随共享表示训练变化是正常现象。

## 6. 参数时序与正确的继承

|阶段|共享部分|旧残差|新残差|训练后保留|
|---|---|---|---|---|
|6|从官方六任务起点训练|无|新R6|共享θ6/ψ6、R6、gate6|
|7|继承本轮θ6/ψ6|R6冻结|新R7|本轮共享状态、R6/R7及门|
|8|继承本轮上一阶段|R6/R7冻结|新R8|0..当前实际存在的模块|
|9|继承本轮上一阶段|R6/R7/R8冻结|新R9|四个残差及共享状态|

每阶段重置优化器/调度器，不重置共享网络，不载入父Coreset的阶段模型。R0–R5不新建；对基任务的零响应由每个增量门的历史约束提供。

**Task6的新残差从其第一个训练batch开始存在。** 零初始化时输出与原策略一致；开始学习后Task6模型可以不同于历史Coreset，不能套用Event实验的“Task6训练完成后必须逐参数一样”断言。需要相等的是新任务开始前公共起点，以及首次更新前零分支的函数输出。

同理，新的R9在Task9训练中有效；Task9训练结束后才写入的Buffer不会影响这次成绩。保存模型和记忆的时间顺序要分开。

## 7. DDP、优化器和checkpoint集成

父程序build时已包装DDP，并在每个任务重建optimizer/scheduler。不能向既有DDP包装器追加参数。单卡版本采用：

1. 阶段边界取出底层模型，释放旧DDP包装引用；不重建或重载共享网络。
2. 冻结历史分支，注册当前分支与门；设置共享/当前分支可训练。
3. 检查参数名单，使用单GPU单rank重新包装DDP。
4. 构造只包含共享T、公共head和当前U/V的AdamW；继承原超参数和调度。
5. 建构前后公共/旧分支tensor必须相同；新增随机初始化使用局部RNG，不改变父全局随机状态。

代码中的可训练白名单由 `temporal_transformer.* / policy_head.*` 扩展到**当前** `residual_bank.{k}.U/V`；不能宽泛地允许全部residual_bank重新训练。

最小化改动参数注册名：原 `policy_head.share/mean_layer/logstd_layer/logits_layer` 名称保留，新分支单独 `residual_bank`。不要把同一个head在两处重复注册，亦不能未经映射将head挪进一个新wrapper导致原checkpoint部分未加载。

任务模型保存：公共下游、当前已存在全部U/V和FP64gate、分支出生顺序、rank与raw-layout、父frontend/feature合同、新源码/模型合同、训练cursor。

恢复checkpoint额外包含：optimizer、scheduler、global step、CPU/CUDA/Python/NumPy RNG、模块模式、梯度、活动分支清单、Buffer哈希、task-end storage事务。中断后按原next_epoch恢复，不重新抽样、不重复计数；部分未提交日志另存。stage完成状态恢复时先重建相同注册/可训练集合，再加载状态。

旧严格加载器只接受公共T/head。新评估器必须扩展为：
- 官方CompletePolicy前端先完整严格加载；
- 只注册该checkpoint已有的R6..Rs，禁止未来R；
- 对公共T/head＋residual_bank的期望键集合/shape/dtype逐项完整匹配并核验实际tensor；
- 缺一个门或U/V都报错，不得strict=False静默漏掉新模块。

不能强求Brief checkpoint含完整视觉/状态前端。前端按官方hash独立加载，训练下游与分支按明确前缀严格覆盖。

## 8. 数值设置与历史基线复用

原型采用更严格全局确定性设置。正式比较必须分别恢复父训练、父评估的实际数值模式，不能让原型进程环境变量、deterministic算法开关或benchmark=False泄漏到新正式运行而仍声称完全同协议。

- gate和乘法精度继续按已通过的FP64/FP32合同。
- 原策略FP32、无AMP；global TF32/cuDNN/CUBLAS/deterministic配置从父执行证据和代码核实并记录。
- 父evaluate明确将cudnn.benchmark=True，本轮主评估保持；不能只对新组改成全局确定性或均值动作。
- 在这个**正式实际模式**下做一次集成前向数值检查（真实10-lane布局、历史长度1/4/8、缓存语言与在线语言）。不重跑原型200步。
- 若无法同时满足门精度与历史协议，记录阻塞。不得自动放宽gate阈值，也不得悄悄改公共精度。若未来要更改公共数值协议，须另行决定同协议基线，本轮不擅自扩展。

单seed/相同初态/相同seed不保证相同行为噪声全过程，尤其原全局采样流会随着终止步数改变。此次只保持原科学协议，不声称逐轨迹噪声已配对。

## 9. 集成检查：只做必要连接验收，不再进行整套前置训练

一次性隔离进程，最多4次优化器调用、0环境step/rollout；全部从克隆状态执行，正式训练另起进程从官方起点开始。

- prototype实际PASS/门/源码/encoder身份绑定；rank/raw布局相符。
- 零初始化与原head输出一致；残差关闭路径在相同batch/RNG下的公共前向和梯度与父实现一致。
- 共享T/head和新U/V梯度路径连通。warmup首个lr可能为0，不能因为第一步参数未动便判失败；记录实际lr与梯度，不跳过warmup。
- 旧U/V不在optimizer、更新后tensor完全不变，但旧分支对z梯度路径仍存在。
- 单任务/混合batch/没有当前样本的batch均有限；不按Task ID硬关门替代数值g。
- 阶段注册和DDP重包不会改变公共/旧tensor、全局RNG；优化器包含新分支。
- 保存快照后一次更新、恢复快照并复现同一更新计入4次调用上限；公共/新分支/optimizer/scheduler/RNG结果一致。
- Complete原始观测路径与同一H的Brief路径分布参数一致；不比较不同增强图像。
- 最终模型重载包含正确通道与FP64门。

不把这几次集成更新计入400epoch，不让其权重/动量/RNG进入正式训练。

## 10. 正式过程中的结构诊断

共享网络已解冻，不能再比较“旧任务训练前后整个输出必须不变”。正确的直接检验是在**同一时刻相同共享参数**上比较新通道开/关：

`eta_all(H,e_old) - eta_without_new_channel(H,e_old)`。

逐阶段检查当前通道对全部旧指令的gate与raw贡献；检查累计效果时，对旧任务j只在诊断克隆里关闭出生任务>j的通道，对比所有通道开启输出。不得在真实策略中使用这一Task ID诊断mask。

继承原型容差：旧gate最大1e-6，当前gate误差1e-5；旧指令新增/累计raw贡献≤1e-6，分布参数atol=rtol=1e-6，选中GMM分量一致。比较时共享theta不变，使用同一固定噪声；这些读数检查零响应本身，不预测成功率。

固定小探针每10epoch检查；epoch99对合法旧Buffer全扫贡献。旧基任务0–5仅用此前批准的固定前向探针或语言向量，不增加基任务动作训练。门和旧U/V每阶段前后逐tensor相等。

长训练中||UVz||可能增大，即使gate很小也可能泄漏。若超出批准容差，应停止并保留数值失效，不夹断gate/残差、不改阈值。若只是整个旧策略输出因共享参数变化，不能误报门失败；这由闭环衡量。

全部诊断在eval克隆或保存/恢复模式与全局RNG的无更新路径执行。不能因为诊断消耗随机数影响后续batch或dropout。

记录每分支贡献、固定旧指令gate、活跃分支范数和非零更新只是排错/机制观察；不据此调整学习率或回放。最终残差几乎未启用时照实报告，不额外训练直到出现作用。

## 11. 正式闭环评估

只评估4个最终epoch99状态：stage6评task6；stage7评6/7；stage8评6/7/8；stage9评6/7/8/9。

沿用父评估器：10-lane、每cell两组、初态0..19各一次、环境seed42/43、启动5步零动作、最多600个策略步、原成功判定与向量组终止方式。每stage按父代码reset42，stage内部不新增每task/每trial reseed。原GMM选argmax logits分量，再从该Normal分布sample；不改为完整Categorical采样、不改均值执行、low_eval_noise=False。

外部eval_metric已经维护长度最多8的原始观测队列，再调用get_action。新增get_action继续对传入序列编码，不能又增加第二个内部历史队列。

两条读出路径必须共用raw GMM＋残差：
1. 训练：原空间cache H → Brief时序z → 新读出。
2. 执行：原raw观测预处理 → frozen spatial H → Complete时序z → 相同新读出。

真实推理gate来自spatial_encode的语言token，不使用评估task_id选分支。新模块初始化、载入和probe不能额外消耗父动作采样的global RNG；评估前记录RNG摘要，但不承诺因策略轨迹不同而全过程摘要相同。

严格恢复stage已有分支，原官方前端完整加载＋所有下游/残差键逐tensor核验后才创建正式环境。记录每个trial的seed/stage/task/epoch99/初态SHA/模型SHA/modal SHA/residual合同SHA/成功bool。父modal合同继续代表相同空间前端；新模型合同分开保存，不改写父modal来源。

每组200个新正式episodes，20个vector attempts（10cells×2组），不得把10-lane当独立训练seed。旧C的200条仅读取。

无训练中闭环择优；建议训练4阶段完成后在独立进程统一评估。常规失败不重试；技术中断保留attempt账目并按原RNG恢复规则审计，不能只重跑失败trial拼满20条。未完成前不汇报最终AUC。

## 12. 指标、预先冻结的推进条件

r(s,t)为学完s后在t上的成功率，计算：

AUC = (1/4) sum_{t=6..9} [sum_{s=t..9} r(s,t)/(10-t)]
FWT = (1/4) sum_{t=6..9} r(t,t)
NBT = (1/3) sum_{t=6..8} [sum_{s=t+1..9}(r(t,t)-r(s,t))/(9-t)]
OldFinal = [r(9,6)+r(9,7)+r(9,8)]/3
FinalAll = (1/4)sum_{t=6..9}r(9,t)

用成功次数/20精确分数计算，不直接等权平均10个矩阵单元。NBT保留符号，不能裁剪负数；FWT采用父定义，不把它改叫零样本迁移。

主比较CR−C：
- AUC > 109/160；
- FWT >= 13/20；
- NBT < -7/120；
- **新增绝对保持护栏**：OldFinal >= 11/15。

前三项沿用前面筛选门槛；第四项是本轮在看新结果前加的设计，避免以较低的初学锚点换取NBT好看。FinalAll、每个旧任务最终成功率和完整矩阵必须报告。

门槛只是保守的资源推进条件，不等价于所有合理效用函数；不通过也可能是学习/保持取舍，而不是算法毫无价值。要保留所有结果且不事后放宽门槛。首轮通过不能宣布统计显著、保证无遗忘或证明零门优于同参数量普通adapter。其他seed需后续明确授权，不自动运行。

随附 `compare_metrics.py` 从新旧原始JSONL复算并验证seed/epoch/初态与前端modal配对。它固定核验父seed42矩阵、不会执行环境或验收训练代码，输出 `METRICS_RECOMPUTED_NOT_FULL_ACCEPTANCE`；最终通过还需独立工程审计。

## 13. 成本与工件

新增预算：1个完整训练组、400epoch、105500有效optimizer steps、4个最终模型、200条闭环。集成≤4步另列；不增加基线或Event组、不增加原型200步、不做额外残差预训练、不搜索rank或学习率。

每分支参数8×512+75×8=4696；四分支18784个FP32标量即75136B（约73.375KiB）。四个512维FP64门16384B；若保留10×512的FP64规范指令表再计40960B。以上不含序列化、optimizer状态和日志，应同时报告实测数值。

Buffer载荷字节不变，但总体模型存储增大，不能声称总记忆完全相等。报告新增训练墙钟/显存、完整评估时长、get_action延迟，诊断/前置/选样开销分别列出。

最低工件：
- `PARENT_BINDING.json`, `PROTOTYPE_BINDING.json`, `INPUT_SHA256.json`, `REUSE_AUDIT.json`;
- `CONTRACT.json`, 原样 `MODAL_CONTRACT.json`, 新 `MODEL_CONTRACT.json`, resolved train/eval配置;
- `GATES_AND_PROTECTION.json`, 新分支出生与初始化摘要，公共初始参数/RNG parity;
- 每阶段合法训练参数名单、optimizer参数名单、dataset顺序摘要、Buffer清单、IFA对及reference;
- epoch日志、global step、实际lr首末值、恢复checkpoint、4个最终checkpoint及所有分支状态;
- `DIRECT_CHANNEL_ISOLATION.jsonl`, `FROZEN_BRANCH_IDENTITY.json`, `NATIVE_PATH_PARITY.json`;
- 10个trials JSONL、attempts账本、各cell结果、RNG快照；
- 指标复算、成本、独立checkpoint加载验收、pipeline exit、最终状态。

不要push原始示范、模型权重、latent、Buffer载荷或凭据。只需按用户后续授权推送冻结源码、diff、合同及轻量报告；本任务只设计，不启动训练或写远端。

## 14. 实施步骤与开发入口合同

新增模块建议：`zrg_exp.prepare`、`zrg_exp.integration_check`、`zrg_exp.train`、`zrg_exp.evaluate`、`zrg_exp.finalize`，以及共用的残差读出和阶段注册逻辑。这些名称是**待在父环境实现/集成的入口**，不是现有仓库已经可运行的API。已通过的原型实现应作为源，不重新编造另一套门控。

目标命令：

```bash
export PARENT_RUN=/data/mlr_ifa/experiments/Coreset_paper_OBJECT_seed42_20261013
export PROTOTYPE_RUN=/absolute/path/to/actual_accepted_prototype
export ZR_SOURCE=/absolute/path/to/new_isolated_source
export ZR_RUN=/absolute/path/to/new_empty_run
export PYTHONPATH="$ZR_SOURCE:$PYTHONPATH"

# 以下入口须先按本规范完成集成，且不调用父pipeline重跑基线
python -m zrg_exp.prepare --spec "$SPEC" --parent "$PARENT_RUN" \
  --prototype "$PROTOTYPE_RUN" --out "$ZR_RUN" --reuse-baseline-only
python -m zrg_exp.integration_check --run "$ZR_RUN" --max-optimizer-calls 4 --no-env
python -m zrg_exp.freeze --run "$ZR_RUN"

torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  -m zrg_exp.train --run "$ZR_RUN" --arm coreset_ifa_zr
python -u -m zrg_exp.evaluate --run "$ZR_RUN" --arm coreset_ifa_zr

# 本包已经提供，可直接执行；不需要GPU/robomimic
python compare_metrics.py \
  --baseline-dir "$PARENT_RUN/coreset/evaluation" \
  --new-dir "$ZR_RUN/coreset_ifa_zr/evaluation" \
  --output "$ZR_RUN/METRIC_COMPARISON.json"

python -m zrg_exp.finalize --run "$ZR_RUN" --no-auto-extend
```

流水线应先冻结合同后执行；每命令记录start/exit，失败停止，正常成功率低不重跑。新run禁止覆盖历史路径；恢复须显式`--resume`及审计。

## 15. 最终结论模板

工程成功：完整新组已完成，Coreset成员/顺序/配额与父基线一致，门与旧分支保持批准的结构约束，新旧权重加载与记录完整。

科学结论：仅描述seed42下CR相对C的AUC/FWT/NBT/OldFinal/FinalAll和逐任务差，披露额外参数及共享网络可以遗忘。原型PASS不能代替这些闭环结果，也不能把新通道零贡献外推为任意新指令/任意共享网络更新的无遗忘保证。

### 依据（固定源码事实，不支持尚未执行的闭环收益）

- 父训练与损失：`experiments/object_coreset_seeds42_43_44/seed42/source/coreset_exp/train.py`，上述固定commit。
- 父优化边界：同source下`libero/lifelong/algos/base.py`。
- 原评估和记录：同source下`coreset_exp/evaluate.py`与`eval_metric.py`。
- 基线指标：`reports/object_coreset_seeds42_43_44/execution/20261013_coreset_extension/FINAL_REPORT.md`。
- 原型设计附件：`zero_response_residual_prototype/PROTOCOL.md`、`spec.yaml`及`residual_reference.py`；实际通过版本另由PROTO_FINAL绑定。
- DDP注册约束：https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html 。文档用于解释接口约束，不要求升级父PyTorch。
