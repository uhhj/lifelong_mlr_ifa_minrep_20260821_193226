# RA-Coreset（Reservoir-Anchored Coreset）完整执行协议 v1

**状态：设计与 CPU 合成测试；尚未集成原仓库、未读取服务器真实权重、未执行新增训练或环境评估。**

研究主张：在同一有限记忆配额下，将“全局最大几何覆盖”与“从原训练窗口总体均匀采样的真实数据支持”结合，使用确定的最小成员交换构造新的 Buffer；通过稳定槽位训练与独立逐轨迹动作噪声，把成员内容效应尽可能从无意的数据重排、评估 RNG 串扰中分离。只以真实闭环成功率判断有效性。

此方法不根据模型损失、梯度、动作不确定性、估计遗忘、训练后验证成功、人工关键事件或任何退化代理选择样本。几何距离是原 K-Center 本身的优化对象，不是被宣称为闭环表现的替代指标。训练与最终评估不增加模型参数或推理分支。

## 1. 为什么不能再照 SC 修改

冻结报告 commit `95503d6799b14862d5d5d21ab065c8ba252863b2`：
- OBJECT，原 Coreset−Random 五 seed AUC 平均 +5.8125pp。
- GOAL，原 Coreset−Random 三 seed AUC 平均 −3.020833pp，Task6 旧能力受损明显。
- Event＋Coreset 五 seed 相对 Coreset 平均 −4.291667pp。
- GOAL seed42 SC-v1 相对 Coreset −5.3125pp：精确联合监督比例带来终点重复和最大距离增大。
- GOAL seed42 SC-v2 相对 Coreset −9.6875pp：覆盖约束通过但仅 Task6–8 11条旧成员改变，进入 Task9 时 810条共同旧成员的位置改变、94.05% batch 内容改变；Task9 评估的 CUDA RNG 起点也不同。

不从上述数据推断任何特定 8步窗口“实际导致”失败；不据既有20条失败初态调算法超参。既有负结果保留，不作为 new-data 源。现有 OBJECT 的正结果也不是新算法必然有效的证据。

## 2. 唯一算法干预：随机锚点＋原 Coreset 的最少交换

在每任务 **训练结束后**，选样输入是当前任务原完整窗口 `U_t` 及已冻结视觉历史—末端动作距离，不会访问未来任务、策略预测或 rollout。

原父 Coreset 先运行 `greedy_kcenter_select` 得到 `S_C`，长度 q=ceil(N/2)。调用原 Random 入库同一排序、同一训练 seed 的 `random.Random(seed).sample(U_t,q)`，取前 a=floor(q/2) 个作为锚点 `A`。从无放回样本的均匀性可知：每个原窗口进入 A 的概率都等于 a/N。这只保证**随机锚点层**的期望比例，而不保证完整新 Buffer 的分布与原训练集完全一致。

随后不从头运行第二套 K-Center，保留原选择 `S_C`，为每个 `x in A - S_C` 在当前集合加入 x，从原被选集合的 `S - A` 中删除一条。删哪个？在同一原距离 D 下选择使新的全候选最大最近距离最小的 y，若并列再选全候选平均最近距离最小，最后原文件名确定。只取真实样本，绝无填充、合成或额外样本。

公式：

- `S ← S_C`，`A = RandomSample(U,q,seed)[:floor(q/2)]`
- 对 A 中每个尚未选中的 x：
  - `y* = argmin_{y in S\A} ( R((S+{x})\{y}), MeanD((S+{x})\{y}), filename(y) )`
  - `S ← (S+{x})\{y*}`，将 x 记录在 y 原有的 Coreset 选择槽中。
- 返回 |S|=q，必须包含 A。

这保证恰好 `|A\S_C|` 次成员交换，即纳入所有锚点所需的最少交换数量。不设置新的事件份额、时间/命令配额、终点复制目标，也**不设置 SC-v2 那样无法兼顾的强覆盖半径硬底线**。报告半径的变化，但不凭半径直接决定是否继续训练或报告成功。

原距离完整不变：只使用 `[8,2,512]` 视觉 token1/2、末端7维专家动作；视觉/动作权重各0.5；原 MAD/IQR/range 标定、4096标定对、256参考样本、FP64、selector_seed42、并列容差1e-12。全任务一次标定，不按随机/非随机部分重新标定。

`selector_reference.py` 是纯NumPy算法参考，只接受已经从父距离计算的 D 和原 K-Center S_C。接到真实缓存的加载与距离构造需要实现在新工作区，不可假称参考代码直接在原仓库已经可运行。

**边界**：原始 Coreset 可能恰好包含所有锚点，那一任务将严格退化为 S_C；GOAL 中每任务的实际成员差异须离线确认。换样不保证 GOAL 成绩改善，也可能降低 OBJECT 的优势。混合随机锚点＋核心集有大量先例，不能仅凭此称 CoRL 级原创；对论文的可主张差异是与 MLR 潜缓存配额、稳定槽位的受控学习研究组合，仍需相关工作和新多 seed 证据。

## 3. 同时冻结训练身份顺序：原目录名排序的影响必须隔离

冻结源码 `libero/lifelong/algos/mlr.py::build_dataset_for_task` 的真实顺序是：Buffer 所有 pt 文件按路径字符串排序，后面接当前任务按路径排序的完整 pt，调用独立 `random.Random(training_seed).shuffle(...)`，再由 `DistributedSampler(seed=0)`按 epoch 打乱索引。

单独在选样清单里“保留 Coreset 原选中顺序”不起作用：真正加载器会按新文件名重新排序。新 RA 组必须启用**只读参考槽位清单**：

1. 对每个已完成旧任务 j，从对应 seed 已完成 Coreset 原入库清单，构造该任务原训练器会看到的 `t{j}__*.pt` 的严格字典序槽位 `slot_C(j)`。
2. 将 RA 和 C 的共同成员绑定在相同 `slot_C` 索引上。
3. 按被移出的父 C 成员槽位顺序，把新 RA 成员（按稳定源 ID 排序）分配到这些空槽；不重命名真实 PT 文件，也不改变文件内部 task_id。
4. 在阶段 k，将所有旧任务的槽位按父全局排序顺序连接，之后拼接原完整当前任务 PT 列表；调用与父实现**完全相同**的局部 `random.Random(seed).shuffle`，再由原 `DistributedSampler(seed=0).set_epoch(epoch)` 执行。
5. `FileTrajectoryDataset` 仍加载原文件，`parse_path` 读真实 `t{j}__` 文件前缀；不更改模型、batch 的任务身份、当前任务位置、样本出现次数与步数。

这种设计仅让被换进、换出的槽位改变；保留下来的旧样本在 shuffle 之前位置相同。**这不是让所有 batch 不变**：任何含有被换槽位的 batch 仍会变化，模型的序列优化路径仍可能发散。

新组 `ra_exp/loader.py` 是待实现的 adapter，不能以输出文件名伪装为原身份。完整保存 `slot_index -> actual_pt_sha256, raw_source_id, task_id`，恢复时严格核验。旧 Coreset 比较组可以保留父实现，但要先证明：输入 S_C 不变时，新 adapter 在四阶段全部 100 个 epoch 的实际样本身份与父 loader 完全相同；再对一个实际 batch 做隔离的相同优化更新张量比较。若无法证明，停止“复用既有 C 训练”的严格比较，必须让**两组都使用新公共 loader 并重新训练基线**，不能强行把旧基线的结果拼接进新方法。

为了归因，建议新增一个纯代码消融 `same_RA_members_legacy_sort_vs_stable_slots`，但它会产生另一完整训练组。第一轮不运行；先严格执行上面的无参数差异/真实更新预检。如果后续论文必须分离顺序效应，另立预算做这个训练消融，不能把 CPU 位置统计当作训练因果结果。

## 4. 评估 E1：按轨迹独立标准高斯噪声，全部比较组共同运行

旧 `coreset_exp.evaluate` 只在每个 stage 开始重置全局 RNG，而 `CompletePolicy.get_action` 从选中最大 logits 的高斯分量上调用 `Independent(Normal).sample()`。先前不同任务的完成步数会影响后面任务的 CUDA 随机流。这在最新审计中已确认。

E1 是**新的、所有组统一使用的评测协议**，不与旧协议的 AUC 混合。

- 每个 `(suite,train_seed,stage,task,trial)` 预先在 CPU 独立生成 `600×7` 的标准正态 `float32` 表，使用冻结 E1 标识和 SHA256 派生的确定种子；**不包括算法名字，也不使用成功/失败结果**。
- 20个初态0..19、每cell 10lane×2 group、同原环境组seed `[seed,seed+1]`、启动5步零动作、上限600步、历史最多8、成功判定都不改。
- 原每个动作先选最大 logits 的高斯分量，然后按 `a=mean_selected+std_selected*epsilon[trial,policy_step,:]` 生成7维动作，保持 Normal 采样的同一条件分布和原后处理；不是均值动作。
- 某一个episode提前结束不影响其他episode使用的噪声；下一被评任务不会继承前一任务的 CUDA 抽样流。保存噪声生成源码、每cell SHA和每条trial的匹配证明。
- 同一随机方案下分别对 **冻结旧 C checkpoint** 与新 RA checkpoint 运行全部10格×20条，即200+200=400条 E1 轨迹；Random checkpoint 可以在门槛通过后用相同E1另评200条作为完整 GOAL 目标参照。
- 环境与模型中若仍存在其他依赖全局 RNG 的随机源，须逐项审计；独立动作噪声不自动控制一切环境数值随机性。

这个改动要求 `ra_exp/eval_independent.py` 拦截原 GMM Normal 采样路径，保持 logits argmax、分布参数、动作后处理、时序历史和评估环境不变；训练原路径不能共享一个被悄悄改动的动作头。要提供给定固定 epsilon 的原生动作函数与新 E1 解码逐元素一致性测试，并检查 E1 与原单步 Normal 采样在分布构造上的等价性。

E1 与原评估协议不同，因此**旧 C 的已发表旧 AUC=78.75%只是历史背景，不是新 E1 的决策阈值**。必须用新 E1 复评的 C AUC 作为主比较基准。旧 C、Random 的训练不用重跑，除非第3节数据加载等价性核验失败。

## 5. 冻结训练与阶段存储协议

首选 GOAL seed42（历史跨套件失败最明显），避免同时引入 OBJECT 的多 seed 训练成本。父版本与原始输入身份以以下已经发布的冻结源码为准：
`reports/goal_random_coreset_seeds42_43_44/final_evidence/queue/seed42/source`（git ref `95503d6`）。实际 GPU 环境内父 run 必须与其输入/权重/原始特征哈希逐项一致。

GOAL：`LIBERO_GOAL`, task_order_index0, 6→7→8→9, seed42,10条示范/任务，state=True，原缓存，官方 `goal_multitask_model_ep96.pth`。原8步输入、末端专家动作监督、冻结视觉语言前端；Temporal Decoder＋5分量7维GMM Head 原实现，`BC+0.1×meanIFA`、gamma0.7。batch10、100完整epoch/任务、AdamW/原lr1e-4及配置、warmup500、每步线性调度，每任务重建optimizer/scheduler，网络继承。无新参数、样本重加权、损失或训练中rollout。

|GOAL任务|完整窗口数|本任务入库数|固定随机锚点数|每epoch批次数|该阶段optimizer步|
|---|---:|---:|---:|---:|---:|
|6|997|499|249|100|10,000|
|7|889|445|222|139|13,900|
|8|904|452|226|185|18,500|
|9|1969|985|492|337|33,700|

GOAL新组总：400epoch、76,100次正式更新、4个epoch99模型，入库总数2381。原 buffer 容量59660000是文件条目门槛，当前未触发旧成员裁剪，继续保持不变。

训练时：完整 Taskk 与早前 RA 所选任务各自记忆混合，必须保持父配额/训练曝光；学习 Taskk 完成并保存最终模型后，才从本任务 U_k 选样入库。Task9自己的成员变化不会影响本次四阶段已完成模型，但仍为同一算法/审计完整入库。IFA任务对算法仍按**实际新 RA Buffer**执行，不能硬编码父结果；如果同一数据仍选择相同任务对，也不意味着实际IFA梯度相同。

OBJECT 扩展只能在 GOAL42 门槛通过后再启动。OBJECT 需继承 **OBJECT 自己的**官方权重、gamma0.3、冻结特征缓存与父配置，不能将 GOAL 权重或配置直接复制过去。OBJECT N为1562/1480/1560/1319，q为781/740/780/660，锚点为390/370/390/330，400epoch、105500次更新；E1 同样 C+RA 各200条。

## 6. 新工作区和最小代码改造

从匹配 suite/seed 的父冻结源码复制隔离工作区，新添加模块：

- `ra_exp/selector.py`：使用父原KCenter+父同seed Random无放回名单，计算 A，固定随机锚点，exact-minimax-removal，对原PT身份和哈希入库；`selector_reference.py`是逻辑参考而非直接的父适配层。
- `ra_exp/storage.py`：原预提交名单、断点幂等复制、旧成员保护；新选样report包括锚点、与C重叠、交换、新旧ID、距离诊断。保持原PT内容不变。
- `ra_exp/slots.py`和`ra_exp/loader.py`：读取父 C 旧Buffer和新 RA 的真实selected清单，建立逐任务稳定槽位，使用原局部shuffle与原sampler；支持 `identity_test`。
- `ra_exp/train.py`：复制父训练器，只扩展 arm/路径与loader调用；`observe`/BC/IFA/模型与优化器计算不得改变。首次入库前Task6真实模型张量对父 C 检查。
- `ra_exp/eval_independent.py`及极小 native GMM adapter：创建方法无关的CPU epsilon，挂接原 `get_action` 的 Normal 解码位置，仅改变随机数来源；原历史、环境、动作判定不动。
- `ra_exp/prepare.py`, `ra_exp/preflight.py`, `ra_exp/freeze.py`, `ra_exp/audit.py`：哈希绑定、离线检查、恢复、逐episode复算和门槛判定。

实际改动必须从源diff逐函数审计，**改动位置是选样＋显式loader顺序适配＋统一E1评估**，不是宣称算法仍只改选样而其他代码逐字节不动。研究主变量是RA选中的成员；后两者作为两臂共同的数据/评估控制。路径应独立且只读旧缓存/旧checkpoint，禁止对旧目录直接patch、写入新模型或重贴合同。

## 7. 运行序列和暂定接口（待开发，非已存在命令）

A. **只读绑定**：核验父 C/R checkpoint、原始PT hashes、预训练、演示/BDDL/初态、完整训练协议、同seed引用源。分别保留父历史报告与新E1的不同评估协议编号。

B. **CPU 离线选样**：重算每task原S_C，必须逐身份/顺序等于归档，重放Random同seed样本，计算RA四任务选择、锚点、交换、内存、时刻重复及半径，仅为设计实现校验；不根据这些数值调整 a/q。选样验证失败不训练。

C. **数据加载对齐**：重建四阶段旧 C 原path-sorted+local.shuffle+DistributedSampler顺序，以及适配器用原C成员时的顺序；检查全部100epoch身份一一相同。对RA检查只在真正替换的旧槽位发生前置身份差异，当前任务位置、任务标签、总曝光和更新数不变。必要时隔离一小步 optimizer 验证回退原C完全等价，丢弃预检模型。

D. **E1评估前置**：确认真实GMM解码参数、Normal分布、历史长度、20初态、噪声bank形状、跨方法SHA一致，配对运行不依赖先前cell步数。0环境预检为默认；可以通过一次隔离无环境native forward单元测试证明。

E. **正式单组训练 RA**：官方GOAL seed42起点和空Buffer，按原任务顺序100epoch/任务，冻结RA选择与slot list，400epoch、76100步、4个stage最终模型。checkpoint完成后入库；训练失败只用完整状态恢复，不重新抽锚点。

F. **E1配对闭环**：独立进程分别对旧 C 和新 RA 的4个最终模型评10格，共400条真实rollout。每个stage/model加载经过严格key/模态/合同验证，原始 trials 0..19，成功/失败逐条保存。任何正常失败计为失败，不能重试挑选成功；技术异常保留attempt和RNG/eps索引证据，不能盲重启。

G. **独立指标复算与停止**：从200条各组原始trial以有理数复算AUC、FWT、NBT、OldFinal、FinalAll，保留全部matrix，完整验收后按固定门槛判定是否推进。可选先对 Random checkpoint进行E1补充参照200条，不能用旧Random协议结果填充E1表。

命令表示**目标接口，不是现有仓库模块**：

```sh
export RA_PARENT_RUN=/absolute/path/to/verified/GOAL_seed42/coreset_run
export RA_SOURCE=/absolute/path/to/new_isolated_RA_source
export RA_RUN=/absolute/path/to/new_empty_run
export RA_SPEC=/absolute/path/to/spec.yaml
export RA_SEED=42
export PYTHONPATH="$RA_SOURCE:${PYTHONPATH:-}"

python -m ra_exp.prepare --spec "$RA_SPEC" --out "$RA_RUN" --reuse-frozen-parent
python -m ra_exp.select_offline --run "$RA_RUN" --all-four-tasks --no-model-forward
python -m ra_exp.preflight --run "$RA_RUN" --verify-legacy-slot-parity --native-no-env
python -m ra_exp.freeze --run "$RA_RUN"

# 唯一正式新增训练组：
torchrun --standalone --nnodes=1 --nproc_per_node=1 -m ra_exp.train --run "$RA_RUN" --arm ra_coreset

# 采用E1协议分别重新评旧C与新RA，原checkpoint无需重新训练：
python -m ra_exp.eval_independent --run "$RA_RUN" --source-archived-coreset "$RA_PARENT_RUN"
python -m ra_exp.eval_independent --run "$RA_RUN" --arm ra_coreset
python -m ra_exp.audit --run "$RA_RUN" --protocol E1 --seed-gate --no-auto-extend
```

所有独立输入、hash和运行日志应先完成原子预提交。命令退出零不是完整验收；需逐checkpoint strict-load、task6父张量和epoch计数、训练顺序证明、200×2 episode及attempt账目，完成之后才`COMPLETED`。

## 8. 单seed门槛与扩展路径

首次只GOAL42。全部工程验收PASS且比较都来自**相同E1**时：

**扩大资源的必要条件**（RA − C）：
1. AUC严格>0；
2. FWT不下降；
3. 最终旧任务平均OldFinal不下降。

全部使用原始20次成功计数和精确分数。不拿NBT降低单独代替绝对保持或新任务学习。Task6、Task8、Task9逐任务始终报告；即使OldFinal相同，也不能声称每个旧任务都被保护。

达到门槛后，再在GOAL42相同E1下评现有Random checkpoint，判断RA是否实际上追平/超过 GOAL Random（这属于更强的**问题已解决**主张，不是必要性门槛）。随后进入OBJECT42：预注册保留OBJECT原C在五seed上的优势，要求RA的OBJECT E1 AUC/FWT/OldFinal不低于原C。OBJECT失败时记录跨套件取舍，不用调整锚点率再称独立验证。

两套件单seed均过门槛，才扩展GOAL seed43、44及OBJECT seed43、44；OBJECT seed40、41可后续补齐。按预定顺序运行，所有参数不变，保留seed43**首次**历史基线为预注册主比较，repeat1另列敏感性分析。因为方案已参考 seed42 的既有失败而设计，42的成功只能作为探索性筛选，不构成独立确认。不得按最后性能调锚点比例、改距离、挑初态或检查点。旧模型在E1的再评使用新的公平协议，原指标仍单列历史不可合并。

## 9. 预算和日志

首轮 GOAL 新增训练：1组×400epoch＝**76,100 optimizer updates**，4个ep99模型；旧基线新增训练0（若loader对照等价性验收失败则另立共同重训协议）。统一E1旧C＋新RA的正式rollout**400条**，若需要Random E1则另200条。历史单组200条评估约70min仅供资源量级参考，不承诺当前耗时；选样数据矩阵在N=1969时以FP64常驻约31MB，较大任务可分块但必须保持确定性与决策规则。

OBJECT42若启动：1组400epoch＝**105,500 optimizer updates**，旧C＋新RA E1 400条，Random按需要另200。各组记忆总条数完全保持GOAL2381、OBJECT2961。模型结构参数与原相同，不增加推理记忆网络；新索引/噪声bank/审计文件字节单列。

必要产物：`PARENT_BINDING.json`, `SOURCE_DIFF.json`, `FROZEN_SPEC.yaml`, `CORES_ET_PARITY.json`, `RANDOM_DRAW_PARITY.json`, `RA_SELECTED_task{t}.json`, `DISTRIBUTION_DIAGNOSTICS.json`, `STABLE_SLOT_MANIFEST.json`, `DATASET_EPOCH_PARITY.json`, `IFA_PAIRS.json`, `TASK6_TENSOR_PARITY.json`, `E1_NOISE_HASH.json`, `E1_NATIVE_ACTION_PARITY.json`, `EPOCHS.json`, `CHECKPOINTS.json`, `TRIALS_AND_ATTEMPTS`, `SUCCESS_MATRIX.json`, `METRICS.json`, `PROTOCOL_ACCEPTANCE.json`, `SEED_GATE.json`, `FINAL_REPORT.md`。

恢复记录必须保留model、optimizer、scheduler、grads、train/eval modes、Python/NumPy/CPU/CUDA随机状态、stage/epoch游标、stable-slot manifest、选择清单和实际buffer哈希；训练中断不可二次抽样、重复入库、只重试失败episode。

## 10. 可支持的结论和新颖性

若效果为正，可以支持“在指定MLR+IFA协议中，用部分均匀锚点替代纯极端覆盖选样，对 GOAL有价值并在OBJECT保留性能”，并通过稳定slot/E1强化可解释性。它**不能**保证每种数据套件都有效，也不意味原动作比例得以严格恢复、能够识别关键行为、证明单条记忆的因果重要性。

RA作为随机样本与K-center的混合本身新颖性有限；本轮先把有效性与协议公平性做扎实。如果科研目标要求顶会原创贡献，应在多seed、多套件确认后，与普通混合样本、K-center、Random和等预算其他coreset进行直接对照，再提出可证伪且具体的结构性研究贡献，而不是把“混合”包装为新的核心理论。

## 固定仓库材料（用于科学设计，不表示新增结果）

- GOAL C/R三seed：`reports/goal_random_coreset_seeds42_43_44/FINAL_REPORT.md`
- OBJECT五seed：`reports/object_five_seed_summary/REPORT.md`
- Event负效应：`reports/object_event_deep_readonly_audit/REPORT.md`
- SC-v1：`reports/goal_sc_coreset_seed42/completion/FINAL_REPORT.md`
- SC-v2：`reports/goal_scv2_seed42/completion/FINAL_REPORT.md`
- SC-v2数据顺序/RNG审计：`reports/goal_scv2_mechanism_audit/REPORT.md`
- GOAL实际父实现：`reports/goal_random_coreset_seeds42_43_44/final_evidence/queue/seed42/source/libero/lifelong/algos/mlr.py`和`libero/lifelong/selectors/greedy_kcenter.py`。

以上相对于已核实 SHA `95503d6799b14862d5d5d21ab065c8ba252863b2`。本设计不创建后台任务、不进行远端写入、不执行用户服务器上的正式实验。
