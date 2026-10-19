# ZR-v2：Coreset＋IFA＋双向隔离稳定特征均值校准，seed43 单seed实验

**状态：执行设计，未运行真实数据校准或 LIBERO 闭环。**

本轮只新增一个 `coreset_ifa_zrv2` 组。复用 seed43 **第二次完整 Coreset＋IFA（repeat1）** 的四个 ep99 阶段模型及其配套数据、Coreset 和评估记录，不重训公共策略、不重新导出 latent、不重跑基线闭环。新增每阶段10个校准 epoch，共40个 epoch、10,550次残差 optimizer step、200条正式闭环。只有相对锁定基线满足预先固定的指标门槛且工程验收通过，才报告“可进入后续seed讨论”；不自动扩展。

## 0. 依据和不能混用的两种“协议对齐”

已通过 GitHub 插件确认 MLR 在本次读取时为 `d4a2420e7b5fe6cbd15dddf9a6ada1b6279ccdd1`。源仓库为 `uhhj/lifelong_mlr_ifa_minrep_20260821_193226`。来源路径见末节。

五seed ZR-v1主报告采用seed43 repeat1，平均AUC差−3.375pp；全部首次口径为−6.5208pp。结构检查通过没有建立稳定的任务收益。该事实支持开展结构重设计，但没有识别唯一失败原因。

本设计沿用刚确定的ZR-v2，不偷偷返回ZR-v1：

1. 公共策略独立按原Coreset＋IFA训练；残差不参与其前向损失和反向传播。
2. 残差使用全局固定官方预训练时序编码器T0的输出，不使用正在变化的公共Tk输出。
3. 门函数按任务到来顺序双向更新；已知指令上仅自己的增量分支有单位响应。
4. 残差只输出7维有界均值修正，不修改公共GMM尺度和logits。
5. 当前残差使用当前全示范，旧残差使用其自己的Coreset重新校准，不永久冻结旧UV。

**对齐范围：**同一实际父模型、同一数据、同一记忆身份/数量/顺序、同一动作监督含义、同一原生执行和评估协议。**不声称整个算法训练预算不变：**新增10轮校准、固定T0支路和推理开销均须计量。这不是“只替换入库选样”的实验，也不能通过与100epoch旧基线比较声称等计算量优势。完整任务增量系统相当于每任务原公共100epoch后另作残差10epoch，但两段优化不同参数。

原论文、旧报告没有规定ZR-v2校准优化器、初始化及具体10轮采样细节；本文件和spec.yaml中的这些数值是本轮预先固定的工程决定，不冒充论文设置或已有验证结果。

## 1. 锁定seed43基线——不再只更换报表数字

### 1.1 唯一主父运行

- run ID：`Seed43_Random_Coreset_repeat_20261016_v1`
- 逐episode记录中的服务器根目录：`/data/mlr_ifa/experiments/Seed43_Random_Coreset_repeat_20261016_v1`
- arm：`coreset`
- 仓库逐episode路径：`reports/object_five_seed_summary/evidence/20261016_seed43_three_arm_repeat_v1/completion_evidence/RC/coreset/evaluation`
- stage6 checkpoint SHA256（已在首条episode中核实）：`73fa1a2b1f0b295a4c078bef18d0e423d64ca04858b902ab0119f37ffd398763`
- modal SHA256：`c379b9930387abc9eed54e8a9f83394f2f8be80ee1536921a53ce2cab66ea78d`
- stage7–9 SHA256须从同一repeat1 episode记录提取，再核验真实文件。不得凭文件名认为权重一致。

路径可能被迁移；只接受哈希相同的迁移文件。执行端必须绑定**四阶段checkpoint＋官方原始预训练权重＋repeat1冻结源码/config＋同次latent＋同次Coreset manifests＋BDDL/init states**。前次ZR-v1绑定首次训练、报表换repeat1的问题，本轮必须在输入层解决。

不得用首次43的缓存、成员或权重，搭配repeat1的78.6458%分数。若某些载荷确实与首次逐字节相同，可以由hash证明复用，但不以“都是seed43”代替证明。若repeat1关键载荷缺失，置 `BLOCKED_PARENT`，不自动切换首次，也不自动重训。

### 1.2 基线精确矩阵与指标

|阶段|Task6|Task7|Task8|Task9|
|---|---:|---:|---:|---:|
|6|40%|—|—|—|
|7|95%|100%|—|—|
|8|75%|100%|85%|—|
|9|65%|95%|100%|55%|

每格20条。精确指标：AUC=151/192，FWT=7/10，NBT=−61/360，OldFinal=13/15，FinalAll=63/80。

首次43 AUC94.375%、ZR-v1 AUC73.4375%仅作为已披露的历史背景，不参与本轮推进门槛，不因新结果更换基线。

## 2. 保持的公共协议、预算与数据访问边界

|项目|锁定设置|
|---|---|
|套件/顺序|LIBERO_OBJECT，task_order_index=0，6→7→8→9|
|seed|43；基线run身份另行绑定，不将seed等同于run|
|每任务示范|原10条，ID和HDF5身份一致|
|模态|vision/language/state=True；保留预训练state元数据unknown的历史披露|
|训练单位|原8步潜特征；只使用真实末端专家动作作为BC目标|
|公共训练|原100epoch、batch10、AdamW 1e−4、betas(0.9,0.999)、eps1e−8、wd0.1|
|公共调度|原逐optimizer-step线性，warmup500，每任务重建optimizer/scheduler|
|公共IFA|原平均角度损失0.1，gamma0.3；作用于共享时序表示|
|公共模型|直接读取同一repeat1各阶段ep99；本轮没有公共optimizer|
|校准/执行模式|公共Ck和固定T0均eval并冻结；残差无dropout|
|原缓存|同一repeat1 state=True latent，不另导出或加入状态到Coreset距离|

逐任务清单：

|任务k|当前Nk|训练/校准前旧B|校准总样本|每epoch batch|本任务入库qk|
|---|---:|---:|---:|---:|---:|
|6|1562|0|1562|157|781|
|7|1480|781|2261|227|740|
|8|1560|1521|3081|309|780|
|9|1319|2301|3620|362|660|

数量是已归档父实验下的预期，启动时必须通过repeat1清单实测，不通过padding、drop_last或重复采样凑数。

每阶段只允许读取Dk全窗口＋B<k。旧任务已丢弃完整示范不重新读取；基任务0–5只保留语言约束/既有探针，不加入其动作训练数据。不能让已预计算的未来目录泄露到当前校准器。校准后按父清单将当前qk条原payload写入新Buffer，旧成员不重选。顺序、原文件名及hash一致；新增元数据放旁路，不能作为额外.pt被原DataLoader读到。

父四个Ck可以预先做元数据校验，但阶段k只加载Ck；不可提前拿C9作为Task6公共策略。阶段快照一经保存不因以后校准改变。

## 3. 算法的精确定义

### 3.1 固定输入和公共模型

空间latent仍为H。公共模型Ck输出：

`b_m(H), sigma_m(H), logits_m(H)`，m=1…5，b_m为7维均值前激活。

固定参考 `q(H)=T0(H)_last` 为512维。T0是从相同官方六任务预训练权重复制出的时序编码器，所有参数和buffers冻结、始终eval；所有任务共用一份。T0与Ck不得共享底层可写参数storage。更换Ck时T0的张量hash必须不变。

T0继续使用父模型的token排列、因果可见范围和末端读取，不在本轮顺便调整语言token位置或增加未来观测。输入H在训练时为原缓存，在执行时为原空间编码器在线输出；历史长度1–8沿用原协议。

### 3.2 双向门：只使用历史＋当前规范指令

规范语言向量来自同一已完整加载LoRA的冻结语言编码器，不换裸CLIP。仅门内部归一化 `e=raw_e/||raw_e||`。在线使用实际H最后时刻第0模态，不查任务ID或canonical vector表。

创建Task k时，旧空间含0…k−1全部已知规范指令，包括六个基任务。用FP64薄SVD构造其正交基Q：

`v=(I−QQ^T)e_k; g_k(e)=v^T e/(v^T e_k)`。

每个旧分支门向量更新为：

`w_j_new = w_j_old − (w_j_old^T e_k) * g_k_vector`；新分支门为g_k。

数学目标：在当前已知增量指令上 `h_j(e_i)=1[i=j]`，基任务指令对全部残差门为0。旧规范指令的响应保持不变；**旧门向量本身允许改变**，不得套用ZR-v1“门张量永久不变”的验收。

不得用未来指令构造本阶段门，不以逆矩阵岭正则、重调秩阈值、输出snap-to-zero或任务ID硬路由掩盖失败。规范线性约束不能保证所有同义改写；该限制继续披露。

数值合同继承原型：rank threshold=max(1e−12,1e−10*sigma_max)，阈值附近factor10的奇异值拒绝；新正交分量norm≥0.01；所有更新后门的norm≤100；规范矩阵误差≤1e−10；实际编码非对角门≤1e−6、自身门误差≤1e−5。实际原型更严格时沿用更严格值。

FP64保存与计算所有门；不因调用model.float()降为FP32。必须用repeat1真实缓存/在线批次1、10、32语言输出复核，不能只沿用seed42规范向量一次通过的结果。失败置BLOCKED_GEOMETRY或FAIL_RUNTIME，不自动改算法。

### 3.3 七维有界均值校准

每任务一个rank8分支：`R_j(q)=U_j V_j q`，U为7×8，V为8×512，无bias、无dropout。

初始化：U=0，V~N(0,0.02²)，局部CPU生成器seed=43+j；旧UV从上阶段ZR-v2暖启动，不载入ZR-v1或原型已训练权重。

`delta(H,e)=0.1*tanh(sum_j h_j(e) R_j(q(H)))`

`mu'_m=tanh(b_m+delta)`，**同一个delta用于所有5个分量**。

`sigma'_m=sigma_m; logits'_m=logits_m`。

必须获取原头的mean_layer前激活b_m，不能用atanh已饱和均值反推。固定sigma/logits来自同一Ck的实际head输出，不重新估计。最终仍用完整MixtureSameFamily计算真实动作NLL。

在同一H、同一Ck下，最大均值差≤0.1（数值容差1e−6），分量概率、尺度、argmax严格沿用公共模型。固定噪声下单步动作差受同样的界约束。此为动作数值约束，不是物理安全或整条轨迹误差界，不能新增全局动作clip改变父执行协议。

## 4. 两段分离的执行方式

理论算法每阶段为：原Coreset＋IFA独立训练100epoch得到Ck，然后冻结Ck校准残差10epoch。因为残差从不进入公共训练，**本轮允许直接复用已有四个Ck**；不会像ZR-v1那样切断残差共同训练导致的共享模型轨迹。

实际循环：

1. 装载并冻结repeat1 Ck；核验hash。固定T0不变。
2. 更新双向门，创建当前零输出分支，保留前阶段旧UV。
3. 验证更新门前后对已知旧规范指令的响应与固定q下修正不变；新支路初始化输出为零。此时新指令不受旧残差作用。
4. 构造合法Dk＋B<k；按原逻辑文件次序、seed43的任务内shuffle建立Dataset。
5. 全部已到来UV允许训练；公共Ck、T0、空间前端和gate buffers不进入optimizer。
6. 固定训练10epoch，不早停，不按NLL或闭环挑轮次；保存最后calibration_epoch_index=9。
7. 原Coreset清单入库，记录hash与事务；进入k+1。
8. 校准全部完成后独立进程按原顺序评估四个不可变阶段快照。上一阶段快照不能引用后来被覆盖的gate/UV内存文件。

**Task6有校准，因此其校准后输出不要求等于基线；在校准前零残差时必须等于基线。**

## 5. 校准训练协议（本轮新增、提前冻结）

|项目|设置|
|---|---|
|每阶段epoch|10|
|batch、workers、drop_last|10、0、False|
|采样|同一逻辑Dataset；单副本DistributedSampler用于索引，shuffle=True、sampler seed=0、set_epoch(0…9)|
|进程|单进程单CUDA设备；不需要DDP，因为公共网络不训练|
|损失|最终校准GMM对真实末端专家动作的batch平均NLL|
|optimizer|一个AdamW，只含全部已到来UV；lr1e−4、betas(0.9,0.999)、eps1e−8、wd0.1|
|梯度裁剪|当前收到梯度的UV整体norm≤100；不是重要性评分|
|调度|原make_linear_schedule_with_warmup(500,10*len(loader))；每batch一次optimizer step后scheduler step一次|
|阶段边界|UV暖启动；重建optimizer/scheduler，清除旧动量，不重新初始化旧UV|
|附加损失|不加IFA反向、蒸馏、损失权重、重要性打分、梯度投影|

公共Ck已经由原BC＋IFA训练；校准阶段不额外反向IFA，因为共享表示被冻结。这不能表述成原公共IFA被关闭。仅将原observe()套在被冻结的公共头上会遗漏真正的校准目标，应建立独立校准入口。

学习率函数完全沿用父源码：s<500时s/500；之后max(0,(S−s)/(S−500))。S为本阶段校准batch数。首个step学习率可能为0，应记录而不是错误判断训练未接通。该500不是论文校准设置，而是本轮选择保留父优化器调度参数的工程规定。

### 5.1 精确落实“每个校准器只由自身任务数据训练”

数学双向门为0，但浮点门有微小非零。不能只依赖它得到近零的跨任务梯度后声称完全独立。

训练时使用真实已知任务标签**仅限定反向路径**：对属于任务j的样本，其他i≠j分支的数值输出仍参与相同前向求和，但对其UV stop-gradient。对本batch没有自身样本的分支，使其完全不连接该batch的loss图。推理完全不用这个标签参数，所有门值仍来自真实语言。

使用 `optimizer.zero_grad(set_to_none=True)`。未收到本任务监督的分支梯度必须为None，而不是全0；调用optimizer前验证该条件，并检查它的UV、Adam一二阶矩和step计数没有更新。这样不会发生“旧分支虽没自己的数据，却因Adam动量或weight_decay漂移”。这项处理不是把推理小门值硬置0。

混合batch中损失仍按每条真实样本等权：各任务子batch的loss_sum相加后除以原batch大小，不改成每任务等权。一个batch仍只有一个optimizer/scheduler step；各分支实际Adam step计数另报。

### 5.2 冻结计算的autograd要求

Ck和T0同时requires_grad=False和eval。它们的前向用torch.no_grad，输出随后作为残差训练的常量。不要因为eval就认为自动冻结；不要在整个校准前向上no_grad，导致UV无法训练。不要将inference_mode生成的常量未经核查送入需反向的矩阵乘法；no_grad产生的正常张量更适合本处。

不保存跨任务完整原示范的固定q副本。必要的临时RAM缓存只能包含当前合法Dk＋B<k，须计量并在阶段结束清理；首版推荐直接前向，避免引入新的持久化训练数据。

## 6. 四类工程检查和停止规则

### 6.1 父模型/输入复用检查

先从repeat1所有10个单元复算基线，验证每单元20条唯一trial与初态，确认矩阵。四个真实Ck文件hash与这些episode的checkpoint_sha256一致。若不一致，不把两份不同模型称为共同起点。

完整核验重复运行的实际代码、环境、modal、feature/manifest身份；仓库原seed43源码仅作为审计参考，不能无检查替代repeat1实际执行源码。按repeat1证据恢复CUDA/cuDNN/TF32/AMP等设置，禁止为新方法单独升级框架。优先使用与repeat1相同GPU类型；不得承诺本会话已验收服务器硬件。跨GPU时披露历史对照限制。

### 6.2 双向门与固定输入

构造仅用已到来规范指令；每阶段保存规范响应矩阵、所有门norm和实际变体误差。固定T0在四阶段逐张量不变、同H输出可复现。gate更新前后的旧UV完全相同；旧gate权重不要求相同，只检查响应不变。

### 6.3 前向与优化器集成

在隔离克隆做至多4次optimizer调用、0个环境step，随后丢弃状态：

- 零残差与原Ck读出一致；非零合成UV能改变自身指令均值，保证不是空分支假通过。
- 只更新一支路时其他已知指令的残差贡献/均值在1e−6以内保持；不靠hard ID推理掩盖泄漏。
- 活动UV可获得BC梯度；冻结Ck/T0无梯度且hash不变；无自身样本分支grad=None且Adam状态不变。
- 同一公共输出的scale/logits/probs和selected完全不被delta改写；均值差界成立。
- 序列化重载、epoch边界恢复后复现同一更新；新分支创建不消耗全局RNG。
- Brief latent路径与Complete原生get_action路径一致；真实历史长度1、4、8检查。

不能复用这4次校准参数作为正式初始化。合成CPU测试不计作真实原型或GPU集成通过。

### 6.4 正式过程与终点验收

每阶段开始、cal epoch0/9和阶段末记录：公共Ck/T0不变、gate固定于本阶段、实际活跃梯度所属、各支路UVnorm、修正幅度和饱和比例、合法旧Buffer全量的交叉贡献检查。幅度/NLL仅为诊断，不作优先级、不改变采样/权重/训练预算。

**不要求自身任务旧UV在重新校准后保持原输出。** 这正是v2的适应部分。也不要求不同Ck之间完整策略输出相同。

直接检查应先固定Ck、q和除测试支路外的全部参数，再比较支路开关或克隆更新；跨阶段模型差不能作为门控泄漏证据。

原型数值门值满足而实际贡献超标时，报告数值隔离失败并停止，不夹断delta之外的门/分支幅值，不放宽容差。未激活的分支如实记录；不为了满足幅度门槛延长训练。最终是否有用依闭环判断。

## 7. 原生评估协议

四个阶段快照=(Ck身份、唯一T0身份、阶段gate矩阵、截至k所有UV、校准终点元数据)。推理共同执行空间编码一次，随后Ck时序和T0时序分别前向，共用同一H历史，避免双重图像增强或双队列。

必须修改实际 `CompletePolicy.get_action()` 对应读出，而非仅改返回latent的forward。父评估器已经维护外部最多8步队列，启动历史1…8原样处理。额外支路构造/加载必须保存恢复全局RNG；读取T0/门不得插入额外采样。

- 每阶段评所有已见增量任务：1+2+3+4=10个单元。
- 每单元20条，初态索引0…19；同一BDDL、相机、控制器、init-state文件和hash。
- 10个向量环境，两组环境seed43/44；开始5步零动作；最多600个策略步。
- 固定 `argmax(logits)` 分量后从Independent Normal采样，不改成均值控制，不增加动作后处理。
- 按父逻辑每阶段重置seed43，不额外引入逐轨迹seed；记录原全局采样耦合的限制。
- 父checkpoint固定ep99，校准checkpoint固定epoch9。日志分别记 `parent_epoch=99` 与 `calibration_epoch=9`，不能把10轮校准写成100轮训练。
- 单元正式记录含parent_checkpoint_sha256、新combined checkpoint_sha256、T0/gate身份、seed/arm/stage/task、trial和初态hash、success布尔值。
- 普通失败不重跑；技术异常的中断记录保留。若无法恢复准确的进程/环境/RNG状态，不只挑失败episode补跑或丢弃；记录未完成并单独授权新attempt。

本轮默认不新增基线rollout；必须承认历史RNG与跨设备影响限制逐episode因果推断。离线off-residual与同一Ck输出一致可以核验结构，却不等于在不同设备重复得到完全相同成功率。

## 8. 指标、预定门槛与解释

r_{i,j}=训练/校准到i后在j上的成功率。继续原定义：

- AUC=(1/4)*sum_{j=6..9}[(sum_{i=j..9} r_{i,j})/(10−j)]。
- FWT=(1/4)*sum_j r_{j,j}。
- NBT=(1/3)*sum_{j=6..8}[(sum_{i=j+1..9}(r_{j,j}−r_{i,j}))/(9−j)]，保留负值。
- OldFinal=(r_9,6+r_9,7+r_9,8)/3。
- FinalAll=(sum_j r_9,j)/4。

严格延续前一闭环方案的四项推进规则，本轮相对repeat1实际父基线：

1. AUC > 151/192 = 78.645833…%。
2. FWT ≥ 7/10 = 70%。
3. NBT < −61/360 = −16.944444…个百分点。
4. OldFinal ≥ 13/15 = 86.666666…%。

以上与所有工程门共同通过，才给 `ELIGIBLE_FOR_REVIEW`。不自动开新seed。显示时四舍五入，判定用真实成功次数精确Fraction。

该门槛保守，尤其提高初学成绩也可能改变NBT基点；因此未全部通过时仍报告全矩阵和具体取舍，不把它等同于“所有指标无效”。不事后删掉失败条件。

本轮只能评价“此校准版本在seed43该父模型上是否改善”，不能证明四项修改各自有效、优于普通任务条件残差或具有新颖性。新增样本曝光/冻结T0支路/校准算力未做等预算消融；若成功，后续才单独设计相同校准预算的简单对照，不能把它当本轮已完成。

## 9. 代码改造合同及建议命令

新隔离源码建议 `zr2_exp/`：

|模块|职责|
|---|---|
|prepare.py|父repeat1 checkpoint/数据/episode绑定，路径迁移核验，新run配置|
|gates.py|逐阶段双向门构造、真实输入验证、版本化保存|
|model.py|独立T0、公共raw头、7维calibration、训练/在线统一解码|
|calibrate.py|四阶段10epoch校准、gradient ownership、optimizer/scheduler/恢复|
|integration.py|少量真实路径与优化器检查，零rollout|
|evaluate.py|包装父评估器，仅替换策略读出和新增记录字段|
|audit.py|精确指标、所有来源/结构/预算/退出验收|

以上入口是**开发合同，尚未提供为可直接训练的程序**。本包提供的 `reference_core.py` 是纯门控/读出/梯度归属参考，`compare_metrics.py` 是可运行的离线复算器；它们不创建LIBERO环境、不加载你的实际模型，也不替代集成代码。

完成集成后，建议入口：

```bash
export PARENT_RUN=/data/mlr_ifa/experiments/Seed43_Random_Coreset_repeat_20261016_v1
export ZR2_SOURCE=/absolute/path/to/new_isolated_source
export ZR2_RUN=/absolute/path/to/new_empty_run
export PYTHONPATH="$ZR2_SOURCE:$PYTHONPATH"
# 若父root迁移，需先作同hash映射，不用旧数字拼接新输入。
python -m zr2_exp.prepare --spec "$SPEC" --parent "$PARENT_RUN" --out "$ZR2_RUN" --reuse-parent-only
python -m zr2_exp.integration --run "$ZR2_RUN" --optimizer-calls-max 4 --no-env
python -m zr2_exp.freeze --run "$ZR2_RUN"
python -u -m zr2_exp.calibrate --run "$ZR2_RUN" --seed 43
python -u -m zr2_exp.evaluate --run "$ZR2_RUN" --arm coreset_ifa_zrv2
python -m zr2_exp.audit --run "$ZR2_RUN" --no-auto-extend
```

不要调用ZR-v1共享训练入口，不使用原型固定75维输出模块，不执行原完整supervise的random/coreset重训。

恢复仅允许显式resume：保存当前阶段、门bank、UV、optimizer/scheduler、Python/NumPy/Torch/CUDA RNG、epoch/batch游标、所有模型模式、父绑定hash与原样本顺序。epoch边界checkpoint为提交权威；半个epoch日志保留为interrupted，不累计两次。若选择从上次epoch完整重算，复现相同RNG并另外记录重复物理计算成本。

## 10. 精确成本和交付物

新增总预算：40校准epoch，10,550次统一optimizer/scheduler step，105,240次真实末端标签曝光，4校准阶段快照，200正式闭环。父公共400epoch/105,500step只复用，不在新账本伪装为实际本轮完成。

10*sum(157,227,309,362)=10,550；每个batch一个optimizer step，参与支路个数不增加公共batch计数，支路实际Adam计数另存。

4个rank8、7维分支为4*8*(512+7)=16,608个FP32可训练标量，原始UV字节为66,432。4个512维FP64门另16,384字节。**最大额外模型成本可能来自一份完整T0**，不能仅报告16,608参数然后声称轻量；实际T0参数/buffer字节、reference指令表、optimizer状态、文件序列化、显存、单步延迟、校准/评估墙钟都须实测。

交付：PARENT_BINDING、REUSE_AUDIT、输入/source hash、真实config、各阶段GATES_RESPONSE与T0/PUBLIC_IDENTITY、calibration日志/样本曝光/梯度归属/支路step记录、四阶段快照及重载、200条trial/attempt记录、EXIT_CODES、精确METRICS、成本和最终验收。保留原报告与失败尝试；不得自动push除非已有独立授权。

报告状态分开：BLOCKED_PARENT / BLOCKED_GEOMETRY / FAIL_RUNTIME / INTEGRATION_FAIL / CALIBRATED / EVALUATED / AUDIT_PASS_METRIC_NOT_PASSED / ELIGIBLE_FOR_REVIEW。工程通过不等于性能通过；reference合成测试不等于GPU前置通过。

## 11. 来源登记

以下均在上述固定commit读取，是真实父信息的来源；其余明确为本轮设计。

- S1：`reports/object_zr_coreset_five_seed/FINAL_REPORT.md`：v1结果与seed43主口径调整。
- S2：`reports/object_five_seed_summary/evidence/20261016_seed43_three_arm_repeat_v1/completion_evidence/queue/FINAL_ACCEPTANCE.json`：repeat1精确分数和矩阵。
- S3：同目录`RC/coreset/evaluation/trials_stage6_task6.jsonl`：run路径、stage6 checkpoint/modal/初态hash；其余9份episode须在执行端同样绑定。
- S4：`experiments/object_coreset_seeds42_43_44/seed43/source/coreset_exp/evaluate.py`：父评估顺序/阶段seed/checkpoint逻辑的源码参考；repeat1实际source仍须匹配。
- S5：同冻结source的`libero/lifelong/utils.py`：线性warmup公式与seed函数。
- S6：PyTorch官方`Optimizer.zero_grad`与`Autograd mechanics`文档：None梯度与零梯度更新区别、eval与no_grad/inference_mode区别。这里只用于实现语义，不能据最新版文档升级父环境；必须在父版本运行集成测试。
- 用户现有`zero_response_residual_prototype/spec.yaml`、`coreset_ifa_zr_seed42_plan/spec.yaml`仅用于继承已明示的容差与旧协议；它们不是ZR-v2已通过的实验证据。
