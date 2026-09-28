# Codex 执行指令 v2：MLR+IFA / LIBERO-GOAL / seed=42 / 显式状态模态


> **v2 已作出的协议决定**：本轮按用户已确定的“论文描述对齐”目标，显式启用 `policy.use_state=true`，并保留语言与两个视觉输入。修正发布默认配置，但不声称从代码中恢复了作者未知的预训练模态设置。特征导出、潜特征训练的元数据、完整策略闭环评估必须采用同一模态合同。官方 checkpoint 缺失 `cfg.policy.use_state` 单独记为 `DISCLOSED_GAP`，不再仅因此暂停正式运行。真实权重缺失、shape 不兼容或一致性测试失败仍然需要停止。
>
> 本文件完整替代 v1。v1 中已经完成且与本次模态修改无关的修复、审计和测试予以保留；不要回滚用户工作区。下面第2.3节、第3节第11–13项，以及新增模态验收项优先于 v1 的笼统暂停规则。未标注模态或按 `use_state=false` 生成的特征、buffer、续训点不得作为本轮正式数据/初始化。

## 0. 任务目标与边界

你需要在实际工作区中审计代码、实现必要的协议对齐和工程修复、执行测试，并在 GPU、演示数据和官方权重均可用时完成一次正式训练及闭环评估。不要只返回实现建议。

本轮实验是 **论文描述对齐的 MLR+IFA 单 seed 基线（含已披露配置修正）**，不是新采样方法实验。仅运行 LIBERO-GOAL、训练 seed=42、任务顺序索引0、4个增量阶段。禁止扩展为多 seed、多个任务套件或多方法消融。禁止加入 Loss/MIR/熵优先选样、覆盖采样、审计集、动态回放比例或闭环反馈控制器。BC Loss 保留为训练目标和诊断日志，不用于选样、早停或挑选最终 checkpoint。

“完整训练”在本指令中指：从官方六任务预训练权重开始，完整执行四个新任务各100个 epoch，然后对四个阶段的最终模型完成规定的闭环评估。**不包括重新执行六任务多任务预训练，也不等于从随机初始化完整复现原论文。** 官方 README 明确尚未发布完整预训练流程；不能自行补写一个预训练方案后称其为作者方案。无法找到合适的官方预训练权重时，正式训练状态必须为 BLOCKED。

实验标签：`goal_seed42_mlr_ifa_paper_aligned_state_v2`。
该标签表示“按论文明确参数对齐、并显式记录工程修复与未公开细节的实现”，不是未经修改的作者脚本，也不承诺复现三 seed 均值。

## 1. 固定代码与证据

官方仓库与本轮固定版本：

```text
https://github.com/yfqi/lifelong_mlr_ifa
commit: a721807f2f7fa7f9833dcc85c592d78e9accd00b
```

先检查当前工作区。若已有用户修改，保存 git status/diff，不得 reset、清理或覆盖。必要时在单独 worktree 中实现。本轮不得修改远端仓库、推送提交或上传权重/数据。

阅读并记录下列真实源码的文件位置和相关行号：

```text
README.md
requirements.txt
save_feature_goal.sbatch
lifelong_adaptation_goal.sbatch
test.sbatch
libero/configs/config_adaption.yaml
libero/configs/data/default.yaml
libero/configs/train/adaption.yaml
libero/configs/train/optimizer/adam_w.yaml
libero/configs/train/scheduler/cosine_annealing.yaml
libero/configs/eval/default.yaml
libero/configs/test.yaml
libero/configs/policy/BriefPolicy.yaml
libero/configs/policy/complete_policy.yaml
libero/lifelong/save_feature.py
libero/lifelong/lifelong_adaptation.py
libero/lifelong/test.py
libero/lifelong/datasets.py
libero/lifelong/algos/mlr.py
libero/lifelong/algos/base.py
libero/lifelong/algos/basic.py
libero/lifelong/models/base_policy.py
libero/lifelong/models/brief.py
libero/lifelong/models/complete_policy.py
libero/lifelong/models/policy_head.py
libero/lifelong/metric.py
libero/lifelong/utils.py
libero/libero/benchmark/__init__.py
```

论文依据是用户提供的《Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment》：第4页式(4)与模块冻结规则，第5页任务协议，第6页指标与Setup，第7页存储比例消融。arXiv v1 附录 A.1/B.1 可以补充结构与系数，但它是另一个明确标注的公开版本，不能默默代替用户附件。

建立 `protocol_audit.md`，逐条写明：论文支持的设置、发布代码的设置、本轮采用值、证据位置、是否影响优化轨迹。不得把本轮补充约定写成论文原话。

## 2. 预注册实验合同

下表是本轮实施合同；不是可直接传给现有 Hydra 的配置文件。你要读取真实 schema 后把它映射到源码和 override，并导出最终 resolved config。

| 项目 | 本轮设置 | 依据与性质 |
|---|---|---|
| 方法 | MLR+IFA | 论文原方法 |
| benchmark | LIBERO_GOAL | 本轮选择的单套件 |
| training seed | 42 | 官方启动脚本；论文正文只写3个seed，未列具体值 |
| task_order_index | 0 | 发布配置；必须导出实际任务名称和顺序 |
| 预训练任务 | 套件顺序中的0–5，共6个任务 | 论文6任务协议＋脚本6起始索引 |
| 预训练来源 | 官方 GOAL multitask checkpoint | README 官方权重入口；原预训练seed若无法查明，写unknown |
| 增量任务 | 套件索引6、7、8、9，依次训练 | start_task=6，end_task=10，右端不包含 |
| 增量演示量 | 每任务10条完整演示 | 论文；保存实际demo IDs，禁止暗中8/2拆分 |
| 训练epoch | 每阶段100次完整数据集遍历，epoch_index=0..99 | 论文；4阶段总计400个epoch |
| 训练batch | 10，全局batch；单GPU、world_size=1 | 论文；发布默认18必须显式覆盖 |
| 特征导出batch | 1 | 发布脚本；这不是优化器训练batch |
| 时序窗口 | 8 | 发布配置与公开附录 |
| 本轮模态开关 | policy.use_vision=true、policy.use_language=true、policy.use_state=true | 论文描述对齐；覆盖发布配置的use_state=false |
| 每步token顺序 | language、agentview、eye_in_hand、state | 完整策略已有的全模态分支；state不得替换为额外视觉token |
| 状态拼接顺序 | joint_states后接gripper_states | 沿用实际spatial_encode；维度从真实数据/权重核验，不猜测 |
| 预训练use_state | 从完整cfg.policy及其他可追溯证据读取；缺失写unknown | 缺字段不伪造、不改写官方权重元数据；已披露缺口不阻塞本轮 |
| 冻结模块 | CLIP及其已有适配参数、投影、状态编码器、FiLM等空间前端 | 持续学习阶段冻结规则；不删除预训练LoRA权重 |
| 更新模块 | 6层时间解码器及GMM策略头 | 论文和发布结构；不改成扩散策略 |
| BC目标 | 原GMM负对数似然，保持窗口最后动作的监督方式 | 发布实现 |
| IFA权重 | 0.1 | 论文 |
| GOAL角度缩放系数 | 0.7，即代码cfg.gamma=0.7 | 论文明确值；覆盖脚本0.72 |
| IFA任务对 | 语言与agent-view各取相似任务对top50%，求交，再保留涉及当前任务的任务对 | 论文与源码；不是回放比例 |
| AdamW | base lr=1e-4；betas=(0.9,0.999)，eps=1e-8，weight_decay=0.1 | lr来自论文，其他沿用发布配置 |
| grad_clip / loss_scale | 100 / 1 | 发布配置 |
| 存储选择 | 维持发布版随机窗口子集，past_data_rate=0.5 | 发布脚本；不是batch旧样本占比50% |
| batch取样 | 当前任务全部窗口＋之前增量任务buffer，整体shuffle | 发布代码；本轮不改为显式新旧配额 |
| epoch数据量 | 按每阶段实际拼接数据集计算 | 对齐固定epoch协议，不擅自改为所有阶段固定steps |
| 主评估模型 | 每阶段完整100 epoch后的最终参数，即epoch_index=99 | 本轮事先规定；论文未完整说明checkpoint挑选规则 |
| 主评估 | 每阶段已见增量任务，各20次闭环执行 | 论文20 trials；只构成4个增量任务的主指标 |
| rollout max_steps | 600次策略动作；保留源码5步零动作初始化 | 发布代码补充约定，非正文明确值 |
| 闭环反馈 | 不进入训练、调参、选样或checkpoint选择 | 本轮无测试反馈的评估协议 |

### 2.1 学习率缺口及本轮明确决定

论文只明确 AdamW、初始学习率1e-4、linear scheduler，没有给出warmup、终点和调用粒度。发布代码的实际配置虽然文件名是 `cosine_annealing.yaml`，内容却是 LambdaLR、warmup_steps=500；`utils.py` 的函数按step计算线性warmup/decay；但训练循环把 scheduler.step() 放在epoch末尾，存在单位不一致。

本轮决定：**沿用发布函数及500个优化器step的warmup，将scheduler推进粒度统一为每次optimizer.step之后一次；每阶段total_steps=100×len(train_loader)，随后线性衰减至0。** 删除同一路径中重复的epoch末scheduler推进。每新任务重建optimizer/scheduler，保留网络权重，遵循原start_task行为。

必须明确披露：500步warmup来自发布配置，不是论文已确认设置；base/peak lr=1e-4不等于warmup第一步的实际lr。该修复会改变发布脚本的优化轨迹。记录此项为 `source_gap` 和 `optimization_affecting_patch`，不得声称精确复现作者未公开的学习率曲线。

干运行整个调度序列并存档；检查warmup后实际下降，既不100个epoch一直只warmup，也不一个batch重复推进两次。不要另行尝试多种scheduler并根据测试成功率选最优者。

### 2.2 存储协议的边界

源码`BUFFER_CAPACITY=59660000`按文件数量计，不是字节容量。未触顶时追加每个旧增量任务约50%的窗口，只有触顶才按任务配额裁剪。本轮保持这个选择行为，并记录实际文件数、各任务数、tensor字节及磁盘字节。

论文的“每任务约五条演示等价量”不意味着发布实现只保存五条完整轨迹。不得把50%窗口说成精确五条完整演示；不得为了匹配论文某个MB数字任意删文件。本轮不引入固定总字节预算或自适应buffer容量。

首阶段buffer为空；不能加入六个预训练任务的回放窗口。未来增量任务即便已有离线缓存，当前训练和IFA也不能读取其特征用于学习、统计或选择。旧任务全部feature缓存不是额外训练库：BC只能访问它们已进入buffer的子集。

### 2.3 状态模态冲突：源码事实与本轮修正（v2，决定已固定）

已核对固定 commit 的直接证据：

- `libero/configs/policy/complete_policy.yaml` 第20行：`use_state: False`；语言与视觉开关为True。
- `libero/lifelong/models/complete_policy.py::spatial_encode`：`use_state is False` 时，最终拼接 `[lang_tok, vision_mod, add_vision_mod]`，不用已计算的 `state_mod`；全模态分支拼接 `[lang_tok, vision_mod, state_mod]`。
- `self.rgbkey`顺序为agentview、eye_in_hand；`add_vision_mod`使用视觉循环最后一次的 `v_feat`，即手眼图像投影经另一组语言条件FiLM得到的表示。它不是第三台摄像机，也不输入机器人状态。
- 两条分支每步都有4个token，因此shape相同不能证明模态一致。
- `save_feature_goal.sbatch`和`test.sbatch`都使用 `policy=complete_policy`，所以仅在训练入口改配置不能解决导出与闭环不一致。
- `libero/configs/policy/BriefPolicy.yaml`没有这三个模态字段；`models/brief.py`仅处理已保存的latent，不重新编码原始状态。给BriefPolicy补字段用于验证和留痕，不会使旧缓存自动变成含状态缓存。

**唯一执行路线**：本轮启用状态，保留语言和两路视觉，使用原有全模态拼接分支；不新建状态编码器，不删除预训练LoRA，不重训/解冻空间前端，不改变embedding维度、token顺序或decoder。

配置实现采取本轮显式override，不改其他实验的全局默认配置。已创建的运行器要让导出、训练、评估三个入口共享如下Hydra参数；`++`用于已存在时覆盖、不存在时添加，先验证本地Hydra解析及最终resolved config：

```bash
MODALITY_OVERRIDES=(
  "++policy.use_vision=true"
  "++policy.use_language=true"
  "++policy.use_state=true"
)
# 三个入口的实际Python/Hydra命令均拼入：
# "${MODALITY_OVERRIDES[@]}"
```

不要把这些参数仅添加到外层shell却不转交子进程，也不要只把说明文档改为true。分别保存export/adaptation/eval的完整resolved config。程序入口检查值为Python布尔True，不接受缺失值或字符串假阳性。

若工作区已有独立实验yaml或新的运行器，按它的真实结构统一配置即可，但不能另造不被Hydra加载的孤立yaml冒充修复。不得无条件恢复checkpoint.cfg覆盖本轮显式模态选择。

官方权重检查完整的`cfg.policy.use_state`，不只检查顶层`cfg.use_state`。缺失时记录unknown并继续；能够定位明确元数据时照实记录。即使source cfg缺字段，也必须检查状态编码器与调制层的权重实际存在、shape兼容并被完整载入。**权重存在并不能证明作者当初训练时使用了状态**，不得作此推断。不要修改原始checkpoint，也不把本轮true回填成作者原始设置。

本轮结果标签为 `goal_seed42_mlr_ifa_paper_aligned_state_v2`；报告须说明“按论文描述启用状态模态的运行，官方预训练模态设置若缺失则仍未知”。该限制影响结果解释，但不是要求用户再次在True/False间选择的理由。

以下命名是v2需要实现的元数据，不是官方现成API：建立`modal_contract.json`，记录schema_version、use_vision/use_language/use_state、每步token顺序、state key顺序与维度、窗口长度、冻结前端权重哈希、官方checkpoint SHA-256、实际导出预处理/增广与模块模式、代码补丁哈希。生成稳定fingerprint并写入feature manifest、buffer manifest和新阶段checkpoint的元数据；不要改模型state_dict的参数键。

新训练或评估必须同时匹配**完整合同内容和fingerprint**，不能只看文件shape，也不能给旧缓存补一个true标签后复用。缓存生成配置为False或缺少可验证来源时，在新目录从官方权重重新导出任务6..9的正式特征、从空buffer重新开始task6。旧结果保留供审计，不删除、不覆盖、不续接到v2。

## 3. 实现与修复规则

不得大幅重构模型、更新全部依赖到最新版，或用替代模型/假数据绕过问题。尽量采用小补丁和独立脚本，保留原文件diff。修改分为：工程修复、论文公式对齐、未公开细节补充。每个影响优化轨迹的补丁都需有测试和记录。

必须核查并处理：

1. **路径与设备**：移除对作者`/home/fyu/...`、`/work`的强依赖，通过参数或环境变量解析。缺GPU时禁止`safe_device`静默回退CPU并将长跑称为正式实验。校验CUDA_VISIBLE_DEVICES与EGL设备映射。
2. **权重前缀与加载**：`_strip_module_prefix1`无module前缀分支可能不返回state_dict；测试有/无前缀都正确。评估脚本不能无条件删key前7个字符。空间前端必须完整来自同一个官方权重，时间解码器/头必须完整来自待测阶段权重。输出missing/unexpected keys与模块级加载覆盖率；不得靠strict=False静默保留随机可训练层。
3. **IFA正参考索引**：源码将当前任务last_feat按batch_mask选择，却从混合batch第一个样本取自身语言参考。用旧样本位于batch首位的单元测试重现，然后让每个当前任务样本配对其自身语言参考。使用实际[B,T,M,E]形状及language索引0，不能凭错误注释改tensor轴。
4. **IFA归一化**：按论文式(4)对参与的任务对平均，不是按任务对数量额外放大正则项。BC保持混合batch均值；IFA对当前样本及所选任务对按明确规则平均。无可用任务对或无当前样本时IFA为0。不要新增晚期激活、权重调度或排斥所有旧任务。
5. **相似度OOM**：可用分块精确计算，或利用“所有归一化向量两两内积的均值=两侧归一化向量均值的内积”的恒等式。小tensor测试结果等价；禁止为了跑通而抽样改变IFA任务对，近似误差与tie handling要记录。
6. **Loss日志**：修复在batch循环内部反复除以len(loader)的日志问题。分别记录BC、未加权IFA、加权IFA、总Loss；日志采集不得新增随机采样或改变实际Loss。Loss不作为能力判据。
7. **最终checkpoint**：原训练只保存70、75、80、85、90、95等epoch快照，需要另外保存epoch_index=99后的最终模型。不能用ep95冒充100 epoch。
8. **训练/评估接口**：明确特征导出、潜特征训练、完整策略闭环推理三个路径。`CompletePolicy.forward()`返回潜特征是当前导出设计，不应机械“修复”为策略输出。评估要走完整策略的get_action路径。不要因改变.train()/.eval()而在导出时无声改变增广或随机性；先记录并尽量沿用发布导出行为。
9. **正式评估控制流**：不运行原test.py中等待缺失文件的无限busy-wait。只评估本run manifest明确登记的四个阶段final checkpoint。文件缺失要报错并退出，不能跳过后仍汇总完整指标。
10. **成功语义**：检查环境done与任务成功的映射，确保计数与原LIBERO成功条件一致，不能把超时也计为成功。若接口版本引入terminated/truncated差异，做显式兼容并记录。

11. **模态配置（v2）**：按第2.3节对三入口统一显式override。空间编码阶段须执行`torch.cat([lang_tok, vision_mod, state_mod], dim=2)`的既有分支；不得在完整策略中把所有消融分支删除或悄悄改变其他实验默认值。
12. **模态语义与缓存（v2）**：建立并验证modal_contract。训练数据载入前检查feature manifest，评估建模前检查阶段checkpoint合同。BriefPolicy上的True字段仅为声明，必须与实际已验证缓存配对。
13. **缺失元数据处理（v2）**：仅缺原始`cfg.policy.use_state`时记录`DISCLOSED_GAP`，不再据此BLOCKED。真正缺失状态权重、权重尺寸不匹配、缺少规定状态观测或模态测试失败才是技术阻塞；禁止随机初始化被冻结的状态分支后继续正式长跑。

若源码与这里描述不符，以固定commit实际读取结果为准，先输出差异。第2.3节的状态配置选择已经明确，不需重复请示；不在v2授权范围内的新科学协议选择仍需停止正式长跑并报告，不擅自设计替代实验。

## 4. 阶段A：环境、数据与权重预检

检查已有环境、数据与模型目录，不猜测本机绝对路径，不覆盖用户已有环境。优先按README的Python3.8与requirements创建独立环境。记录Python、torch、CUDA、驱动、GPU、MuJoCo、robosuite、robomimic、transformers、peft等实际版本。

requirements可能存在需要验证的兼容关系；先实际检查import和关键调用。确需兼容补丁时，另存环境lock和最小差异，说明是否改变数值/环境行为。不能为了导入成功而替换策略架构或环境任务定义。

明确定位：
- GOAL演示HDF5；
- BDDL任务定义；
- 官方init-state文件；
- README链接提供的官方GOAL多任务预训练权重；
- CLIP/tokenizer与必要模型缓存。

生成data_manifest，包含实际排序后的10个任务名称、索引、前6/后4分组、每个增量任务的10条demo ID、窗口数、数据文件大小与哈希。`get_dataset()`当前意图是选择dataset.demos的前10个；若robomimic版本不支持现有调用，适配后要验证索引、缓存和len(dataset)全部只覆盖这10条演示，而不是只改n_demos显示值。

权重必须记录SHA-256、来源、checkpoint.cfg与关键shape；确认是GOAL而不是OBJECT，且空间前端与下游模型兼容。仅文件名包含goal不是完整证据。原预训练seed无法确认时记录unknown，不把它当作42。

遇到GPU/数据/官方权重/必要依赖无法取得时：完成能完成的静态测试与脚本输出，写`BLOCKED.md`列出已检查路径、失败命令、缺失资源和恢复命令；不得生成任何虚构训练曲线或成功率。

## 5. 阶段B：隔离smoke test和特征导出

先在独立smoke目录执行，不复用正式buffer或模型状态，不将smoke更新接入正式训练。

最低测试：
- 导出少量真实8步窗口，确认feature为预期[1,8,4,512]、actions对齐、float与数值有限；
- 用真实潜特征执行至少2次优化器step，验证梯度、参数更新、mask和LR；第一个warmup step可为0，但不能据此误判后续都不更新；
- 验证只有时间解码器和策略头更新，冻结参数和权重哈希不变；
- 合成混合任务batch测试IFA正参考、空任务对、多任务对平均；合成测试仅用于软件验证，不能替代正式数据；
- 完整策略加载一个真实权重，执行至少1个真实环境闭环episode，确认图像、状态、语言、8步历史、动作shape和环境成功计数路径；成功率为0不自动代表代码错误；
- 分别核查潜特征前向和完整策略前向在相同预处理、非随机测试条件下的一致性；不要拿增广缓存强行与无增广观测作逐位一致要求；
- checkpoint保存/重载与epoch边界恢复测试；
- 指标单元测试（见第8节）。
- **v2模态配置测试**：三个真实入口均得到布尔`use_state=True`与`use_language=True`，合同相同；BriefPolicy仅补字段而使用旧缓存的情况必须被拒绝。
- **v2状态token等价测试**：在确定性测试模式和相同已预处理真实观测上，确认`H[:, :, 3, :]`等于按`joint_states -> gripper_states`、原state_encoder、原state_scale/state_shift计算的state_mod，而不是视觉替身；前三个token的身份与原顺序一致。
- **v2状态依赖测试**：固定图像、语言和所有随机状态，只改变关节/夹爪输入；在若干非退化输入上第四token应响应，前三token保持一致。用False分支作一次软件负对照时，改变状态不应影响送入decoder的H。负对照不是正式训练消融，也不用于根据成功率选开关。
- **v2路径与权重测试**：相同预处理和确定性条件下，导出H后用BriefPolicy的时间解码器/头得到的GMM参数，应与CompletePolicy用同一下游权重得到的GMM参数一致。比较分布参数，不比较两次随机采样动作；不要调用当前BriefPolicy在eval时不返回值的forward并据此误判模态故障。
- **v2审计隔离**：上述测试可暂时关闭dropout/随机增广，但需保存并恢复各模块模式和RNG；不能借单测改变正式特征导出的既定随机性。若用合成输入或依赖替身，只能作为额外软件单测，真实权重/真实数据测试仍须执行。
- **v2错误分类**：若扰动测试无响应，检查加载覆盖率、状态层数值、ReLU死区及容差，保存真实诊断；不能伪造测试通过。语义/兼容性测试通过不代表旧state_encoder必然经过有效预训练，也不证明闭环收益。

通过后重新从官方权重与seed42启动正式流程，使用全新RUN_DIR。

正式特征提取：参照save_feature_goal.sbatch，batch=1、仅0-based任务6..9、每任务10条演示。保持作者输入预处理与导出模式，记录增广开关、模块train/eval状态和随机种子，不新增增广。使用同一官方空间前端，确保导出过程没有optimizer更新。

保存feature_manifest，包括checkpoint哈希、任务映射、source demo/window元数据、导出顺序、tensor shape/dtype、导出配置和文件数。元数据用于追踪与后续实验，不用于本轮改变选样。

v2缓存规则优先：False或缺乏可核验模态来源的旧缓存不可复用；先使用新目录重新导出。仅对已经按v2生成、全部fingerprint及文件完整性匹配的缓存允许恢复。正式buffer初始为空，正式导出不从smoke目录复制。已经发生的旧模态正式训练不能跨模态resume；须保留旧目录并从官方预训练权重重新开始v2。

## 6. 阶段C：单seed四阶段连续训练

训练链必须是：

```text
官方GOAL六任务预训练权重
  -> task_id=6: 100 epochs -> final_6
  -> task_id=7: 100 epochs -> final_7
  -> task_id=8: 100 epochs -> final_8
  -> task_id=9: 100 epochs -> final_9
```

同一策略沿链继承权重；不能每个任务都重新加载pretrain。每阶段optimizer/scheduler按源码start_task重新初始化。数据只由当前任务全部潜窗口与buffer中的旧增量任务窗口构成。

保持源码shuffle与DistributedSampler语义，world_size=1，每epoch调用set_epoch。不要引入新旧样本重加权或按任务均匀采样。记录每epoch实际新/旧样本出现次数、重复次数、实际旧样本占比和总optimizer steps。

每阶段100 epoch结束后：
- 保存阶段最终权重、训练状态和parent checkpoint哈希；
- 按原随机逻辑用seed42及rate0.5将当前任务子集写入buffer；
- 保存各任务buffer文件清单、计数、字节数和选择日志；
- 执行下一任务前检查buffer不包含未来任务或重复追加的当前任务。

buffer更新要幂等：恢复运行时不能再次把同一任务加入两次。采用临时目录/清单与完成标记的事务式写入，只有写入完整后才标为完成。

保存滚动epoch-end恢复点及四个stage-final恢复点，避免无界保存所有完整AdamW状态。恢复点至少包含模型、optimizer、scheduler、epoch/global_step、Python/NumPy/torch CPU/CUDA RNG、sampler状态、完成任务、配置和feature/buffer manifest。明确保证的是epoch边界恢复；若不能恢复DataLoader中间状态，不声称支持精确的batch中途续训。

评估不应扰动训练随机数或更新参数。本轮优先先保存全部四阶段final，再在单独进程统一评估；训练期间不运行用于模型选择的闭环测试。

## 7. 阶段D：正式闭环评估

不需要也不允许用Loss决定哪些模型被测试。只测试四个预注册的stage-final模型。

按下表执行，每个单元20个episode：

| 已完成增量任务 | 当前checkpoint评估的任务索引 | episode数 |
|---|---|---:|
| 6 | 6 | 20 |
| 7 | 6,7 | 40 |
| 8 | 6,7,8 | 60 |
| 9 | 6,7,8,9 | 80 |

主评估总计200个episode，不包括smoke test。预训练任务0..5不进入这个主矩阵；本轮也不额外开展它们的保持评估。

评估加载方式：先完整加载同一个官方预训练空间前端，再用当前阶段的时间解码器/策略头覆盖对应参数。严格检查两部分来源及覆盖率，禁止把四个阶段都评成原始预训练模型。

v2附加要求：构造CompletePolicy时显式启用状态与语言，并在权重覆盖后再次检查合同。旧checkpoint缺cfg字段不能覆盖本轮True。用第4个token的状态依赖测试确认线上spatial_encode仍为状态分支；只有tensor尺寸正确不足以通过验收。

沿用原策略动作产生方式，不擅自把采样动作改为GMM均值/确定性动作，不修改动作尺度、相机预处理和历史初始化。

正式eval.n_eval显式设为20。当前源码内部env_num为10，实际需要2批；max_steps=600，并保留5步零动作初始化与每episode历史重置。每个任务使用官方init-states前20个不同索引，必须检查数量足够，不允许取模制造重复还宣称20个不同初态。

本轮补充随机数约定：每个(stage,task)评估单元开始时，将策略评估RNG重置为42；环境批次沿用源码see=42与see+i，即两批42和43。保存实际worker seeds、初态索引与哈希。43是评估内部派生种子，不是第二个训练seed。此隔离约定是本轮预注册的可重复性决定，不声称正文给出了这些细节。

每个episode逐条记录：run_id、training_seed、stage、task_name/id、checkpoint_sha、eval_seed/worker_seed、init_state_id/hash、success、policy_steps、终止原因、运行时长、异常。基础设施错误应显式记录并只按预先规则重试同一初态，不能把失败episode换成更容易的初态。没有完成规定数量时不要输出完整汇总。

## 8. 阶段E：按论文定义汇总

把任务6,7,8,9重新映射为增量阶段/任务1,2,3,4。令M=4，r[i,j]为学完第i个增量任务后在第j个增量任务上的闭环成功率，先用0..1存储。上三角未评估单元用NaN，不用0。

```text
FWT = mean_j r[j,j]
NBT_j = mean_{i=j+1..M} (r[j,j] - r[i,j]), j<M
NBT = mean_{j=1..M-1} NBT_j
AUC_j = mean_{i=j..M} r[i,j]
AUC = mean_{j=1..M} AUC_j
FinalAvgSR = mean_{j=1..M} r[M,j]   # 本轮辅助指标，不等同论文AUC
```

不要把AUC替换为sklearn曲线面积、全下三角直接均值或最后一行均值；不要把FWT改为训练曲线面积或相对初始模型增益；不要给最后一个任务计算除以0的NBT；不要将负NBT强制截断为0。

必须加入纯软件单元测试，以下矩阵不是实验结果：

```text
[[0.8, NaN, NaN, NaN],
 [0.7, 0.9, NaN, NaN],
 [0.6, 0.8, 0.7, NaN],
 [0.5, 0.7, 0.6, 0.8]]
```

其FWT=0.8，NBT=0.15，AUC=0.725，FinalAvgSR=0.65。另设后续表现提高的测试，验证负NBT不会被截断。

报告单seed点估计和每任务成功次数/20。不得把20个episode当成20个训练seed，也不得输出三seed标准差。可以清楚标注rollout统计不确定性，但它不能替代训练随机种子间方差。

论文GOAL MLR+IFA三seed结果是FWT 80.0±2.5、NBT 6.9±0.9、AUC 77.2±1.8（百分点）。只能列为参考，不能作为本轮测试通过阈值，不能为了靠近这些值修改epoch、初态、seed、gamma或选择最好快照。

## 9. 脚本接口与输出交付

以下文件名与CLI是需要你新建的目标接口，不是声称官方仓库已有的API：

```text
scripts/run_goal_seed42.sh
scripts/eval_goal_seed42.sh
scripts/summarize_goal_seed42.py
configs/experiments/goal_seed42_paper_aligned_state_v2.yaml
# 配置路径可按实际Hydra结构调整，但README中必须给出真实有效路径
```

提供preflight、smoke、full、resume四种明确模式；--help和--dry-run可用。正式执行需要路径参数，例如：

```bash
bash scripts/run_goal_seed42.sh \
  --mode full \
  --pretrained /ABS/PATH/TO/OFFICIAL_GOAL_CHECKPOINT.pth \
  --data-root /ABS/PATH/TO/DATASETS \
  --bddl-root /ABS/PATH/TO/BDDL \
  --init-states-root /ABS/PATH/TO/INIT_STATES \
  --run-dir /ABS/PATH/TO/NEW_RUN_DIR
```

先发现真实路径并完成schema映射，再执行生成的真实命令；不要原样执行上述占位路径。真实底层入口仍基于save_feature.py、lifelong_adaptation.py与完整策略闭环评估；保留单GPUtorch.distributed启动需求。脚本应set -euo pipefail，保存退出码、开始/结束时间和完整命令，不因echo“完成”掩盖子进程失败。

建议RUN_DIR至少包含：

```text
protocol_audit.md
resolved_config.yaml
resolved_export.yaml
resolved_adaptation.yaml
resolved_eval.yaml
modal_contract.json
modality_tests.json
environment.json
requirements.lock.txt
git_status.txt
source.patch
data_manifest.json
feature_manifest.json
pretrained_manifest.json
features/
buffer/
buffer_manifests/
checkpoints/
logs/train.jsonl
logs/commands.log
evaluation/rollouts.jsonl
evaluation/success_matrix.csv
metrics.json
report.md
RUN_STATUS.json
```

如果没有可用GPU执行环境，不要启动付费资源或声称训练已经在后台进行。只交付已实际生成的代码、测试结果、BLOCKED信息与启动命令。若确实启动了已有调度系统的作业，记录真实job ID与日志位置，并如实区分SUBMITTED/RUNNING/COMPLETED。

## 10. 验收条件

v2额外前提：已执行三入口True配置检查、真实状态token语义/依赖测试、完整与潜策略确定性分布参数一致性测试；feature、buffer、阶段checkpoint的模态合同匹配；无False缓存或未证实缓存混用。官方checkpoint原始模态未知必须在报告中披露，不填推测值；这个已披露缺口单独存在时不阻塞本轮完成状态。


标为COMPLETED需要同时满足：固定commit与全部补丁存档；正式seed=42；4阶段每阶段100 epoch且权重连续继承；每任务只用规定10条演示；导出、训练和评估权重来源可追踪；随机回放与buffer状态记录完整；4个最终checkpoint存在且可加载；200条正式闭环记录完整；4×4下三角矩阵有10个有效单元；FWT/NBT/AUC按原定义计算；没有Loss选样、测试集反馈、挑最好seed/epoch或隐藏额外训练。

如果训练完成而评估未完成，状态应为TRAINING_COMPLETED_EVAL_PENDING，不要写实验已完成。如果任何source_gap未解决，报告仍可给出本轮结果，但必须标注paper-aligned implementation with disclosed gaps，不能声称完全精确复现作者实现。

最终给用户返回：实际执行命令、实际状态、阶段/epoch进度、GPU和时间、修改清单、测试摘要、checkpoint路径、闭环矩阵、三指标，以及未解决的协议缺口。没有运行结果的项目用未运行/未知，不填推测值。

## 来源索引（供Codex核查）

S1. 用户附件1：Yu 等，《Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment》，CVPR 2026。第4页式(4)与MLR；第5页协议；第6页指标与Setup；第7页存储比例。

S2. 官方仓库固定commit的README、sbatch脚本、配置和上述Python源码。当前代码审计确认batch_size=18、eval.n_eval=2、GOAL gamma=0.72、warmup_steps=500、发布评估遍历70..95。

S3. 公开arXiv v1补充材料，附录A.1和B.1（另一个明确标注的版本）：

```text
https://arxiv.org/html/2603.10929v1
```

来源优先级：附件中明确的实验参数与公式 > 明确标注版本的补充材料 > 固定commit补充未写明细节。冲突必须写进protocol_audit，不能默默合并成一个声称完全忠实的原协议。


## v2补充来源与修正范围

本次直接核对的是固定commit，不以默认分支的未来变动替代：

```text
libero/configs/policy/complete_policy.yaml      use_state=False
libero/lifelong/models/complete_policy.py      spatial_encode与get_action
libero/configs/policy/BriefPolicy.yaml         没有模态字段
libero/lifelong/models/brief.py                只处理latent
libero/lifelong/models/base_policy.py          导出actions/feature；训练读取feature
save_feature_goal.sbatch                      policy=complete_policy
lifelong_adaptation_goal.sbatch                policy=BriefPolicy
test.sbatch                                   policy=complete_policy
```

论文附件1第4页MLR与第5页Implementation Details支持本轮状态模态选择；源码支持上述分支与调用链。原始checkpoint内部cfg缺失这一情况来自用户执行环境审计，须由Codex在当地完整核验；不能声称本指令已经直接读取了用户服务器上的权重。

本文件只改变v1遗漏的模态合同、相关缓存/元数据/测试与运行标签。原有学习率、batch、gamma、随机回放、100epoch、200episode及指标合同不因本修正而再次搜索或调优。

Hydra 1.2的`++key=value`语义已对照官方文档核查（已有字段覆盖、缺失字段添加）：

```text
https://hydra.cc/docs/1.2/advanced/override_grammar/basic/
```
