# E1：LIBERO-GOAL state_v2 同初态闭环配对审计

## 结论与执行范围

下一步只分析已完成基线的200条正式闭环记录，区分“既有成功被保留”与“失败转成功抵消成功转失败”。不新增训练、不新增rollout、不载入checkpoint二进制、不引入Loss选样、不改变任何已完成实验参数。

本任务是**回顾性描述分析**，不是新的持续学习方法实验。低NBT与较多结果翻转可以同时出现，但翻转不能直接证明回放造成任务内错误转移。

附带脚本 `analyze_paired_closed_loop.py` 已实现本合同；`test_paired_closed_loop.py` 只使用合成软件测试数据，不包含真实配对分析结果。执行者必须在真实证据目录上运行后才能提交真实报告。

## 1. 固定来源

- 发布仓库：`uhhj/lifelong_mlr_ifa_minrep_20260821_193226`。
- 发布分支：`MLR`；本分析固定发布提交：`eb15698ae8ee356f6098b9777e5c7dad21450b0e`。
- 原实验标签：`goal_seed42_mlr_ifa_paper_aligned_state_v2`。
- 原作者基点：`a721807f2f7fa7f9833dcc85c592d78e9accd00b`，不要与发布提交混淆。
- 服务器现有证据根：`/data/mlr_ifa/experiments/goal_seed42_mlr_ifa_paper_aligned_state_v2/run`。
- 发布仓库中的对应证据根：`reports/goal_seed42_state_v2/run`。

优先使用已经存在的服务器证据或发布工作区证据，不重复下载大文件。执行者须确认实际路径存在；本地Windows证据副本也可作为输入，但要记录其来源，不能仅把常量commit写进报告就宣称Git版本已核验。只读记录实际HEAD和来源；若当前分支已变化，可用独立固定commit快照，不reset、clean或覆盖工作区。

### 必需轻量输入

相对于证据根：

```text
evaluation/rollouts.jsonl
metrics.json
final_acceptance.json
checkpoints/manifest.json
modal_contract.json
source.patch
```

脚本还会记录以下已存在文件的输入哈希，但不依赖它们启动模型：

```text
evaluation/success_matrix.csv
resolved_eval.yaml
data_manifest.json
pretrained_manifest.json
logs/commands.log
```

**不需要**：GPU、CUDA、MuJoCo、PyTorch、演示HDF5、latent tensor、模型权重。不要调用原训练driver的preflight/full/resume/eval；该driver的HEAD检查针对训练复现，与本次纯数据分析无关。

## 2. 数据检查合同

检查必须先于结果解释，不自动去重、插补或跳过异常记录。

1. 正式记录总数为200；10个合法下三角单元，每个20条；任务与阶段均在6—9且task_id<=stage。
2. 每单元初态ID为0—19且哈希互异；同任务的同一初态ID在所有阶段对应同一哈希。
3. `kind=FORMAL`，run_id正确，training_seed=eval_seed=42；success为JSON布尔值；exception=null。
4. 与实际评估实现一致，env_batch_seed=42+floor(init_state_id/10)，worker_seed=env_batch_seed+(init_state_id mod 10)。跨阶段这些配对元数据必须一致。
5. 每条记录的checkpoint_sha与对应阶段manifest和原验收记录一致，epoch_index=99；modal_fingerprint与合同一致。
6. 策略步数为1—600；成功/超时终止语义一致；失败达到600步。恰在第600步成功仍是成功，不能按步数倒推成功标签。
7. 从200条记录重新计算四阶段矩阵、FWT/NBT/AUC/FinalAvgSR及逐任务分量，与metrics.json和final_acceptance.json核对，容差1e-12。
8. source.patch字节哈希与合同及验收记录一致；全部读取的源文件在分析前后哈希不变。

失败时生成独立输出目录的BLOCKED.md并以非0退出，不修改原始证据或COMPLETED状态，也不运行额外rollout补齐。

检查范围是**日志和轻量证据内部一致性**，不等于重新核验服务器checkpoint文件、不等于重新验证环境初态二进制、不等于独立训练复现。

## 3. 两种配对视角

匹配键是 `(run_id, training_seed, task_id, init_state_hash)`；stage标明前后观察。init_state_id、eval_seed和worker信息用于交叉检查，不能只按JSONL行号或“第几个成功”配对。

### 3.1 相邻训练阶段：哪次任务学习前后出现了结果变化？

| 前后阶段 | 被评估旧任务 |
|---|---|
| 6→7 | 6 |
| 7→8 | 6、7 |
| 8→9 | 6、7、8 |

共6个比较，每个20对，共120条配对记录。

### 3.2 相对于刚学完：最初成功是否一直保留？

| 被评估任务 | 参考阶段与后续阶段 |
|---|---|
| 6 | 6→7、6→8、6→9 |
| 7 | 7→8、7→9 |
| 8 | 8→9 |
| 9 | 没有后续阶段，记NA |

也是6个比较、120条配对记录。

两种视角存在重叠：最终240条视角记录只对应180条去重后的“任务/前后阶段/初态”配对，并复用原来的200条episode。不得把它们当作240或180次新增独立试验，不得把两种视角汇总成新的样本量。

## 4. 每个比较必须报告的四类结果

| 标签 | 前→后 | 解释 |
|---|---|---|
| SS | 成功→成功 | 两次观察都成功 |
| SF | 成功→失败 | 观察到原成功结果不再出现 |
| FS | 失败→成功 | 观察到新成功结果 |
| FF | 失败→失败 | 两次观察都失败 |

四类计数总和必须为20。对每个比较输出：

- 成功→失败份额：SF/20；其对成功率的减少贡献为100×SF/20个百分点。
- 失败→成功份额：FS/20；其增加贡献为100×FS/20个百分点。
- 净成功率变化：100×(FS−SF)/20个百分点，必须等于矩阵对应单元的差值。
- 结果翻转份额：(SF+FS)/20。
- 原成功保留率：SS/(SS+SF)，分母为0时必须为NA/null而不是0。
- 原成功丢失率：SF/(SS+SF)，同样处理零分母。

这组新统计是**本轮定义的描述性诊断量**，不是原论文额外报告的指标，不替代FWT/NBT/AUC。用“观测成功→失败”而不是“已证实技能被回放伤害”等因果措辞。

### 4.1 与原NBT的可验证关系

只用from_learned视角，对每个任务先平均所有后续阶段的SF/20和FS/20，再对任务6、7、8等权平均，得到L和G。

```text
100×(L−G) = 原论文口径NBT（百分点）
```

不要把全部120条记录直接求总平均替代分层平均；较早任务有更多后续比较，会被不当加权。保留负NBT，不做截断。

本分析回答“净变化较小是否伴随两种相反变化”，不依据经验阈值给稳定性贴高/低等级，不设置让预想假设必定通过的门槛。

## 5. 初态结果历史与步数辅助分析

为每个任务的20个初态建立完整成功序列，总共80行：任务6长度4、任务7长度3、任务8长度2、任务9长度1。

记录初次失败数、所有观测均成功/均失败/混合的序列、相邻观测翻转次数、初次成功后曾观察到失败的初态，以及其中最终又成功的数量。仅把它们称为已观察到的结果历史。

任务9没有后续观测，遗忘、恢复和后续稳定性必须为NA/null。不要把85%的单次成功率解释为抗遗忘能力，也不要把“初次失败”直接解释为可通过增加epoch解决的欠拟合。

步数只做辅助：在同一配对两端均成功的SS子集上，报告样本数和`after_policy_steps - before_policy_steps`的中位数。负数只表示该配对子集的观测完成步骤较少。不能把失败的600步当作成功完成耗时，不能比较两端不同成功子集的均值后宣称效率提升，也不用GPU并发受影响的duration_seconds评价策略质量。

## 6. 必须保留的解释边界

- 每个checkpoint对每个初态只有一次已记录执行；这不是反复执行估计出的条件成功概率。
- 初态、seed配对不保证逐episode随机动作序列完全相同；共享随机数流、不同执行长度及批内其他环境会影响随机数消耗。只做描述分析，不做未经验证的显著性或因果检验。
- 没有无回放对照、不同选样对照或阶段行为事件，本轮不能证明回放有害、定位抓取/搬运/释放问题、识别失败责任窗口，或证明“任务内误差转移”。日志只有成功/超时不能支持这些推断。
- 本分析不重新运行初态、录制视频或改变评估协议。不调用模型推理生成额外材料冒充既有证据。
- 所有任务都报告，重点讨论6、8但不选择性省略7、9。
- 原测试结果已经用于回顾分析。不能把失败初态列表交给训练采样器，不生成样本权重，不将测试初态映射到训练窗口并优先回放。任何利用这些结果启发的方法，后续评价必须区分开发反馈与独立确认测试。
- 不自动决定下一轮ratio/样本策略/训练长度，不启动下一组实验。先交付本次分析。

## 7. 执行命令

将执行包放到一个新的工具目录。下面的TOOLS和OUT是**本次新建位置**，不是已存在产物路径；不得覆盖同名目录。EVIDENCE是已完成基线。

```bash
TOOLS=/data/mlr_ifa/analysis_tools/MLR_E1_Paired_Audit
EVIDENCE=/data/mlr_ifa/experiments/goal_seed42_mlr_ifa_paper_aligned_state_v2/run
OUT=/data/mlr_ifa/analyses/goal_seed42_state_v2_paired_audit_v1

# 先将本包解压到TOOLS，确认下列两个.py文件位于其中。
# 使用Python >=3.8；不需要激活机器人训练环境或安装任何第三方包。
python --version
export PYTHONDONTWRITEBYTECODE=1
python -m unittest discover -s "$TOOLS" -p 'test_*.py' -v
python "$TOOLS/analyze_paired_closed_loop.py" \
  --evidence-root "$EVIDENCE" \
  --out-dir "$OUT"
```

也可以把EVIDENCE改成固定commit发布工作区中的`reports/goal_seed42_state_v2/run`。Windows命令使用实际确认的路径并加双引号；同一Python脚本不依赖POSIX路径。

如果文件系统或Python版本不满足要求，只做最小兼容修正并记录；不修改受审计证据。不要运行原有`run_goal_seed42.sh --mode full/resume/eval`。

## 8. 交付物和验收

```text
paired_audit_v1/
  success_matrix_recomputed.csv
  metrics_recomputed.json
  paired_transitions.csv        # 240行；保留两种视角标签
  paired_summary.csv            # 12行，6+6比较
  initial_state_histories.csv   # 80行；任务9后续指标为NA
  nbt_decomposition.json
  REPORT.md
  audit_acceptance.json
```

成功验收条件：

- 200条正式输入、10单元×20条；80个任务—初态分组。
- 120条adjacent配对、120条from_learned配对、180条去重后的跨阶段配对。
- 每个比较SS+SF+FS+FF=20；净差与成功率矩阵相符。
- 原四项指标及逐任务分量复算相符；分解后的L−G与原NBT一致。
- 源证据读取前后哈希一致，原COMPLETED状态、缓存、checkpoint和评估记录均不修改。
- 新训练步骤0、新rollout0；工具不载入权重。
- 报告准确披露匹配条件、一次执行的限制、共享episode非独立性、阶段事件缺失及测试用途边界。

`audit_acceptance.json=PASS`指分析与轻量输入一致性通过；不继承或扩大原服务器二进制验收范围。

## 9. 可直接发给Codex的执行指令

```text
执行附件《MLR_E1_Paired_Audit_Codex.md》，使用同包的
analyze_paired_closed_loop.py与test_paired_closed_loop.py。

本轮只做E1：既有闭环记录的同初态配对审计。
原实验state_v2已经完成，不启动任何训练或环境执行。

1. 只读定位服务器run目录或固定发布commit的轻量证据目录，
   记录真实来源和可确认的Git版本，不覆盖工作区。
2. 在新工具目录运行附带软件测试；它们只检验程序，不是实验结果。
3. 在基线run外新建输出目录，用纯标准库脚本分析真实200条记录。
4. 从日志复算矩阵与指标，再输出两种视角的配对统计、
   初态结果历史、NBT分解、SS子集步数变化及报告。
5. 检查所有输入前后哈希不变、计数和恒等式成立，
   交付报告、文件路径、真实命令及退出码。

严格禁止：
修改原run、重新full/resume/eval、追加rollout、补造缺失记录、
加载checkpoint、根据Loss判定遗忘、生成训练采样权重、
把结果翻转称为已证明的回放伤害、自动启动新实验或push远端。

遇到数据不一致时输出BLOCKED，保留证据，不自行修补输入。
成功后停止。本轮只报告观察，不依据原测试数据挑选下一轮最佳超参数。
```

## 10. 源码核查依据

本方案根据固定发布提交中实际读取的以下文件设计：

- `reports/goal_seed42_state_v2/run/evaluation/rollouts.jsonl`：记录字段与配对标识。
- `reports/goal_seed42_state_v2/run/final_acceptance.json`：四阶段、单元数、checkpoint关联及计数。
- `reports/goal_seed42_state_v2/run/metrics.json`：原始指标及逐任务分量。
- `libero/lifelong/runbook_v2.py::eval_main`：每单元20初态、10环境×2批、seed与执行上限的实际设置。

新增的配对诊断与软件实现是本次分析设计，不是原论文或仓库已存在的新增方法结果。
