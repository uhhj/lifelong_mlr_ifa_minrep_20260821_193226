# OBJECT seed42：RA-Coreset + COVR 完整闭环实验执行方案

结论：建议采用“原始 RA-Coreset 完整训练协议 + 在线反事实恢复 + 固定预算辅助监督”的方案，在 OBJECT seed42 上完成 Task6–9 四阶段持续学习，并用原论文的 200 条正式闭环评估检验 COVR 是否带来增益。

已核对远端 `MLR` 分支的训练、选样、GMM 策略头、评估代码及最新状态恢复实验。

下文将用户所说的 RC-Coreset 视为此前已验证的 RA-Coreset。仓库中对应的实际实验方法名为 `ra_coreset`。

## 一、实验目标与冻结基线

本次只新增一个正式实验组：

\\[ \boxed{\text{MLR}+\text{IFA}+\text{RA-Coreset}+\text{COVR}} \\]

不重新训练 Random、普通 Coreset 或 RA-Coreset 基线。

现有 OBJECT seed42 的正式成绩如下：

| 方法               | AUC    | FWT    | NBT      | OldFinal | FinalAll |
| ---------------- | ------ | ------ | -------- | -------- | -------- |
| Random + IFA     | 66.15% | 66.25% | -0.56 pp | 70.00%   | 63.75%   |
| Coreset + IFA    | 68.13% | 65.00% | -5.83 pp | 73.33%   | 66.25%   |
| RA-Coreset + IFA | 71.56% | 70.00% | -3.06 pp | 65.00%   | 63.75%   |

来源：远端 [OBJECT RA-Coreset 正式实验报告](https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226/blob/MLR/reports/object_ra_conditional/launch/FINAL_REPORT_20261020.md?utm_source=chatgpt.com)。NBT 按现有报告口径，越低表示后续遗忘越少。

本次的主要判断标准是：

\\[ \Delta \mathrm{AUC} = \mathrm{AUC}\_{RA+COVR}-71.5625\\% \\]

要求 AUC 严格提高，同时完整报告 FWT、NBT、OldFinal、FinalAll，不能只展示改善的任务。

### 原论文协议中必须保持不变的部分

| 项目         | 本次固定设置                                      |
| ---------- | ------------------------------------------- |
| 数据集        | LIBERO-OBJECT                               |
| 随机种子       | 42                                          |
| 初始模型       | 官方 OBJECT 六任务预训练权重                          |
| 增量任务       | Task6 → Task7 → Task8 → Task9               |
| 每任务原始演示    | 10 条                                        |
| 输入         | 双相机视觉 + 语言 + 机器人状态                          |
| 序列长度       | 8                                           |
| 主网络        | 原 Temporal Transformer + GMM Head           |
| 每任务训练      | 100 epoch                                   |
| Batch size | 10                                          |
| 优化器        | AdamW，学习率 1e-4                              |
| 调度器        | 当前已审计的逐 optimizer-step 线性调度，warmup 500 step |
| IFA        | 权重 0.1，OBJECT gamma 0.3                     |
| RA 入库配额    | 不改变                                         |
| 正式评估       | 20 次/任务/阶段，最多 600 个策略控制步                    |
| 评估模型       | 每阶段 epoch99 最终模型                            |
| 其他 seed    | 不启动                                         |

四阶段共 400 epoch，按已验证的缓存规模，原训练循环应执行 105,500 次 optimizer step。

需要明确区分两种协议一致性：

- 正式闭环评估协议一致： 完全可以保持，与现有 RA 实验直接比较。
- 训练信息预算一致： COVR 需要额外旧任务仿真交互，因此不能声称与原始离线 MLR 在训练信息量上完全相等。

这个差异必须在论文中披露，并记录全部额外仿真成本。

## 二、先解决当前代码中的三个关键问题

### 2.1 COVR 应使用在线状态分叉，而不是历史演示状态重建

这是本次方案最重要的修正。

仓库最近已经开展相近的 RCTR 实验：

- [RCTR Stage A 报告](https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226/blob/MLR/reports/rctr_goal_seed42_stageA/REPORT.md?utm_source=chatgpt.com)：历史演示轨迹无法严格重现，实验在分叉前停止。
- [RCTR Stage A2 报告](https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226/blob/MLR/reports/rctr_goal_seed42_stageA2/REPORT.md?utm_source=chatgpt.com)：改成当前策略在线运行时保存完整状态，恢复一致性测试通过，报告快照时完成9/333条分支。

RCTR A2 的探索发生在 GOAL seed42，不能直接证明 OBJECT seed42 已具备相同能力。

因此，本次 COVR 采用：

当前策略运行 → 即时保存完整仿真状态 → 观察最终成败 → 从已保存状态恢复 → 尝试反事实动作。

不从 HDF5 中间时间步重新构建旧演示。

这样避免了此前已发现的动作与观察错位、MuJoCo 历史版本差异和原始控制器隐状态缺失问题。

### 2.2 必须保持真实 GMM 动作执行方式

核对 [policy_head.py](https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226/blob/MLR/reports/object_ra_conditional/launch/source/libero/lifelong/models/policy_head.py?utm_source=chatgpt.com) 后，当前策略具有：

- 5 个 GMM 分量。
- 7 维连续动作。
- 推理时选择最大 logits 分量。
- 然后从该分量的 Normal 分布采样。

因此 COVR 不能简单地把模型整体改成均值动作策略。

本次规定：

正常执行始终使用原来的采样规则；只在被选中的分叉时刻，临时替换一个动作。

### 2.3 COVR 生成的样本必须兼容原 MLR 的潜特征训练

当前训练主要使用 `BriefPolicy`：

```
feature: [B, 8, 4, 512]
actions: [B, 8, 7]
```

实际 BC 损失使用窗口最后一个动作作为监督。

所以 COVR 不能把原生图像和动作直接塞进已有潜特征 DataLoader。

需要通过原来的冻结空间前端，将在线观察编码为同样格式的潜特征，再提供给时间解码器及 GMM Head。

这也意味着不必引入第二个大型策略网络。

## 三、确定 COVR 的完整算法

### 3.1 训练流程

Task6：RA-Coreset 原始训练

100 epoch，COVR 不启用；验证与现有 RA Task6 参数一致

Task7：前50 epoch 正常训练

使用任务7演示 + 任务6 RA 记忆

COVR 在线闭环采集与分叉验证

用当前模型执行旧任务6，搜集经验证的恢复动作

Task7：后50 epoch 联合训练

原 BC＋IFA 损失 + COVR 辅助损失

Task8、Task9 重复同一流程

分别检查所有已学习旧任务；每阶段独立采集

原论文协议的四阶段正式闭环评估

4 个 ep99 checkpoint、10 个评估单元、200 条 episode

将每阶段 COVR 采集时点固定在 epoch49 完成以后，因此训练总 epoch 数不会增加。

这不是论文公开参数，而是本次预先冻结的实验设计。不能看完 seed42 成绩再调整采集时点。

### 3.2 哪些任务需要执行 COVR？

| 正在学习的任务 | 原始训练数据                | COVR 检查的旧任务 |
| ------- | --------------------- | ----------- |
| Task6   | Task6 演示              | 无           |
| Task7   | Task7 + RA(Task6)     | Task6       |
| Task8   | Task8 + RA(Task6,7)   | Task6、7     |
| Task9   | Task9 + RA(Task6,7,8) | Task6、7、8   |

每个阶段使用当前冻结在 epoch49 的模型生成恢复数据，不能调用尚未训练完成的未来模型。

COVR 不要求新任务结束后额外训练十几个 epoch，而是在原有100 epoch内部使用剩余的50 epoch完成修复。

### 3.3 如何选择反事实实验的状态？

不采用随机挑选失败时间步，也不根据正式评估结果选择。

建议使用固定的、与正式评估不重叠的初态集合。

预注册的 COVR 数据采集设置

| 项目        | 设置                      |
| --------- | ----------------------- |
| 采集初态      | 每个旧任务初态索引20–25，共6个      |
| 独立诊断初态    | 索引26–31，不用于产生训练标签       |
| 候选决策步     | 40、120、240              |
| 单个任务-阶段组合 | 最多选择3个失败初态，每个1个分叉状态     |
| 选样依据      | 固定初态顺序、预设时间表和正常轨迹是否失败   |
| 数据限制      | 不使用正式评估初态0–19生成 COVR 监督 |

这些初态编号是新实验设计，不是已经验证存在的 OBJECT 数据事实。启动前必须检查每个任务初态文件至少有32项；不满足则 BLOCKED，不得改用正式评估初态补足。

对于初态20–25，分叉时刻依次固定为40、120、240、40、120、240。

先执行正常完整轨迹。如果正常策略失败且指定时刻存在有效状态，就将该状态列为候选。按初态编号升序最多取三个，不因为某个状态看起来更容易恢复而更换名单。

### 3.4 每个分叉状态尝试什么动作？

设当前 GMM 输出：

\\[ \\{(\mu_k,\sigma_k,\ell_k)\\}\_{k=1}^{5} \\]

其中 \\(\mu_k\\) 是第 \\(k\\) 个分量的7维均值。

在同一个完整仿真状态上执行六个分支：

| 分支     | 首个分叉动作                 |
| ------ | ---------------------- |
| Normal | 原 argmax 分量的 Normal 采样 |
| Mode 0 | 分量0的均值                 |
| Mode 1 | 分量1的均值                 |
| Mode 2 | 分量2的均值                 |
| Mode 3 | 分量3的均值                 |
| Mode 4 | 分量4的均值                 |

原 argmax 分量的均值同时充当一个必要对照：它能检查提升是否仅来自取消随机采样，而不是切换到了更好的动作分量。

分叉动作执行一次后，所有分支恢复原来的策略采样方式。

不要连续多步强制使用同一个 GMM 分量，否则同时改变了更长时间的控制规则，无法将差异归因于当前决策。

对于每个分支使用3份预先固定、相互独立的噪声序列。同一重复编号的所有分支共享后续采样噪声。

从分叉点继续运行到任务成功或总策略控制步达到600，取先到者。若分叉点在第120步，后续最多执行480步，不重新给予600步。

### 3.5 怎样判断动作真的具有恢复效果？

假设同一状态得到：

| 分支        | 重复1 | 重复2 | 重复3 |
| --------- | --- | --- | --- |
| Normal    | 失败  | 失败  | 失败  |
| Argmax 均值 | 失败  | 失败  | 失败  |
| 分量2均值     | 成功  | 成功  | 失败  |

那么分量2具有初步恢复信号。

本次预先固定以下训练样本准入条件：

1. 相对 Normal，至少出现2/3次配对的失败→成功，并且没有成功→失败。
2. 相对 argmax 均值，也达到同样条件。
3. 使用另外3份独立噪声重新认证，仍满足上述条件。
4. 实际执行的替代动作与对照动作不同，且动作有效、无 NaN/Inf。
5. 整个恢复过程通过状态一致性检查。

最多接受每个物理状态一个候选动作。如果多个候选均合格，选择训练筛查中配对恢复次数最多者，并以固定分量编号打破并列；然后独立认证，认证失败不寻找替补。

这样可以避免在多个候选中不断试到一个偶然成功的动作，随后把偶然结果写成方法收益。

需要注意：这些是工程与样本认证门槛，不是统计显著性检验。

### 3.6 成功后到底保存什么？

为了兼容现有 MLR，建议保存两个部分：

第一部分：经过认证的恢复动作。

\\[ (z_t,a_t^{\mathrm{recover}}) \\]

这是核心监督。

第二部分：成功续跑最开始的至多7步。

\\[ (z\_{t+u},a\_{t+u}),\qquad 1\leq u\leq7 \\]

每个 \\(z\\) 都通过对应时刻及其之前的最多8帧原生观察编码，禁止包含动作之后的未来观察。

每条认证成功分支最多生成8个训练窗口，第一步权重1，后续窗口权重0.25。

采用这样的小范围成功续跑监督，是为了避免只学会修正某一个动作，却无法保持随后的控制连续性。

所有生成数据放在独立临时目录，不能写入原 RA buffer。

## 四、COVR 的训练损失如何实现

这是核心代码改造。

### 4.1 原损失完全保留

当前 `coreset_exp/train.py` 的 `PaperMLR.observe()` 实际计算：

\\[ \mathcal L\_{\mathrm{base}} = \mathcal L\_{\mathrm{BC}} +0.1\mathcal L\_{\mathrm{IFA}} \\]

并执行原 optimizer step 和 scheduler step。

这部分不改。

### 4.2 增加恢复动作的 GMM 似然监督

对于已经认证的恢复动作 \\(a^\*\\)：

\\[ \mathcal L\_{\mathrm{recover}} = -\log \pi\_\theta(a^\*\mid z) \\]

该损失直接复用现有 GMM 的 `log_prob()`。

但仅靠整体 GMM 似然并不能保证部署时最大 logits 分量会执行正确动作，因此再增加一项执行分量导向损失。

令：

\\[ q_k=\operatorname{softmax}(\ell_k/\tau) \\]

\\[ \mathcal L\_{\mathrm{exec}} = -\log\sum\_{k=1}^{5} q_k\mathcal N(a^\*;\mu_k,\operatorname{diag}(\sigma_k^2)) \\]

这里使用较低温度的 softmax，使训练更关注实际可能被 argmax 选中的分量。

它不要求某个 GMM 分量在整个持续学习过程中保持固定身份，比直接给分量编号添加硬标签更稳妥。

最终：

\\[ \boxed{ \mathcal L = \mathcal L\_{\mathrm{BC}} + 0.1\mathcal L\_{\mathrm{IFA}} + \lambda_C \left( \mathcal L\_{\mathrm{recover}} + \beta\mathcal L\_{\mathrm{exec}} \right) } \\]

建议本次固定：

| 参数              | 预注册数值               |
| --------------- | ------------------- |
| \\(\lambda_C\\) | 0.1                 |
| \\(\beta\\)     | 0.5                 |
| \\(\tau\\)      | 0.25                |
| COVR batch size | 4                   |
| COVR 损失频率       | 每10个原始训练 batch 追加一次 |
| 采集后启用时间         | epoch50–99          |
| 无合格数据时          | COVR 损失严格为0         |

这些参数是待验证的工程假设，不能称为已经调优的最优值。

COVR 辅助 batch 与原来的 BC batch 在同一个 optimizer step 内合并反向传播，不增加 optimizer.step() 的调用次数，也不修改原 DataLoader 长度和学习率调度。

所有新梯度仅更新原来的时间解码器和 GMM Head，空间前端继续冻结。

### 4.3 核心训练伪代码

```
    normal = torch.distributions.Independent(        torch.distributions.Normal(means, scales), 1    )    log_component = normal.log_prob(target[:, None, :])    log_mix = torch.log_softmax(logits, dim=-1)    l_recover = -torch.logsumexp(        log_mix + log_component, dim=-1    )    tau = 0.25    log_exec = torch.log_softmax(logits / tau, dim=-1)    l_exec = -torch.logsumexp(        log_exec + log_component, dim=-1    )    l_covr = weighted_mean(        l_recover + 0.5 * l_exec, weights    )    loss = loss + 0.1 * l_covroptimizer.zero_grad()loss.backward()clip_gradients_as_original()optimizer.step()scheduler.step()
```

这是需要新增的实现逻辑，不是声称仓库中已经存在的代码。

特别要求：原 BC＋IFA 的 mask、任务语言 reference、损失归约方式不得因为 COVR 加入而改变。

## 五、代码修改位置

以现有已完成的 OBJECT RA 实验源码为父版本，不从原始 MLR 仓库重新搭建。

源版本：`MLR@2cd20be9a52eaa02adc27654bce79cc55dde5dfa`。科学父实现取 [OBJECT RA 已冻结源码](https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226/tree/MLR/reports/object_ra_conditional/launch/source?utm_source=chatgpt.com)。RCTR 文件只作为状态恢复工程参考。

| 文件                                          | 处理方式                               |
| ------------------------------------------- | ---------------------------------- |
| `coreset_exp/train.py`                      | 作为训练基类参考；核心原 BC＋IFA 不改             |
| `ra_exp/storage.py`                         | 保留原 RA 选样、成员和容量制度                  |
| `libero/lifelong/models/brief.py`           | 保留原时间解码器和头                         |
| `libero/lifelong/models/complete_policy.py` | 保留原生观察编码与动作接口                      |
| `libero/lifelong/models/policy_head.py`     | 复用 `forward_fn()` 返回的均值、方差和 logits |
| `coreset_exp/evaluate.py`                   | 正式评估逻辑不变                           |
| `coreset_exp/eval_metric.py`                | 正式闭环执行规则不变                         |

新增独立的 `covr_exp/`，避免破坏已验证的 RA 实验。

```
covr_exp/
├── __init__.py
├── prepare.py          # 绑定现有RA、官方数据及缓存
├── preflight.py        # 状态恢复、动作、模态准入
├── snapshot.py         # 完整环境状态保存与恢复
├── branch.py           # 六候选反事实闭环
├── certify.py          # 成功认证和样本准入
├── encode.py           # 原生观察→4-token潜特征
├── sidecar.py          # 临时恢复数据集
├── loss.py             # COVR GMM辅助损失
├── train.py            # 400epoch顺序训练
├── pipeline.py         # 阶段编排、异常与恢复
├── summarize.py        # 五指标复算
└── audit.py            # 实验合同独立审计
```

其中有三处必须实施代码级验证。

第一处：snapshot。 可以参考 RCTR A2 的 [recovery_runner.py](https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226/blob/MLR/reports/rctr_goal_seed42_stageA2/recovery_runner.py?utm_source=chatgpt.com)，但不能直接复制 GOAL 的任务、父模型路径和扰动规则。

第二处：encode。 必须从冻结的 `CompletePolicy.spatial_encode()` 获取原生特征，并验证它与完整模型原生推理产生的 GMM 参数一致。

第三处：train。 应复用 `PaperMLR`、原 optimizer/scheduler、`DistributedSampler.set_epoch()`、原 RA storage，而不是重新实现一套模仿学习框架。

## 六、实验执行顺序

### 阶段0：冻结资产和检查环境

建立独立工作目录：

```
/data/mlr_ifa/experiment_workspaces/RA_COVR_OBJECT_seed42_20261021/source
```

建立独立结果目录：

```
/data/mlr_ifa/experiments/RA_COVR_OBJECT_seed42_20261021
```

现有父实验保持只读：

```
/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed42_20261019
```

正式运行前验证：

- 父 RA seed42 四个 checkpoint、缓存、选样报告、正式成功记录完整且哈希不变。
- 原始 OBJECT 数据、BDDL、初态文件和官方预训练权重存在。
- 每个需要采集的旧任务具备预定的非评估初态。
- 当前硬件、CUDA、PyTorch、MuJoCo、robosuite及资产版本已经记录。
- 没有其他实验占用同一个输出根目录或覆盖原模型文件。

本次直接复用已冻结的 RA 特征缓存，不重新导出另一套特征。

### 阶段1：COVR 状态恢复工程准入

先用现有 RA seed42 的 Task7 checkpoint，完成仅用于工程检验的在线分叉测试，不产生正式训练样本。

固定检查：

| 检查                      | 硬门槛                      |
| ----------------------- | ------------------------ |
| 完整 native checkpoint 加载 | 所需参数完整，形状匹配              |
| GMM 原生与提取接口             | 均值、标准差、logits及动作误差 ≤1e-5 |
| 历史长度                    | 1、4、8 分别检查               |
| 恢复前物理状态                 | 最大绝对误差 ≤1e-10            |
| 同快照重复执行                 | 相同动作、物理与原生观察重复误差 ≤1e-10  |
| 观察历史                    | 恰好恢复相应历史，不混入其他分支         |
| 环境生命周期                  | 从快照保存到分支结束不销毁或硬重置对应模拟器   |

这里严格使用当前在线快照，而不是要求重现原 HDF5 演示。

所有测试均需真实执行并保存误差，不能只依靠单元测试 Mock。

工程准入失败则 BLOCKED，停止分叉与训练。 不放宽误差阈值，也不更换“更容易恢复”的状态凑通过。

### 阶段2：训练 Task6，验证零干预一致性

从官方 OBJECT 预训练 checkpoint 开始，按原 RA 协议训练 Task6 100 epoch。

此时 COVR 尚未启动。

训练完成后与原 RA 的：

```
mlrifa_6_model_ep99.pth
```

逐张量比较。

要求在相同源码、环境和输入条件下确认参数一致，并保存初始化 RNG、模型参数及优化器步数的证据。

如果 Task6 已经偏离原 RA，那么后续 Task7–9 的差异不能仅归因于 COVR，应停止并排查。

然后按现有 RA 逻辑入库781个样本。

### 阶段3：Task7–9 正式训练

每个任务执行下面的固定流程。

训练前半段

epoch 0–49

使用原始 BC＋IFA 和 RA buffer，完成50 epoch；保存完整模型、优化器、调度器、梯度、RNG及 DataLoader 进度，正常结束训练进程。

COVR 独立采集

从 epoch49 checkpoint 加载当前模型，对所有已学旧任务进行有上限的在线失败收集、分叉、认证及特征导出。不调用 optimizer。

训练后半段

epoch 50–99

从完整恢复状态继续原训练，按固定频率加入 COVR 辅助损失。无认证样本则保持原损失。

任务结束

保存 `mlrifa_{task}_model_ep99.pth`，严格加载验证；再执行原 RA 选样与 buffer 入库。进入下一任务。

训练和 COVR 采集必须使用独立进程，避免同时加载完整空间前端和 DDP 训练模型造成显存冲突。

重新启动训练进程时必须恢复原 RNG、optimizer、scheduler、model、梯度和模块 train/eval 模式，不能只加载模型权重。

### 阶段4：固定 COVR 交互预算

预先限制额外仿真，避免无限搜索成功样本。

| 项目             | 最多数量                |
| -------------- | ------------------- |
| 旧任务-当前阶段组合     | 6                   |
| 正常采集 episode   | 36                  |
| 待筛查快照          | 18                  |
| 筛查分叉 rollout   | 324                 |
| 独立认证 rollout   | 162                 |
| 分叉 rollout 总上限 | 486                 |
| 单个分叉完整任务时域     | 最多600策略步，减去已经执行的前缀步 |

最坏情况下486条分叉的理论控制步上限为291,600步；实际通常低于此值。

采集前缀、5步零动作初始化、状态恢复诊断以及异常技术尝试必须另外记账，不能写成486条完整实验。

若发现某个任务没有合格恢复样本，记录 `NO_CERTIFIED_RECOVERY`，令该任务的 COVR 损失为零，不得临时增加候选、放宽条件或更换初态。

这保证实验是有限且可审计的。

## 七、正式闭环评估

这部分直接使用原 RA 实验评估协议，不采用 RCTR/COVR 的配对噪声、动作均值分叉或其他诊断性推理规则。

当前仓库正式评估器已经实现：

- `SubprocVectorEnv`，10个并行环境。
- 每单元20个初态，索引0–19。
- 每组环境 seed 为42、43。
- 初态加载后先执行5次零动作。
- 每次取最近最多8帧构建输入。
- 动作执行使用 `CompletePolicy.get_action()`。
- 每阶段重置原全局动作 RNG。
- 最多600策略控制步。
- 正常失败不重试，技术异常单独记录。

对应代码：

[coreset_exp/evaluate.py](https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226/blob/MLR/reports/object_ra_conditional/launch/source/coreset_exp/evaluate.py?utm_source=chatgpt.com)

[coreset_exp/eval_metric.py](https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226/blob/MLR/reports/object_ra_conditional/launch/source/coreset_exp/eval_metric.py?utm_source=chatgpt.com)

### 7.1 正式评估矩阵

| 训练完成阶段 | Task6 | Task7 | Task8 | Task9 | 本阶段 episode |
| ------ | ----- | ----- | ----- | ----- | ----------- |
| Task6  | ✓     | —     | —     | —     | 20          |
| Task7  | ✓     | ✓     | —     | —     | 40          |
| Task8  | ✓     | ✓     | ✓     | —     | 60          |
| Task9  | ✓     | ✓     | ✓     | ✓     | 80          |
| 合计     |       |       |       |       | 200         |

每个单元使用20条正式记录。

必须完成全部10个单元，不能仅评估最终 Task9，也不能挑选 COVR 改善的任务做结果展示。

### 7.2 指标计算

设 \\(r\_{i,j}\\) 是完成任务 \\(i\\) 后，在任务 \\(j\\) 上20次正式测试的成功率。

主要指标：

\\[ \mathrm{AUC} = \frac14\sum\_{j=6}^{9} \frac{1}{10-j}\sum\_{i=j}^{9}r\_{i,j} \\]

其他指标：

\\[ \mathrm{FWT} =\frac14\sum\_{j=6}^{9}r\_{j,j} \\]

\\[ \mathrm{NBT} =\frac13\sum\_{j=6}^{8} \left[ r\_{j,j} -\frac{1}{9-j}\sum\_{i=j+1}^{9}r\_{i,j} \right] \\]

\\[ \mathrm{OldFinal} =\frac{r\_{9,6}+r\_{9,7}+r\_{9,8}}3 \\]

\\[ \mathrm{FinalAll} =\frac14\sum\_{j=6}^{9}r\_{9,j} \\]

所有指标先由整数成功次数和20次分母精确计算，再转成百分比。

此外应报告相对原 RA 的逐单元成功次数差异，避免总 AUC 掩盖单任务明显退化。

## 八、实施命令与运行编排

以下是新增 `covr_exp` 模块完成后应实现的命令接口，不是声称现有仓库已经包含这些命令。

### 8.1 环境配置

```
source /data/mlr_ifa/activate.sh

export CORESET_SEED=42

export RA_BASE=/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed42_20261019

export CORESET_SOURCE=/data/mlr_ifa/experiment_workspaces/RA_COVR_OBJECT_seed42_20261021/source

export CORESET_OUT=/data/mlr_ifa/experiments/RA_COVR_OBJECT_seed42_20261021

export PYTHONPATH="$CORESET_SOURCE"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$CORESET_SOURCE"
```

这里 `RA_BASE` 表示已完成的 RA 父实验。若复用旧 `ra_exp.prepare.py`，其原有 `RA_PARENT` 实际指向普通 Coreset 父实验，不能直接混用这两个含义。

因此建议新增 `covr_exp.prepare` 来验证并绑定两级父实验。

### 8.2 运行入口

```
# 1. 新实验资产准备
python -m covr_exp.prepare

# 2. 软件和在线状态恢复准入
python -m covr_exp.preflight

# 3. 单seed顺序训练及中途COVR采集
python -m covr_exp.pipeline train

# 4. 原评估器：仍将内部训练组命名为ra_coreset
python -m coreset_exp.evaluate ra_coreset

# 5. 五指标汇总与独立审计
python -m covr_exp.summarize
python -m covr_exp.audit
```

为直接复用原评估器，新实验内部可以继续使用：

```
$CORESET_OUT/ra_coreset/models/
$CORESET_OUT/ra_coreset/evaluation/
```

方法身份在新实验合同中明确标注为 `RA_Coreset+COVR`，不能因为内部目录仍叫 `ra_coreset` 就在最终报告中误记为纯 RA。

训练调度器须在内部通过原来的 `torch.distributed.run --standalone --nproc_per_node=1` 调起单卡 DDP 训练，保持既有优化器初始化方式。

### 8.3 断点恢复和异常处理

每次 epoch 完成后持久化：

```
model
optimizer
scheduler
gradient_state
module_modes
rng_python
rng_numpy
rng_torch_cpu
rng_torch_cuda
task
next_epoch
optimizer_step
buffer_hashes
covr_manifest_sha256
```

COVR 采集应采用先写独立原始证据、完成认证、再原子提交 sidecar manifest 的方式。

若进程异常：

- 尚未提交的分支记录为技术尝试，不记成失败轨迹。
- 已提交的认证样本不得重新随机生成。
- 恢复前重新校验模型、缓存、manifest、噪声表及状态。
- 只有具备完整随机状态的情况下才能继续；否则标为 BLOCKED。
- 原正式评估禁止盲目只重跑失败单元。

## 九、最终验收与实验报告

### 9.1 必须满足的验收条件

| 检查内容           | 完成标准                             |
| -------------- | -------------------------------- |
| 单 seed         | 只包含42                            |
| 训练阶段           | Task6–9完整                        |
| Epoch          | 每阶段100，总400                      |
| Optimizer step | 105,500，除非实际合法缓存规模与冻结预期不符，此时停止审计 |
| RA 记忆          | 与原 RA 有序成员、SHA和配额一致              |
| Task6          | 无 COVR 干预，与父 RA 逐张量一致            |
| COVR 数据        | 全部具有状态恢复及独立成功认证记录                |
| 长期 memory      | 不增加、不替换原 RA 成员                   |
| 模型             | 四个 ep99 严格加载                     |
| 正式闭环           | 200条，10个单元各20条                   |
| 结果             | 五指标及完整矩阵可独立复算                    |
| 成本             | 训练时间、额外交互、分支次数、显存及临时存储完整记录       |
| 源码             | 冻结源码和输入 SHA256，父实验未被修改           |

### 9.2 结果决策

若 AUC 严格高于71.5625%

判定为 OBJECT seed42 探索性正向结果。结合旧任务最终成功率和逐任务退化情况评价收益，保留后续 seed 扩展的依据，但本次不自动扩展。

若 AUC 没有提高

如实报告负结果，并根据认证样本数量、恢复动作的训练损失、GMM 最大分量变化及每个闭环单元判断失败发生的位置。不在同一 seed42 上反复调参直到成功。

若恢复工程或训练合同未通过

标记 BLOCKED，不得把不完整结果作为算法有效或无效的证据。

### 9.3 必须交付的产物

```
RA_COVR_OBJECT_seed42_20261021/
├── PROTOCOL.md
├── CONTRACT.json
├── INPUT_MANIFEST.json
├── SOURCE_SHA256.json
├── RESTORATION_GATE.json
├── COVR_INTERACTION_LEDGER.jsonl
├── COVR_CERTIFICATION.json
├── COVR_SIDECAR_MANIFEST.json
├── TRAINING_COMPLETE.json
├── EVALUATION_COMPLETE.json
├── METRICS.json
├── PAIRED_COMPARISON.json
├── FINAL_ACCEPTANCE.json
├── REPORT.md
└── ra_coreset/
    ├── models/
    │   ├── mlrifa_6_model_ep99.pth
    │   ├── mlrifa_7_model_ep99.pth
    │   ├── mlrifa_8_model_ep99.pth
    │   └── mlrifa_9_model_ep99.pth
    ├── buffer/
    ├── covr/
    └── evaluation/
```

实际权重、潜特征、原始动作、仿真快照及噪声表保留在服务器私有目录。远端 GitHub 仅发布审核后的代码、协议、统计、哈希和轻量报告。

## 十、对论文创新性与公平性的最后约束

本次设计有一个不能回避的限制：

只要 COVR 在学习 Task7–9 时重新访问旧任务环境，并从中获得新的成功监督，它就已经突破了原论文仅使用历史缓存进行持续学习的训练信息约束。

因此实验应被准确命名为：

RA-Coreset with Outcome-Verified Interactive Recovery

它与原 MLR 保持模型、演示初始化、基础优化步数、长期 memory 配额及正式评估协议一致，但额外获得了仿真交互。

所以，如果 seed42 正向，下一轮正式验证不仅应扩展种子，还应增加一个相同交互预算、但不使用反事实恢复认证的回放对照，以区分算法贡献和额外环境数据带来的收益。本轮只运行要求的 RA+COVR 单组，暂不增加该组训练开销。

## 最终执行决策

本次最合理的实施顺序是：OBJECT 在线恢复工程准入 → RA+COVR 四阶段训练（epoch49 插入认证恢复）→ 原始200条闭环评估 → 相对现有 RA seed42 的配对指标审计。

其中最需要优先投入精力的不是新网络结构，而是三件事：

1. 保证在线恢复后，物理状态、原生观察和 GMM 推理完全一致。
2. 确保 COVR 监督真正改变部署时会执行的动作分量，而非只提高混合分布似然。
3. 确保原 RA 选样、基础训练步数及正式评估全部不变，同时把额外环境交互单独记账。

这样得到的 seed42 结果，才能用来判断 COVR 是否值得进一步开展多 seed 闭环实验。