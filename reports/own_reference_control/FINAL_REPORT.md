# Own-reference 受控对账：最终报告

## Verdict

**OWN_REFERENCE_RECONCILIATION_COMPLETE_WITH_DOCUMENTED_DELTA**

取样错误及其梯度影响已确认；仅修正该取样的三 seed 正式对照已完成，原模型与结果保留。错误能解释当前协议下旧 IFA 的大部分平均 AUC 劣势，但不是所有低分的唯一原因，也不意味着修正必然改善每个 seed。**本报告不是“完整精确复现论文成功”，也不是对论文方法有效性的否定。**

主因分类：**mixed：已证实的实现偏差 + seed 相关的后续训练影响 + 尚未恢复的论文协议细节。** 环境不是本次对照中的变化因素；没有证据把剩余差异确定归因于环境。

## Scope / provenance

- Starting SHA: `a4ed54d8502a10dd41a1a5ec83f9234e97a0071a`。
- Branch: `codex/ifa-own-reference-reconciliation`。
- Repository: `uhhj/lifelong_mlr_ifa_minrep_20260821_193226`。
- Server: `dm-MS-7D70` / `wzp2026`；训练时地址 `172.21.105.127`，最终交付时通过 `172.17.27.219` 连接并验证为同一已信任主机；目录 `/media/dm/Elements/wzp2026/lifelong_mlr_ifa_minrep`。
- 单卡 GPU0，RTX4090；driver580.173.02；Python3.8.20 / PyTorch1.13.0+cu116 / CUDA runtime11.6 / MuJoCo3.2.3 / robosuite1.4.0 / transformers4.26.1。
- LIBERO-OBJECT tasks6–9；seed42/43/44；相同官方 `checkpoints/object_multitask_model_ep99.pth`，相同 `data/libero_object` 和各 seed 的冻结 features。
- 全部100 epochs/stage，batch10，AdamW初始lr1e-4，weight_decay0.1，batch-step linear scheduler / warmup500，gamma0.30，lambda0.1，past_data_rate0.5。
- 新鲜独立 `buffers/own_reference_control/ifa/seed*/`；输出 `runs/own_reference_control/ifa/seed*/`。
- 主评估固定ep99、20 trials；原GMM采样动作、任务调用顺序和初始状态文件不变。没有最好epoch选择，没有调参或新方法。
- 旧 MLR、旧取样 IFA 取自既有正式运行，不重跑。这两个对照的训练、数据、任务配置已核对一致。这里的 `release_ifa` 标签指**同一冻结协议下沿用 release 取样实现的旧运行**，不是未修改的整套官方 release 配方。
- 正式队列于2026-09-06 00:11:29完成，自动汇总于00:11:30完成；人工完整性核验及最终归因于9月6日完成。最终提交SHA在Git交付记录中给出，避免文档自引用其所属提交。

## 已确认的问题与修正边界

旧代码先用当前任务mask取得 global latent，却把整个混合batch第一条样本的语言向量作为own-reference。它可能属于旧任务；若恰好属于选中的负参考任务，正负角距离项抵消，留下与global latent无关的常数margin。

修正只增加显式 `ifa_own_reference_fix=true`：用 `cur_feat[batch_mask,0,0,:]` 对齐当前任务的各个global latent。默认false保留旧行为。负参考、pairs、角距离、margin、loss权重、优化器和scheduler均不变；另更正两处维度注释。

这符合[论文 Eq4 / own-task reference 的要求](https://arxiv.org/html/2603.10929v2#S3.SS2)，但只核实这一机制，不据此宣称其余未公开的论文训练细节完全一致。

## 完成证据

- 三 seed 共1200个epoch全部完成且训练记录finite；每个seed28个预期checkpoint。
- 全部1200行实际batch数、优化器步数及LR轨迹与旧取样IFA完全相同，只有实验标签不同。
- 每seed实际buffer文件选择及IFA pairs与旧运行一致：42为7→6、8→7；43/44仅7→6；全部task9无pair。
- 三seed task6新旧权重逐张量相同；对应首次评估成功率也分别精确复现35%、65%、60%。
- 9个正式对照矩阵（3 arms ×3 seeds）各有10个唯一、完整的预期单元，共90项；无缺失、重复或选报。新实验共600次正式rollout。
- 24个历史ep99 checkpoint均包含完整且shape匹配的decoder/head参数。CompletePolicy先严格加载完整官方预训练权重，再加载lifelong部分权重；没有发现decoder/head漏加载。
- 17,763个缓存文件中，每个任务所有时间位置的语言参考完全一致，跨seed也相同。因此按当前样本取own-reference等价于该任务固定语言参考，不引入新定义。
- 54个固定checkpoint梯度探针：24个负任务首样本batch旧IFA对g梯度为0、修正后非0；6个其他旧任务首样本batch梯度余弦0.338–0.357；24个参考正确的对照batch一致。这是分层诊断样本，不是训练全程的无偏均值估计。
- 最终19项协议/实际observe逻辑回归测试通过。测试不替代上述正式运行与矩阵证据。

详见 `COMPLETION_AUDIT.json`、`checkpoint_loading_audit.json`、`REFERENCE_SAMPLING_AUDIT.md`、`REFERENCE_GRADIENT_AUDIT.md`、`REFERENCE_STABILITY.md`。

## 正式结果

单位为百分比/百分点；std为sample standard deviation。NBT越低表示该指标下遗忘越少，但不能脱离初次学习成功率单独解释。

| 实现 | seed42 FWT / NBT / AUC | seed43 | seed44 |
| --- | --- | --- | --- |
| MLR | 75.000 / 7.222 / 71.146 | 86.250 / 6.667 / 82.604 | 86.250 / -2.222 / 86.562 |
| 旧取样IFA | 77.500 / -0.278 / 77.708 | 70.000 / 12.778 / 63.542 | 82.500 / 3.056 / 80.833 |
| 修正IFA | 75.000 / 3.889 / 72.917 | 81.250 / 3.611 / 78.750 | 86.250 / -1.667 / 86.458 |

| 实现 | FWT mean±std | NBT mean±std | AUC mean±std |
| --- | ---: | ---: | ---: |
| MLR | 82.500±6.495 | 3.889±5.300 | 80.104±8.006 |
| 旧取样IFA | 76.667±6.292 | 5.185±6.783 | 74.028±9.214 |
| 修正IFA | 80.833±5.637 | 1.944±3.131 | 79.375±6.792 |
| 论文MLR参考 | 83.3±2.6 | 12.3±2.4 | 77.6±3.0 |
| 论文IFA参考 | 84.6±1.9 | 11.4±5.6 | 79.4±1.5 |

| 修正IFA−旧取样IFA | ΔFWT | ΔNBT | ΔAUC |
| --- | ---: | ---: | ---: |
| seed42 | -2.500 | +4.167 | -4.791 |
| seed43 | +11.250 | -9.167 | +15.208 |
| seed44 | +3.750 | -4.723 | +5.625 |
| 三seed均值 | +4.167 | -3.241 | +5.347 |

修正IFA−MLR均值：ΔFWT=-1.667、ΔNBT=-1.945、ΔAUC=-0.729。**论文报告的IFA平均AUC优势没有在这里复现。**

| 修正IFA−论文IFA | signed gap | standardized descriptive gap |
| --- | ---: | ---: |
| FWT | -3.767 | -1.982 |
| NBT | -9.456 | -1.689 |
| AUC | -0.025 | -0.017 |

最后一列只是除以论文报告std的描述性量，不是z-test或显著性。AUC均值碰巧接近论文不证明完整复现；本地AUC std6.792远大于论文报告1.5，FWT/NBT及相对MLR趋势仍不同。

## 差距来源

### 1. 不是单纯seed42异常，也不是单一系统错误解释全部

seed42在MLR及修正IFA中均为三个seed最低者，其task6首学成功率35%也低于另外两个seed的65%/60%。task6不启用IFA且新旧模型相同，因此这部分共享低分不能归因于own-reference错误。

旧取样IFA最低的却是seed43；修正使其AUC增加15.208pp，而seed42下降4.791pp、seed44上升5.625pp。因此取样错误有真实、明显且依seed而异的后续训练影响，不能把所有差异都称作随机波动，也不能保证修正逐seed改善。

### 2. 平均AUC恢复主要来自task9

| 任务 | 旧取样→修正：平均首学成功率 | 旧取样→修正：任务列AUC |
| --- | --- | --- |
| 6 | 53.333→53.333 | 46.667→45.833 |
| 7 | 100.000→93.333 | 97.778→91.667 |
| 8 | 95.000→90.000 | 93.333→93.333 |
| 9 | 58.333→86.667 | 58.333→86.667 |

task9的改善给总AUC贡献+7.083pp，task6/7列分别抵消约0.208/1.528pp，task8净贡献0，得到总改善+5.347pp。task9逐seed从75/25/75变为80/80/100。

所有task9都没有直接IFA pair；43/44更只在task7存在pair。其后期差异来自此前修正造成的模型/优化轨迹差异继续传播，而不是为task9强行创建了pair。这里确认的是受控改动的后续结果，不是已定位每个梯度更新的详细因果路径。

按恒等式 AUC=FWT−平均列内retention penalty，平均改善分解为首学项+4.167pp和penalty下降带来的+1.181pp。**这是指标代数分解，不是独立识别“学习”和“遗忘”的因果机制。**

### 3. 剩余本地MLR差距主要在task7

MLR与修正IFA的task6/8/9平均首学成功率相同；task7为100%对93.333%，解释全部平均FWT差距1.667pp。

任务列AUC（修正IFA−MLR）为task6:+1.250、task7:-5.000、task8:+0.833、task9:0，平均后为-0.729pp。task6并不是修正后平均劣势的主要来源。

42的8→7 pair之后，stage8上的task7新旧都为95%，但stage9变为65%对旧100%；43/44没有该pair，却有不同的后期效果。不能用不同seed之间的pair差异作隔离因果比较：feature cache和训练seed也不同。本任务没有修改pair selection来追指标。

## 可以与不可以作为最终结论的内容

**可以：** 在此固定初始化、缓存和训练/评估协议下，own-reference实现错误真实存在；修正使平均FWT/AUC提高4.167/5.347pp，明显减小旧IFA相对MLR的AUC劣势（6.076pp缩至0.729pp），但效果异质，且未复现平均IFA AUC优势。修正实验可作为明确标注的本地基线。

**不可以：** 因AUC79.375接近79.4就宣布完整论文精确复现；宣称IFA普遍优于/劣于MLR；宣称所有剩余差异都由seed或环境解释；把较低NBT当作无条件更好学习。

剩余不确定性：论文具体seed及每seed预训练权重/脚本未恢复；scheduler步进单位、warmup、feature augmentation、checkpoint规则等paper-unspecified选择保持既定fallback；20-trial随机GMM评估和n=3不足以证明普遍优势，且没有论文逐任务矩阵可用于一对一来源定位。本轮未切换环境、增加seed、搜索参数或重做预训练。

## 成功执行的命令

在仓库根目录：

```bash
PYTHON=/home/dm/miniconda3/envs/wzp_mlr_tmp/bin/python \
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
bash scripts/run_own_reference_control.sh

PYTHON=/home/dm/miniconda3/envs/wzp_mlr_tmp/bin/python \
bash scripts/summarize_own_reference_after_queue.sh

/home/dm/miniconda3/envs/wzp_mlr_tmp/bin/python scripts/audit_control_completion.py
/home/dm/miniconda3/envs/wzp_mlr_tmp/bin/python scripts/audit_checkpoint_loading.py
/home/dm/miniconda3/envs/wzp_mlr_tmp/bin/python scripts/audit_reference_stability.py
```

正式启动使用nohup并把stdout/stderr保存到 `runs/own_reference_control_driver/`。完整启动参数在launcher和各run/config.json。测试在已设置LIBERO_CONFIG_PATH、PYTHONPATH、离线HF缓存及EGL的环境下执行：

```bash
/home/dm/miniconda3/envs/wzp_mlr_tmp/bin/python -m pytest -q \
  tests/test_own_reference_control.py tests/test_final_paper_text_faithful.py
```

## 资产与下一步

完整数据：`CONTROLLED_RESULTS.json`、`CONTROLLED_RESULTS.md`、`success_matrices.csv`、`per_task_diagnostics.csv`，以及本目录各审计文件。历史报告原样归档在 `reports/final_paper_text_faithful/`，其旧“paper-text-faithful / NOT_SUPPORTED”解释已由 `HISTORICAL_RESULTS_NOTICE.md` 和本报告限定；历史数值没有被修改。

算法文件仅修改 `libero/lifelong/algos/base.py`；新增的launcher、汇总、只读audit和测试由本分支Git diff精确列出。未提交模型、缓存或buffer。

**Single next action：冻结并使用这个明确开启 `ifa_own_reference_fix=true` 的本地基线开展后续对照；所有新方法首先与同协议MLR/修正IFA比较，不以命中论文表格为目标。** 原launcher默认仍保留旧取样行为，不能把默认运行误称为已修正。本轮不启动新方法开发。
