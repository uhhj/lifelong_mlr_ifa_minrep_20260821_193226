# MLR＋聚合回放梯度约束：可安装实验包

本实验回答：在已完成的 MLR 复现协议下，一个简单的聚合回放梯度约束能改善多少保持—学习折中，额外成本是多少？

方法名：`MLRReplayConstraint`，报告中写作 **MLR + aggregate replay-gradient constraint (A-GEM-style)**。它不是原版 A-GEM 的严格复现。

## 1. 已对齐的基线

依据你提供的 Codex 任务 `01a02326-7ade-7c10-a1c6-a9811bae5b77`，以及 `own_reference_control/PROTOCOL.md` 和正式报告：

| 项目 | 主协议 |
|---|---|
| 基准 | LIBERO-OBJECT，task6–9 |
| 初始化 | 同一官方 OBJECT pretrained checkpoint |
| 数据 | 已有 `features/object_seed42/43/44`；不重新提取或跨 seed 复制 |
| 训练 | 每任务100 epochs，batch10，单GPU |
| 优化器 | AdamW，lr=1e-4，betas=(0.9,0.999)，eps=1e-8，weight_decay=0.1 |
| 调度 | 原有 batch-step linear schedule，warmup500 |
| 回放 | 原数据拼接、shuffle、buffer更新规则，past_data_rate=0.5 |
| 主评估 | 固定ep99；每格20trials；原始整矩阵调用顺序和GMM动作采样 |
| 对照 | 原MLR；已修正own-reference的IFA作为补充比较 |

服务器仓库在最近报告中为 `/media/dm/Elements/wzp2026/lifelong_mlr_ifa_minrep`。本包不登录服务器、不启动正式训练、不覆盖历史结果。目标是将代码接入该仓库当前已验证的版本。

早期 `legacy_epoch` 调度、最好epoch选择，以及逐任务独立调用的配对rollout诊断结果，不能与本主实验直接混合比较。当前历史MLR参考是 FWT 82.500、NBT 3.889、AUC 80.104（三seed均值）；它们是历史结果，不是本实验结果。

## 2. 实际计算规则

原代码使用混合batch的平均GMM负对数似然。本实现只调用一次：

```python
per_sample, _ = policy.compute_loss(data, None, None, reduction="none")
loss_mlr = per_sample.mean()
loss_ref = per_sample[task_ids < current_task].mean()
```

因此，若新旧样本数为Nnew/Nold：

```text
L_MLR = (Nnew/B)*Lnew + (Nold/B)*Lold
g_MLR = ∇(loss_scale * L_MLR)
g_ref = ∇(loss_scale * Lold)
```

**没有另外引入lambda_r，没有把两个均值直接相加，没有重新平衡任务，没有额外抽样参考批次。** 原有每批新旧占比保持不变。参考梯度是当前混合批次中旧样本的均值，不是所有旧任务的精确平均。

当`dot(g_MLR,g_ref)<0`且参考范数足够大时：

```text
g_final = g_MLR - dot(g_MLR,g_ref)/dot(g_ref,g_ref) * g_ref
```

否则`g_final=g_MLR`。随后执行原有全局梯度裁剪和一次AdamW更新；不再追加回放梯度。

投影向量是解码器与策略头的所有可训练参数拼接而成的整体；不是逐层或逐块投影。实现无需构造完整拼接向量，内积在各参数张量上求和。

特殊情况：

- 无旧样本：直接原MLR更新，不额外抽旧样本，不额外消耗随机数。
- 只有旧样本：参考与原梯度一致，通常不触发投影。
- 参考范数≤1e-12：跳过投影并记录。该值是范数阈值，不是给分母加epsilon。
- 所有样本使用同一次前向中的dropout实现，不进行第二次随机前向。
- 原loss_scale与裁剪顺序保留。暂只支持单进程/单GPU、FP32或FP64，不支持AMP或多卡局部投影。
- task0–5没有在当前MLR缓冲中出现，因此本约束没有对这些预训练任务额外提供参考梯度。

原始梯度约束只提供局部一阶方向解释。AdamW的动量、缩放和权重衰减、批次估计误差及非线性，都可能使实际旧损失上升；本实验不添加二阶、风险、子空间或缩步模块。

## 3. 文件布局与接入范围

```text
replay_constraint_experiment/
  install.py                     安装/预览；检查目标基线接口
  run.py                         smoke、正式训练和固定ep99评估
  summarize.py                   严格矩阵检查、三seed汇总与配对差值
  parity_real.py                 真实策略/缓存的native-vs-shadow一致性检查
  src/
    replay_constraint.py         梯度约束核心，只依赖torch
    mlr_replay_constraint.py      继承现有MLR，仅替换observe
    mlr_replay_constraint.yaml    算法配置
  tests/test_experiment.py        数学、GMM、更新、日志与协议测试
```

安装只新增上述3个`src`文件到目标`libero/`，并在算法注册文件追加一行import。原`mlr.py`、`base.py`、模型、数据集、调度器、测试器不改。要求目标是完整仓库；本机某些`*_work`目录只有差异文件，不能独立作为安装目标。

## 4. 安装与本地逻辑测试

将本文件夹放入服务器仓库根目录，激活原来的Python3.8/PyTorch1.13环境：

```bash
cd /media/dm/Elements/wzp2026/lifelong_mlr_ifa_minrep
export REPO="$PWD"
export PKG="$REPO/replay_constraint_experiment"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

python "$PKG/install.py" --repo "$REPO" --dry-run
python "$PKG/install.py" --repo "$REPO"
python -m unittest discover -s "$PKG/tests" -v
```

重复安装同一版本是幂等的。若目标已有不同内容的实验文件，安装器停止，不覆盖。无需安装新训练依赖，也不要升级已复现环境。

## 5. 先运行真实数据一致性检查

```bash
source scripts/_common.sh

python "$PKG/parity_real.py" \
  --repo "$REPO" \
  --baseline-config "$REPO/runs/final_paper_text_faithful/mlr/seed42/run_001/config.json" \
  --feature-root "$REPO/features/object_seed42" \
  --checkpoint "$REPO/checkpoints/object_multitask_model_ep99.pth" \
  --output "$REPO/reports/replay_constraint/real_parity_seed42.json"
```

此检查直接使用目标仓库的原MLR和新增子类、真实GMM策略及缓存，依次验证纯新、混合、纯旧batch。投影关闭时，损失、梯度、更新后的参数、AdamW状态、调度器状态和随机状态要求逐项精确一致。

检查使用临时模型，不写历史checkpoint；临时输出自动清理。若失败，先定位差异，不降低容差后直接开始正式实验。该检查没有DDP包装；下一步torchrun smoke用于覆盖真实入口。

## 6. 冒烟实验

```bash
python "$PKG/run.py" train --repo "$REPO" --arm constraint --seed 42 --smoke
python "$PKG/run.py" train --repo "$REPO" --arm shadow --seed 42 --smoke
```

smoke独立目录：task6和task7各最多4步、1epoch；task6更新缓冲后task7能够遇到旧样本。检查`replay_constraint_epochs.csv`中task7的`reference_batches`；若为零，该短smoke尚未覆盖参考路径，需要另做足够长的检查，不能声称已经验证。

smoke只检查训练入口、参考梯度、checkpoint与日志。它不产生正式成功率；正式评估拒绝smoke输出。

## 7. 正式训练与评估

默认只新跑`constraint`，使用已有正式MLR和修正IFA作对照：

```bash
for SEED in 42 43 44; do
  python "$PKG/run.py" train --repo "$REPO" --arm constraint --seed "$SEED"
  python "$PKG/run.py" eval  --repo "$REPO" --arm constraint --seed "$SEED"
done
```

上述命令逐seed串行运行，任一步失败请停止并检查。若在普通交互shell执行循环，建议先`set -e`，或逐条执行。

可选运行：

```bash
# shadow计算参考梯度和诊断，但不投影，用来验证轨迹与测量诊断开销。
python "$PKG/run.py" train --repo "$REPO" --arm shadow --seed 42

# 只有需要同硬件/同时间窗口的成本对照时，才另跑native MLR。
python "$PKG/run.py" train --repo "$REPO" --arm mlr --seed 42
```

`--arm ifa_corrected`也受支持，要求仓库已有own-reference修复；通常直接使用现有修正IFA结果即可。

训练输出：

```text
runs/replay_constraint/aggregate_v1/formal/constraint/seed42/
  experiment.json
  train_stdout.log
  train_execution.json
  TRAINING_COMPLETE
  run_001/
    config.json
    mlrifa_6_model_ep99.pth ... mlrifa_9_model_ep99.pth
    lr_protocol.csv
    replay_constraint_parameters.json
    replay_constraint_epochs.csv
  primary_ep99.log
  metrics.json
  PRIMARY_EVAL_COMPLETE
```

缓冲与结果按arm、seed、smoke/formal独立存储。启动器拒绝非空目标，不删除、不自动续跑失败训练。改`--tag`可建立新独立运行。

`--feature-root`、`--checkpoint`、`--data-root`可指定已有路径；`--gpu`选择单卡。`--dry-run`只打印命令，不创建运行目录。主协议默认batch_step；如使用其他调度，必须另设tag并重新建立同协议对照。

评估使用原`test.py`一次完整调用，固定ep99，不调用最佳epoch选择，不拆成单任务诊断调用。每seed应有10个唯一矩阵单元。日志解析不能独立证明确实完成20trials，需结合启动参数与原评估器；它会检查成功率是否与20次二元试验相容。

## 8. 汇总并复用现有对照

```bash
REF_ARGS=()
for SEED in 42 43 44; do
  REF_ARGS+=(--reference "mlr:$SEED:$REPO/runs/final_paper_text_faithful/mlr/seed$SEED/primary_ep99.log")
  REF_ARGS+=(--reference "ifa_corrected:$SEED:$REPO/runs/own_reference_control/ifa/seed$SEED/primary_ep99.log")
done

python "$PKG/summarize.py" \
  --run-root "$REPO/runs/replay_constraint/aggregate_v1/formal" \
  "${REF_ARGS[@]}" \
  --output "$REPO/reports/replay_constraint/aggregate_v1_results.json"
```

汇总保留原项目FWT/NBT/AUC定义。其中FWT是阶段对角成绩，不将其改称独立测得的零样本迁移；NBT越低表示该指标下遗忘越少。另报告最终平均成功率、最差旧任务成功率、最大峰值到最终遗忘、逐seed配对差值和样本标准差。单seed没有标准差时写null。

缺失、重复、非ep99矩阵会报错；不将缺失项填零，也不挑选有利epoch。历史日志接入仍需人为确认配方、缓存、初始化、评估器和调用顺序匹配；仅相同seed不能证明这些条件相同。

## 9. 诊断日志怎么解释

每个epoch的CSV累计所有batch，包含：

- `projection_rate_all`：投影次数/全部batch数。
- `projection_rate_valid_reference`：投影次数/有足够参考范数的batch数。
- `no_reference_batches`、`tiny_reference_batches`：不能检测或数值跳过的覆盖盲区。
- `mean_correction_ratio`：修正范数/原混合梯度范数，分母不含AdamW更新。
- 新/旧样本数及分开BC损失，混合BC损失，内积、余弦和梯度范数。
- `observe_seconds`：默认同步GPU后测量的observe时间，不含数据加载/评估；开启同步会有计时开销。
- `stage_peak_allocated_bytes`：阶段内PyTorch峰值allocated显存，不等于驱动占用或reserved显存。
- 启动器`train_execution.json`另存端到端训练墙钟时间，含启动及checkpoint写盘。与native成本对照须同GPU、相同负载和协议。

每批额外最多一次参考梯度遍历，持久状态不含每任务子空间，参考梯度只在该步临时保留。旧任务梯度互相抵消、参考近零、AdamW方向改变仍可能存在；实验结果只能支持这条简单规则是否有效，不能单独否定全部梯度控制方法。

## 10. 交付验证边界

本机CPU验证记录见`VALIDATION.md`。服务器真实模型一致性、torchrun smoke和正式GPU训练/rollout需要在既有环境运行；本次没有声称这些实验已经完成。

研究顺序：一致性检查 → smoke → seed42完整运行 → 核对协议和日志 → seed43/44 → 配对汇总。是否值得做复杂控制器，要根据成功率、遗忘、新任务拟合速度和额外成本一起判断。
