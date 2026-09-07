# MLR + aggregate replay-gradient constraint：正式结果

## 结论

**执行：COMPLETE。方法收益：NOT_SUPPORTED（本协议、三seed）。**

三seed正式训练和评估全部完成。相对冻结MLR对照，FWT下降4.583、AUC下降4.028
个百分点；NBT下降0.741，但最终平均成功率也下降4.583个百分点。因此本轮结果
不支持该聚合回放梯度约束改善整体保持—学习折中。这个结论仅针对本实现和协议，
不是原版A-GEM的严格复现结论，也不能否定全部梯度控制方法。

## 来源与协议

- 冻结基线：`d75fb371cf575beb079ce064d70d2ae323da95dd`。
- 实际训练代码：`b9f448b63b726254b5a4f15372036d319dc17f5e`。
- 分支：`codex/aggregate-replay-constraint`；报告所属最终提交由Git记录标识，避免自引用。
- 仓库：`uhhj/lifelong_mlr_ifa_minrep_20260821_193226`。
- 服务器：dm-MS-7D70，`/media/dm/Elements/wzp2026/lifelong_mlr_ifa_minrep`。
- GPU0 RTX4090，driver580.173.02；Python3.8.20、PyTorch1.13.0+cu116、CUDA runtime11.6、
  MuJoCo3.2.3、robosuite1.4.0、transformers4.26.1；未改变既有环境。
- LIBERO-OBJECT task6–9，seed42/43/44；同一官方OBJECT checkpoint、同一数据和各seed
  既有features；每任务100epochs、batch10、AdamW lr1e-4、betas=(0.9,0.999)、
  eps1e-8、weight_decay0.1、batch-step linear schedule/warmup500、past_data_rate0.5。
- 本方法关闭IFA；只在原混合batch中取旧样本平均BC梯度为参考，当与完整混合BC梯度
  内积为负时进行全参数联合投影，再执行原裁剪和AdamW。无额外抽样、无额外回放权重。
- 主评估固定ep99，原test.py完整矩阵调用顺序、20trials、原GMM动作采样与success条件；
  每seed10个矩阵单元。新实验共600次正式rollout，不混入smoke或逐task诊断评估。
- MLR和修正own-reference IFA采用已有同协议正式结果，不重跑。并非声称本地协议已
  完全恢复论文未公开配方；论文数字不是本轮优化目标。
- 队列2026-09-06 15:55:20启动，19:18:40结束，约3小时23分20秒；所有seed顺序执行，
  没有按成绩筛选后续seed、调参、选择最好epoch或重跑。

## 完成核验

真实策略的native-vs-shadow检查在纯新、混合、纯旧batch上，loss、梯度、更新参数、
AdamW状态、scheduler状态和随机状态精确一致。基础测试15通过、1可选快照测试跳过；
constraint/shadow单卡torchrun smoke均通过，task7各有3个参考batch，短smoke未触发投影。

正式运行三seed各400个epoch、105500个batch；训练/评估退出码为0，完成标记齐全，
finite检查通过。三seed完整train配置与MLR相同；1200行实际LR/步数日志除arm标签外
逐项相同。9个方法×seed矩阵共90单元均从原日志重新解析并与汇总核对。
这些检查支持执行完整性，不是对每条浮点计算或因果机制的穷尽证明。

## 三seed主结果

单位：百分比；均值 ± 样本标准差，n=3。FWT/AUC越高越好，NBT越低越好。

| 方法 | FWT | NBT | AUC | 最终平均成功率 |
| --- | ---: | ---: | ---: | ---: |
| MLR | 82.500 ± 6.495 | 3.889 ± 5.300 | 80.104 ± 8.007 | 78.333 ± 13.826 |
| 修正IFA | 80.833 ± 5.637 | 1.944 ± 3.130 | 79.375 ± 6.792 | 77.083 ± 11.880 |
| 聚合回放约束 | 77.917 ± 6.415 | 3.148 ± 2.891 | 76.076 ± 6.883 | 73.750 ± 10.680 |

| seed | 约束FWT | 约束NBT | 约束AUC | ΔFWT vs MLR | ΔNBT | ΔAUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 72.500 | 2.222 | 71.250 | -2.500 | -5.000 | +0.104 |
| 43 | 76.250 | 6.389 | 73.021 | -10.000 | -0.278 | -9.583 |
| 44 | 85.000 | 0.833 | 83.958 | -1.250 | +3.056 | -2.604 |
| 平均配对差 | — | — | — | -4.583 | -0.741 | -4.028 |

配对差样本标准差：FWT4.732、NBT4.048、AUC4.998。不做n=3显著性检验。
全部逐阶段/逐任务结果保存在`success_matrices.csv`及汇总JSON的`r`字段，未隐藏反方向结果。

## 指标定义与逐任务定位

沿用本项目原定义，r为百分比，任务下标0–3对应task6–9：

```text
FWT = mean_i r[i,i]
NBT = mean_{i=0..2} mean_{q=i+1..3} (r[i,i] - r[q,i])
AUC = mean_{i=0..3} mean_{q=i..3} r[q,i]
```

FWT在此表示首学阶段对角成绩，不是独立测量的零样本迁移；NBT可为负。

| task | 平均Δ首学成功率 | 平均Δ该task的阶段平均成功率 | 平均Δ最终成功率 |
| --- | ---: | ---: | ---: |
| 6 | 0.000 | +1.667 | -3.333 |
| 7 | -3.333 | -1.111 | +3.333 |
| 8 | 0.000 | -1.667 | -3.333 |
| 9 | -15.000 | -15.000 | -15.000 |

task9是主要损失来源：三seed约束/MLR分别为70/80、50/80、95/100。
task9贡献整体AUC变化中的-3.750个百分点，约为净降幅4.028的93%。这是指标的
描述性分解，不是训练因果归因。task9没有后续阶段，故这一主要学习损失不会进入NBT。
另外，首学水平降低本身也可能降低后续差值型遗忘指标，不能仅用较小NBT判断更好保持。

seed42的task6最终从MLR的5%变为0%；约束最差峰值到最终遗忘为60个百分点，
MLR为40。三seed最大峰值到最终遗忘均值为26.667 vs MLR20.000，最差旧任务最终
成功率均值35.000 vs38.333。没有一致的旧任务保持优势。

## 实际投影与成本

| seed | 投影次数 | 全batch触发率 | 有效参考batch触发率 | 全batch平均修正范数比例 | 训练分钟 | 评估分钟 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 926 | 0.878% | 1.037% | 0.138% | 26.87 | 38.56 |
| 43 | 842 | 0.798% | 0.942% | 0.126% | 27.20 | 42.65 |
| 44 | 905 | 0.858% | 1.013% | 0.128% | 27.78 | 40.26 |

正式投影已实际触发，并非开关失效。有效参考batch约89300/seed；无旧样本时保留
原MLR更新。全batch平均修正比例约0.13%包含未投影的零值，不能解释成每次触发都很小。
参考直接来自混合梯度包含的旧样本部分，较高一致性是可能的；本轮不以约1%触发率
单独证明低收益原因，也不据此人为调阈值。

PyTorch阶段峰值allocated为656416256字节（626MiB）；不等于驱动显存或reserved。
observe累计每seed约23.91–23.96分钟；同步计时本身有开销。没有同期同负载完整native
MLR/shadow计时对照，所以**只报告绝对成本，不能给出可信的额外开销百分比**。

投影约束的是原始梯度，不是AdamW实际位移；动量、预条件、weight decay、旧任务梯度
抵消和非线性均可能改变有限步保留效果。本轮不能区分这些解释，亦未验证复杂控制器。

## 可复核资产与实际命令

- `aggregate_v1_results.json`：全部指标、矩阵、配对差和样本标准差。
- `COMPLETION_AUDIT.json`：启动参数、版本、完成检查、配置/LR一致性、绝对耗时。
- `success_matrices.csv`、`per_task_results.csv`、`projection_epochs.csv`：完整小型结果表。
- `real_parity_seed42.json`：真实策略精确一致性。
- 原始日志、checkpoint、buffer保留在忽略的`runs/replay_constraint/aggregate_v1/`及
  `buffers/replay_constraint/`，不提交大型资产。历史基线未覆盖。

实际执行（仓库根目录，原Python环境）：

```bash
nohup bash replay_constraint_experiment/run_queue.sh > runs/replay_constraint_driver/queue.log 2>&1 < /dev/null &
/home/dm/miniconda3/envs/wzp_mlr_tmp/bin/python replay_constraint_experiment/export_results.py
```

队列内的train、validate、eval、summarize命令见`replay_constraint_experiment/run_queue.sh`；
原输出非空时不可直接重跑。export_results仅读取已有实验并刷新小型报告证据。

## 下一步

保留本轮为负结果对照，继续采用原MLR和修正IFA作为基线。不自动启动新训练、不调阈值；
下一轮方法选择需单独决策，本结果不支持直接将该约束替换为默认基线。
