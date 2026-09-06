# 修正基线后的干扰诊断

## 结论

**FROZEN_DIAGNOSIS_COMPLETE — MIXED / CAUSAL_MECHANISM_UNRESOLVED**

离线诊断完成，不是新方法实验，也不是新的论文复现 verdict。
已有证据不支持把剩余差距概括为“某个固定轨迹阶段存在一致的 IFA 冲突”。
task7 的成功率差距有 seed/阶段差异；task9 的模型函数差异在正式学习它之前已存在，
但 BC loss 和本次梯度余弦都不足以单独解释最终 rollout 成功率。
因此**不据此实现梯度门控，不修改 IFA，不继续调参**。

## 冻结与执行

- 起点：`d75fb371cf575beb079ce064d70d2ae323da95dd`。
- 分支：`codex/corrected-baseline-interference-diagnosis`。
- 在同一服务器、同一环境、GPU0单张RTX4090执行。
- MLR、旧取样IFA、修正IFA，seed42/43/44，各task6–9的ep99：共36个既有checkpoint。
- 没有新增训练、optimizer step、rollout或checkpoint选择；所有训练资产保留。
- 1,200条固定feature探针，与首10条demonstration完整action序列逐条唯一匹配；
  匹配的轨迹进度与既有元数据一致。每任务五段、每段20条，分两批10条。
- 所有36个checkpoint严格加载；分析前后权重逐张量一致；task6三组结果逐项完全一致。
- 输出1,440条BC记录、2,160条BC任务对记录、1,920条IFA-BC关系记录。

`COMPLETE.json`记录完成检查，`probe_manifest.csv`记录真实demo、序列位置和样本。
这是缓存demonstration分布上的离线分析，不是历史混合replay batch重放；
旧任务探针也不局限于当时buffer中的样本。

## 1. task7：不能用统一的 BC 欠拟合解释

以下BC为连续动作GMM负对数似然，越低越好；负值正常，不是成功率。
ΔBC=修正IFA−MLR，按五个进度段等权。成功率来自已完成的正式20-trial评估。

| seed | task7刚学完 ΔBC | task7首学成功率 MLR→修正IFA | task9学完后的task7 ΔBC | 此时task7成功率 MLR→修正IFA |
| --- | ---: | --- | ---: | --- |
| 42 | +0.035 | 100→100 | +2.716 | 85→65 |
| 43 | +0.432 | 100→85 | -0.100 | 95→90 |
| 44 | -0.225 | 100→95 | -0.575 | 100→100 |

- seed43首学BC变差与首学成功率下降方向一致，但只是描述性一致。
- seed44首学BC反而更好，成功率却从100降至95：不能把BC改善等同于成功率改善。
- seed42首学没有成功率缺口，主要问题出现在后续保留：task7成功率在stage8均为95，
  stage9后修正IFA为65、MLR为85、旧取样IFA为100。
- seed42最终task7的五段ΔBC依次约为`+1.68,-0.43,+3.36,+5.66,+3.31`，
  本探针中的退化偏向后半段；但不能将这五段未经标注就命名为抓取/搬运/放置，
  也没有跨seed一致的同段退化模式。

证据：`bc_by_bin.csv`、`task7_task9_bc_deltas.csv`；成功率来源为冻结基线的
`reports/own_reference_control/CONTROLLED_RESULTS.json`。

## 2. task9：早期差异存在，但“更好的进入状态BC”不是一致解释

task9无直接IFA pair。以下比较均为修正IFA−旧取样IFA。

| seed | stage7后task9 ΔBC | stage8后、进入task9前 ΔBC | task9学完 ΔBC | task9正式成功率 旧→修正 |
| --- | ---: | ---: | ---: | --- |
| 42 | +7.680 | +0.773 | -0.393 | 75→80 |
| 43 | +0.368 | +8.566 | -0.293 | 25→80 |
| 44 | -26.215 | -20.625 | -0.279 | 75→100 |

stage6时全部差异精确为0，stage7后已出现差异。因此它们先于task9学习发生，
并非task9阶段启用了额外IFA。seed43/44只在task7有pair，随后函数差异继续存在。

但seed43在进入task9前的BC更差，最后成功率却大幅更高；seed44则BC和成功率
改善方向一致。这排除了一个简单、统一的解释：“修正只是让进入task9时的
demonstration BC更低，所以task9最终更成功。”尚未分离具体优化轨迹、动作分布
校准与闭环状态分布变化的贡献。

## 3. 梯度方向随进度段变化，但没有稳定的通用冲突段

分析使用eval/dropout-off的固定模型，取完整可训练decoder+head梯度。
IFA使用修正own-reference及原gamma0.30、weight0.1、原pair。
IFA不直接依赖head，因而全参数余弦也受BC head分量影响；接近0不能解释为
“IFA没有作用”。这不是AdamW更新方向，也不是历史train-mode梯度。

task7刚学完时，修正IFA梯度与task7 BC梯度的各段平均余弦：

| seed | 0–20% | 20–40% | 40–60% | 60–80% | 80–100% |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | +0.00994 | +0.00606 | +0.00184 | -0.00074 | +0.00130 |
| 43 | +0.00720 | +0.00591 | -0.00242 | -0.00261 | +0.00541 |
| 44 | +0.00499 | -0.00506 | -0.00568 | +0.00090 | -0.00447 |

正负号确实改变，但幅值小、每段仅两个探针batch，且跨seed负号位置不一致。
task7 hinge激活比例为93%/96%/99%，所以不能把这种不一致解释成IFA完全没有激活。
这些是end-of-stage局部几何，不能反推出训练全程一直处于同样冲突。

另一个直接反例：stage8末、进入task9前，修正IFA模型的task7/task9 BC梯度余弦，
seed42五段均为正，约`0.084,0.200,0.031,0.090,0.038`，随后task7成功率仍由95降至65。
seed43早两段为负、后三段为正；seed44五段为正。故静态同段BC余弦不能当成
闭环保留保证。未来AdamW更新方向、不同batch混合及状态分布都不在这一探针内。

seed42的8→7 pair，在stage8末对task7 BC的平均余弦约`+0.000125`，部分分段为负；
同样不足以把后来的task7下降直接归因于这一pair。

证据：`ifa_pair_by_bin.csv`、`ifa_at_pair_stage.csv`、`bc_pair_by_bin.csv`。
完整文件也保留非当前stage的IFA反事实探针，不能将其误报成当时启用的训练loss。

## 边界与可信程度

已确认：冻结模型可重复测得的函数差异、分段梯度正负变化、指标与成功率的不完全一致。
未确认：某个行为阶段是成功率差距的原因，某个梯度门控能改善rollout，或修正IFA
普遍优于MLR。没有新增因果干预支持这些主张。

100条/任务探针来自用于训练的缓存数据，不是独立held-out集；demo/相邻片段相关，
未按demo等权；二十条/段只是探索性采样。五段是序列起点的归一化时间位置，
不是语义行为阶段；跨任务同进度也不等于同一行为。没有做显著性检验。
正式success矩阵仍是此前结果，本轮未把离线探针伪装为新rollout成绩。

## 运行命令与文件

在服务器仓库根目录执行成功：

```bash
bash scripts/run_corrected_baseline_diagnosis.sh
/home/dm/miniconda3/envs/wzp_mlr_tmp/bin/python scripts/summarize_corrected_baseline.py
```

实际首次启动以nohup后台运行，日志在`runs/corrected_baseline_diagnosis/diagnosis.log`。
运行器固定GPU0，已完成产物有防覆盖检查。没有剩余训练队列。

新增三个脚本：`diagnose_corrected_baseline.py`、`summarize_corrected_baseline.py`、
`run_corrected_baseline_diagnosis.sh`；新增本报告目录的协议、manifest、原始CSV、汇总CSV、
`SUMMARY.json`与`COMPLETE.json`。`libero/`及所有算法配置相对冻结基线均未修改。

## 唯一建议的下一步

**先做固定checkpoint、配对initial-state的失败rollout定位，而不是开发梯度门控。**
优先比较seed43 task7首学、seed42 task7在stage9后的保留、seed43 task9终局差异；
记录轨迹与失败发生位置，以检验离线BC无法解释的闭环差距。保持原模型和训练不变，
不挑选成功trial、不调参。本轮未启动这批新rollout。
