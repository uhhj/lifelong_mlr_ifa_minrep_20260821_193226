# 已完成机制与设计对照实验统计（截至2026-11-01）

本次只读取已接受的本地轻量矩阵并重新计算指标，没有训练、仿真或远程检查。表中单位为%；差值为百分点。多seed表为种子均值，完整逐seed矩阵和样本标准差（ddof=1）见DATA.json。

AUC沿用项目的“各任务学习后各阶段均值，再跨任务平均”；FWT沿用项目的首次学会时对角线均值定义，不是相对零样本基线的迁移增益。NBT为后续相对首次学会时的下降，越低越好，可为负值；OldFinal为最终三个旧任务均值，FinalAll为最终四任务均值。

## 1. 早期选样比较：OBJECT，seed40–44首次运行

| 方法 | AUC↑ | FWT↑ | NBT↓ |
|---|---|---|---|
| random | 74.19 ± 7.73 | 71.75 ± 9.87 | -4.56 ± 10.33 |
| coreset | 80.00 ± 11.02 | 76.50 ± 10.73 | -6.44 ± 4.32 |
| event_coreset | 75.71 ± 8.36 | 71.25 ± 9.56 | -8.61 ± 5.08 |

共15个首次运行、3000正式回合；seed43另一次重复的600回合不纳入此均值。该五seed集合与后续三seed基线的取数口径不同，不能直接当成RA的单变量对照。

## 2. 原方法→RA→RA+ALIGN：seed41、42、43

| 任务套件 | 方法 | AUC | FWT | NBT | OldFinal | FinalAll |
|---|---|---|---|---|---|---|
| GOAL | MLR_IFA | 76.35 | 76.67 | 0.19 | 77.22 | 76.67 |
| GOAL | RA | 81.98 | 83.75 | 2.59 | 77.78 | 80.83 |
| GOAL | RA_ALIGN | 81.11 | 85.83 | 10.83 | 73.33 | 76.25 |
| OBJECT | MLR_IFA | 68.30 | 66.67 | -2.78 | 67.78 | 65.42 |
| OBJECT | RA | 73.23 | 73.33 | -0.09 | 75.00 | 69.58 |
| OBJECT | RA_ALIGN | 80.17 | 77.50 | -4.91 | 74.44 | 75.83 |

历史3方法×2套件×3seed，共18个200回合结果。OBJECT seed43原基线沿用冻结repeat1；本轮不重新挑选结果。历史基线不保证与后续S逐轨迹动作噪声相同。

## 3. CCBP-v1：共同Stage8起点，仅训练Task9，GOAL seed41

| 方法 | 最终Task8 | Task9 | 两任务平均 |
|---|---|---|---|
| A | 65 | 80 | 72.5 |
| B | 80 | 80 | 80.0 |
| C | 80 | 85 | 82.5 |
| D | 70 | 80 | 75.0 |

A=ALIGN；B=固定执行Normal蒸馏；C=持续专家cost+原margin路由；D=教师误差预算+可接受集合路由。共400epoch、134800主更新、160正式回合。这轮没有完整序列AUC，不能与下一表的AUC互换。

## 4. 完整GOAL seed41设计对照

| 轮次 | 方法 | 最终T6 | 最终T7 | 最终T8 | T9 | AUC | FWT | NBT | OldFinal | FinalAll |
|---|---|---|---|---|---|---|---|---|---|---|
| v2 | B | 35 | 100 | 85 | 55 | 76.25 | 83.75 | 14.44 | 73.33 | 68.75 |
| v2 | C | 30 | 100 | 85 | 90 | 83.12 | 90.00 | 12.22 | 71.67 | 76.25 |
| v2 | N | 30 | 100 | 85 | 75 | 78.75 | 88.75 | 18.89 | 71.67 | 72.50 |
| v2 | V | 45 | 100 | 80 | 80 | 82.50 | 88.75 | 12.22 | 75.00 | 76.25 |
| v3 | C | 40 | 100 | 100 | 90 | 87.50 | 92.50 | 7.78 | 80.00 | 82.50 |
| v3 | R | 35 | 100 | 60 | 80 | 77.81 | 88.75 | 22.22 | 65.00 | 68.75 |
| v3 | G | 50 | 100 | 80 | 85 | 83.12 | 93.75 | 21.11 | 76.67 | 78.75 |
| DOS | C | 40 | 100 | 100 | 90 | 87.50 | 92.50 | 7.78 | 80.00 | 82.50 |
| DOS | D | 55 | 100 | 60 | 75 | 78.33 | 86.25 | 16.39 | 71.67 | 72.50 |
| DOS | O | 45 | 100 | 85 | 90 | 87.50 | 92.50 | 9.44 | 76.67 | 80.00 |
| DOS | S | 45 | 100 | 100 | 95 | 86.88 | 91.25 | 5.56 | 81.67 | 85.00 |

v2：B固定教师蒸馏；C专家监督+原margin；N专家监督+代价差softplus路由；V混合评分J+新路由。
v3：C为该轮实际强对照；R保留J、回退margin；G专家监督+纠错线段锚+原margin。v2和v3的C是独立运行，不混并。
DOS：D示范历史锚；O实际全轨迹历史锚；S成功轨迹历史锚，均固定任务获得时教师。这里S是早期未门控版本；C复用v3，不增加一次实验。D/O/S各40训练采集，共120；缓存窗口为512/512/499（S的Task8为115），无补采。

## 5. 最新完整对照：GOAL与OBJECT各seed41–43

| 套件 | 方法 | AUC | FWT | NBT | OldFinal | FinalAll |
|---|---|---|---|---|---|---|
| GOAL | C_normal | 82.50 | 85.00 | 5.56 | 80.56 | 80.00 |
| GOAL | C_equal_budget | 83.72 | 87.08 | 7.59 | 77.22 | 80.00 |
| GOAL | Q | 83.51 | 85.83 | 4.44 | 78.33 | 80.42 |
| GOAL | S_latest | 87.26 | 89.58 | 5.28 | 83.89 | 85.00 |
| OBJECT | C_normal | 85.00 | 81.25 | -8.06 | 77.22 | 82.50 |
| OBJECT | C_equal_budget | 80.90 | 79.58 | -2.41 | 72.78 | 77.50 |
| OBJECT | Q | 84.62 | 78.75 | -10.93 | 78.33 | 82.92 |
| OBJECT | S_latest | 86.49 | 80.42 | -11.02 | 81.11 | 85.00 |

C_normal=原主更新预算强C；C_equal_budget=匹配S正式优化器调用数的C；Q=关闭门控的完整对照；S_latest=最新门控S。相同S结果跨三类对照只引用一次，不算三套新S实验。

最新S相对各对照（均值差，百分点）：

| 套件 | 比较 | AUC | FWT | NBT | OldFinal | FinalAll |
|---|---|---|---|---|---|---|
| GOAL | S - C_normal | +4.76 | +4.58 | -0.28 | +3.33 | +5.00 |
| GOAL | S - C_equal_budget | +3.54 | +2.50 | -2.31 | +6.67 | +5.00 |
| GOAL | S - Q | +3.75 | +3.75 | +0.83 | +5.56 | +4.58 |
| OBJECT | S - C_normal | +1.49 | -0.83 | -2.96 | +3.89 | +2.50 |
| OBJECT | S - C_equal_budget | +5.59 | +0.83 | -8.61 | +8.33 | +7.50 |
| OBJECT | S - Q | +1.88 | +1.67 | -0.09 | +2.78 | +2.08 |


## 6. 可以从对照中读出的结论

- 原教师预算D不如B/C；单seed结果支持继续采用持续专家监督，但不证明某个单步失败的因果机制。
- v2中N相对C：AUC -4.375pp、Task9 -15pp，新路由没有体现收益。V相对N的增益同时含教师、评分缩放和后续教师差异，不能全归因于一个权重。
- v3中G相对C：AUC -4.375pp，最终Task8 -20pp、Task9 -5pp；方向锚没有通过预先登记的探索门。
- DOS中O相对D：AUC +9.1667pp、FinalAll +7.5pp；S相对O：AUC -0.625pp、FinalAll +5pp。成功筛选体现保持收益，但不能宣称所有指标均提升，也非完全相同历史池上的纯筛选因果实验。
- 最新S相对等优化器预算C：GOAL/OBJECT的AUC分别+3.5417/+5.5903pp，说明仅增加C的更新数未解释全部优势；这不是等墙钟或等交互对照。
- 最新S相对完整Q：GOAL/OBJECT AUC分别+3.75/+1.875pp；六个seed-suite中AUC为5胜1负，FinalAll为3胜2平1负，仍保留13个负向阶段任务单元。S正式优化器调用1012500，Q为544800，不能把均值差解释为等计算下的纯门控增益。

## 7. 规模与去重说明

| 实验轮次 | 新增训练分支 | 训练epoch | 主更新/正式优化器调用 | 训练采集 | 正式回合 |
|---|---|---|---|---|---|
| v1 Task9分叉 | 4 | 400 | 134800 | 0 | 160 |
| v2完整 | 4 | 1600 | 304400 | 0 | 800 |
| v3完整 | 3 | 1200 | 228300 | 0 | 600 |
| DOS完整 | 3 | 1200 | 228300 | 120 | 600 |
| 最新S，两套件三seed | 6 | 2400 | 544800主更新 / 1012500物理调用 | 240 | 1200 |
| 等优化器预算C，两套件三seed | 6 | 4200 | 1012500 | 0 | 1200 |
| 完整Q，两套件三seed | 6 | 2400 | 544800 | 240 | 1200 |

此处列各轮规模，不把重用基线相加为全项目唯一总数；早期RA/ALIGN、普通C扩展及旧S扩展有独立账目。隔离准入、额外autograd、诊断和失败版本成本不混入正式主更新；完整审计原记录保留。本汇总没有将取消或未完成的分叉列为已完成科学结果。

所有seed41及既有初态均经历反复探索；三seed均值不等于独立确认结论。历史版本/硬件和训练交互成本见各原协议。

## 8. 数据来源（SHA绑定）

- `execution/20261017_five_seed_summary/RESULTS.json`
- `execution/20261017_five_seed_summary/REPORT.md`
- `execution/20261025_ccbp_v4_full/GOAL_THREE_SEED_BASELINES.json`
- `execution/20261025_ccbp_v4_full/OBJECT_THREE_SEED_BASELINES.json`
- `execution/20261024_ccbp/final_readonly_20261024_224310/FINAL_ANALYSIS.json`
- `execution/20261024_ccbp/final_readonly_20261024_224310/LOCAL_FINAL_AUDIT.json`
- `execution/20261024_ccbp_v2_full/final_readonly_20261025_054702/LOCAL_FINAL_AUDIT.json`
- `execution/20261025_ccbp_v3_full/final_readonly_20261025_181226/LOCAL_FINAL_AUDIT.json`
- `execution/20261025_ccbp_v4_full/final_readonly_20261026_014654/LOCAL_FINAL_AUDIT.json`
- `execution/20261029_support_gate_full_s_goal/final_readonly_20261030_111546/PRIMARY_C_COMPARISON.json`
- `execution/20261029_support_gate_full_s_goal/final_readonly_20261030_111546/LOCAL_FINAL_AUDIT.json`
- `execution/20261028_support_gate_full_s_object/final_readonly_20261029_201235/PRIMARY_C_COMPARISON.json`
- `execution/20261028_support_gate_full_s_object/final_readonly_20261029_201235/LOCAL_FINAL_AUDIT.json`
- `execution/20261030_equal_optimizer_budget_cs_two_suites/final_readonly_20261031_065009/PRIMARY_S_COMPARISON.json`
- `execution/20261030_equal_optimizer_budget_cs_two_suites/final_readonly_20261031_065009/LOCAL_FINAL_AUDIT.json`
- `execution/20261031_full_q_vs_latest_s_two_suites/final_readonly_20261101_032229/PRIMARY_S_COMPARISON.json`
- `execution/20261031_full_q_vs_latest_s_two_suites/final_readonly_20261101_032229/LOCAL_FINAL_AUDIT.json`
