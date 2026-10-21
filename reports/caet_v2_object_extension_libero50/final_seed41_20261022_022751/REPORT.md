# CAET-v2 OBJECT seed41 final acceptance

Independent metadata and exact episode recomputation PASS. Separate algorithm from v1.

|Metric|RA %|CAET-v1 %|CAET-v2 %|v2−RA pp|
|---|---:|---:|---:|---:|
|AUC|73.020833|71.250000|65.000000|-8.020833|
|FWT|75.000000|73.750000|67.500000|-7.500000|
|NBT|4.166667|3.888889|5.833333|+1.666667|
|OldFinal|78.333333|66.666667|73.333333|-5.000000|
|FinalAll|70.000000|66.250000|58.750000|-11.250000|

Single-seed exploratory result; support thresholds selected after observing v1/GOAL. Native BC remains one-action-index earlier than causal auxiliary labels, so this experiment alone cannot isolate timing correction from the proposed mechanism. Global evaluation action RNG is not per-trajectory isolated. Historical-best retention refers to offline next-action NLL on retained samples, not guaranteed closed-loop success. CAET-v1 retains its post-action-input scientific limitation. FWT is the original trained-diagonal metric; lower NBT is favorable under the original definition. Extra effect fitting and functional-reference scoring cost must be reported. This seed is part of the explicitly authorized fixed OBJECT41/43 extension.

## 逐任务与成本复算

FWT沿用各任务刚训练完成的对角线平均成功率；不是未训练任务的零样本迁移。NBT沿用原遗忘定义，越低越好。

|任务|RA 对角线|v1 对角线|v2 对角线|RA 最终|v1 最终|v2 最终|
|---|---:|---:|---:|---:|---:|---:|
|6|60.0|60.0|60.0|45.0|20.0|30.0|
|7|95.0|75.0|95.0|95.0|80.0|100.0|
|8|100.0|95.0|100.0|95.0|100.0|90.0|
|9|45.0|65.0|15.0|45.0|65.0|15.0|

旧组NLL约束可能在最终仍超限；有限双对偶惩罚不是严格可行性保证，更不是闭环成功率保证。迁移/保护激活、预测器留出通过不能替代实际结果。

主更新105500；隔离预检8；效果模型额外1840更新。主epoch累计76.93分钟，正式vector累计79.81分钟。参考评分无单独计时。负墙钟字段原样保留、标无效，不改成0、不重跑。

```json
{
  "effect_cost": {
    "7": {
      "status": "FALLBACK_RA",
      "optimizer_updates": 0
    },
    "8": {
      "optimizer_updates": 850,
      "fit_seconds_raw": -5.327922582626343,
      "fit_seconds_valid": false,
      "fit_seconds": null,
      "anchor_exposures": 26900,
      "total_window_exposures": 30120,
      "pair_exposures": 3220
    },
    "9": {
      "optimizer_updates": 990,
      "fit_seconds_raw": 6.393023490905762,
      "fit_seconds_valid": true,
      "fit_seconds": 6.393023490905762,
      "anchor_exposures": 31480,
      "total_window_exposures": 35740,
      "pair_exposures": 4260
    }
  },
  "constraints": {
    "8": {
      "max_mu": 0.6700560000731103,
      "violated_group_epochs": 102,
      "final_violated_groups": 4,
      "groups": 18
    },
    "9": {
      "max_mu": 0.8693873544037336,
      "violated_group_epochs": 146,
      "final_violated_groups": 12,
      "groups": 24
    }
  }
}
```

## seed41结果解读

相对RA，AUC下降8.020833pp，FWT下降7.5pp，NBT增加1.666667pp（原定义越低越好），最终旧任务平均成功率下降5pp。本seed没有改善FWT或总体旧任务保持。相对v1，AUC/FWT均下降6.25pp，NBT增加1.944444pp，但最终旧任务平均成功率提高6.666667pp。两种遗忘汇总并不等价，不应只选择其中更有利的一项。

FWT下降全部来自Task9对角线45%→15%，其余三个任务对角线与RA相同。最终Task6/7/8相对RA分别-15/+5/-5pp。Task8阶段Task6曾由RA的35%提高至65%，但Task9阶段降至30%；早期收益没有保持。Task9相对v1为65%→15%。这些是同初态结果差异，不代表已经隔离动作随机流或证明单一组件造成下降。

按冻结的任务等权AUC公式，Task6/7/8/9对总AUC差的贡献分别为+0.9375/-0.833333/-0.625/-7.5pp。因此主要数值损失集中在Task9；不能用Task8中间改善掩盖最终负结果。原v1有post-action输入科学问题，分数保留但不作因果迁移已成立的证据。

该seed的科学结果为负，不是应重新训练或重复评估的工程失败。seed43仍按用户明确固定授权继续，之后才汇总三seed；不改门槛、超参数或任务预算。
