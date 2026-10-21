# CAET-v2 OBJECT seed43 final acceptance

Independent metadata and exact episode recomputation PASS. Separate algorithm from v1.

|Metric|RA %|CAET-v1 %|CAET-v2 %|v2−RA pp|
|---|---:|---:|---:|---:|
|AUC|75.104167|75.937500|74.166667|-0.937500|
|FWT|75.000000|75.000000|75.000000|+0.000000|
|NBT|-1.388889|-2.222222|1.944444|+3.333333|
|OldFinal|81.666667|85.000000|81.666667|+0.000000|
|FinalAll|75.000000|75.000000|72.500000|-2.500000|

Single-seed exploratory result; support thresholds selected after observing v1/GOAL. Native BC remains one-action-index earlier than causal auxiliary labels, so this experiment alone cannot isolate timing correction from the proposed mechanism. Global evaluation action RNG is not per-trajectory isolated. Historical-best retention refers to offline next-action NLL on retained samples, not guaranteed closed-loop success. CAET-v1 retains its post-action-input scientific limitation. FWT is the original trained-diagonal metric; lower NBT is favorable under the original definition. Extra effect fitting and functional-reference scoring cost must be reported. This seed is part of the explicitly authorized fixed OBJECT41/43 extension.

## 逐任务与成本复算

FWT沿用各任务刚训练完成的对角线平均成功率；不是未训练任务的零样本迁移。NBT沿用原遗忘定义，越低越好。

|任务|RA 对角线|v1 对角线|v2 对角线|RA 最终|v1 最终|v2 最终|
|---|---:|---:|---:|---:|---:|---:|
|6|60.0|60.0|60.0|50.0|55.0|50.0|
|7|95.0|100.0|95.0|95.0|100.0|100.0|
|8|90.0|95.0|100.0|100.0|100.0|95.0|
|9|55.0|45.0|45.0|55.0|45.0|45.0|

旧组NLL约束可能在最终仍超限；有限双对偶惩罚不是严格可行性保证，更不是闭环成功率保证。迁移/保护激活、预测器留出通过不能替代实际结果。

主更新105500；隔离预检8；效果模型额外1820更新。主epoch累计72.28分钟，正式vector累计71.22分钟。参考评分无单独计时。负墙钟字段原样保留、标无效，不改成0、不重跑。

```json
{
  "effect_cost": {
    "7": {
      "status": "FALLBACK_RA",
      "optimizer_updates": 0
    },
    "8": {
      "optimizer_updates": 840,
      "fit_seconds_raw": 4.851348876953125,
      "fit_seconds_valid": true,
      "fit_seconds": 4.851348876953125,
      "anchor_exposures": 26820,
      "total_window_exposures": 30070,
      "pair_exposures": 3250
    },
    "9": {
      "optimizer_updates": 980,
      "fit_seconds_raw": 5.2822043895721436,
      "fit_seconds_valid": true,
      "fit_seconds": 5.2822043895721436,
      "anchor_exposures": 31360,
      "total_window_exposures": 35740,
      "pair_exposures": 4380
    }
  },
  "constraints": {
    "8": {
      "max_mu": 0.8028772184087173,
      "violated_group_epochs": 83,
      "final_violated_groups": 10,
      "groups": 17
    },
    "9": {
      "max_mu": 1.0,
      "violated_group_epochs": 175,
      "final_violated_groups": 15,
      "groups": 23
    }
  }
}
```
