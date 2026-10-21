# CAET-v2 OBJECT seed42 final acceptance

Independent metadata and exact episode recomputation PASS. Separate algorithm from v1.

|Metric|RA %|CAET-v1 %|CAET-v2 %|v2−RA pp|
|---|---:|---:|---:|---:|
|AUC|71.562500|77.083333|74.895833|+3.333333|
|FWT|70.000000|68.750000|72.500000|+2.500000|
|NBT|-3.055556|-16.944444|-5.833333|-2.777778|
|OldFinal|65.000000|83.333333|66.666667|+1.666667|
|FinalAll|63.750000|81.250000|70.000000|+6.250000|

Single-seed exploratory result; support thresholds selected after observing v1/GOAL. Native BC remains one-action-index earlier than causal auxiliary labels, so this experiment alone cannot isolate timing correction from the proposed mechanism. Global evaluation action RNG is not per-trajectory isolated. Historical-best retention refers to offline next-action NLL on retained samples, not guaranteed closed-loop success. CAET-v1 retains its post-action-input scientific limitation. FWT is the original trained-diagonal metric; lower NBT is favorable under the original definition. Extra effect fitting and functional-reference scoring cost must be reported. No other seeds are automatically authorized.

## Task-level interpretation

FWT uses the frozen mean trained-diagonal success definition, not untrained-task zero-shot forward transfer. V2 improves FWT by 2.5pp versus RA and 3.75pp versus v1; it is not an across-task uniform gain. The RA-to-v2 Task8 diagonal falls 90% to80%, while Task9 rises60% to80%; Task6/7 stay30%/100%.

|Task|RA diagonal|v1 diagonal|v2 diagonal|RA final|v1 final|v2 final|
|---|---:|---:|---:|---:|---:|---:|
|6|30.0|30.0|30.0|10.0|55.0|20.0|
|7|100.0|85.0|100.0|90.0|95.0|80.0|
|8|90.0|85.0|80.0|95.0|100.0|100.0|
|9|60.0|75.0|80.0|60.0|75.0|80.0|

Relative to RA, final old-task changes are Task6 +10pp, Task7 -10pp, Task8 +5pp: net +1 successful episode across60 old-task episodes (+1.6667pp). Relative to v1 they are -35pp/-15pp/0pp: old-task mean -16.6667pp. Thus the surrogate protection did not establish uniformly improved retention. Task6 still falls from its best measured80% to20% (60pp); Task7 falls from100% to80% (20pp). NBT is lower/better than RA by2.7778pp, but worse than v1 by11.1111pp; negative NBT can coexist with severe later loss because this metric averages changes relative to each task first-learned checkpoint, including earlier improvements.

No attribution to transfer alone or protection alone is established: causal rebinding, supported transfer, conflict gating and functional retention changed together, and Task7 intentionally falls back. V1 post-action-input leakage prevents treating its higher retention as validated causal-effect transfer, but its observed scores remain intact. Single seed,20 episodes per cell and global action RNG preclude strong generalization claims.

## Cost and constraint disclosure

Main105500 optimizer updates,8 isolated preflight updates,1820 additional predictor updates. Main summed epoch time 77.80 minutes; completed evaluation vector time 73.12 minutes. Epoch time includes auxiliary policy gradients and end-of-epoch full-group scoring; historical-reference setup is not separately timed.

```json
{
  "effect_fit_cost": {
    "8": {
      "optimizer_updates": 840,
      "fit_seconds": 5.559746026992798,
      "anchor_exposures": 26710,
      "total_window_exposures": 29780,
      "pair_exposures": 3070
    },
    "9": {
      "optimizer_updates": 980,
      "fit_seconds": 6.568246364593506,
      "anchor_exposures": 31270,
      "total_window_exposures": 35490,
      "pair_exposures": 4220
    }
  },
  "final_constraints": {
    "8": {
      "max_mu": 0.6389982775279452,
      "violated_group_epoch_count": 92,
      "final_violated_groups": 7,
      "groups": 17
    },
    "9": {
      "max_mu": 0.8111919374465953,
      "violated_group_epoch_count": 176,
      "final_violated_groups": 15,
      "groups": 23
    }
  }
}
```

The soft retention constraint remains violated at the final epoch for 7/17 groups in Task8 and15/23 in Task9. Maximum observed mu is0.639/0.811; activation does not imply feasibility or elimination of forgetting. These are reported scientific outcomes, not a reason to retry or alter thresholds.

## Complete success matrices

Each cell has20 trials; entries are RA / v1 / v2 percentages.

|Stage|Task6|Task7|Task8|Task9|
|---|---:|---:|---:|---:|
|6|30.0 / 30.0 / 30.0|-|-|-|
|7|80.0 / 65.0 / 80.0|100.0 / 85.0 / 100.0|-|-|
|8|35.0 / 40.0 / 35.0|95.0 / 100.0 / 85.0|90.0 / 85.0 / 80.0|-|
|9|10.0 / 55.0 / 20.0|90.0 / 95.0 / 80.0|95.0 / 100.0 / 100.0|60.0 / 75.0 / 80.0|
