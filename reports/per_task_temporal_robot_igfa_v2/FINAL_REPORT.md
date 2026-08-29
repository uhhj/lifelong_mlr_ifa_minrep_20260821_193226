# Per-Task Temporal Robot-IGFA V2

## Identity

- Verdict: `PER_TASK_TEMPORAL_ROBOT_IGFA_NOT_SUPPORTED`
- Practical benefit vs MLR: `SUPPORTED`
- Branch: `research/per-task-temporal-robot-igfa-v2`
- Starting SHA: `83d98eb1e553fe08fae6019f0cbc874c18859115`

## Method

Policy-head only; five progress bins; one constraint per present old task; order-independent minimum-change projection by active-set enumeration (at most three constraints); alpha 1; Temporal V1 delta correction; transformer unchanged; IFA disabled; replay and historical scheduler semantics unchanged.

## Tests and smoke

Focused solver, order-invariance, masked-conflict, feasibility, minimum-change, and no-anchor tests passed. The formal runs were admitted only after the tasks 6-8 smoke observed multi-old constraints with zero post-projection violations.

## Real-batch parity

- Gate=0 head max abs / relative L2: 0 / 0
- Gate=0 transformer max abs / relative L2: 0 / 0
- Single-old V1 max abs / relative L2: 4.76866262e-08 / 7.72277455e-07

## Per-seed metrics (FWT / NBT / AUC)

- seed42: 80.000 / 14.444 / 74.062
- seed43: 81.250 / -3.056 / 83.021
- seed44: 85.000 / 18.889 / 75.417

## Three-seed mean ± sample std

- FWT: 82.083 ± 2.602
- NBT: 10.093 ± 11.601
- AUC: 77.500 ± 4.829

## V2 vs Temporal V1

- ΔFWT: 1.250; same-seed wins 2/3
- ΔNBT: 4.352; same-seed wins 0/3
- ΔAUC: -0.972; same-seed wins 1/3

## V2 vs MLR

- ΔFWT: 4.583; same-seed wins 3/3
- ΔNBT: 7.315; same-seed wins 0/3
- ΔAUC: 1.216; same-seed wins 2/3

## Mechanism

- Candidate bins: 522901
- Multi-old candidate bins: 112252
- Initial-conflict bin rate: 49.390%
- Mean active constraints: 0.019746
- Active-count distribution: {'0': 512804, '1': 9875, '2': 216, '3': 6}
- Mean projection cost: 0.085385
- Post-constraint violation rate: 0.000000%
- Compatible-became-active: 6
- Zero-old-gradient count: 0

## Seed44 task7

- V2 diagonal / post-stage8 / post-stage9: 95.000 / 80.000 / 40.000
- V1 forgetting: 47.500
- V2 forgetting: 35.000
- Delta: -12.500

## Observed

All metrics and mechanism statements above come from the frozen three-seed formal protocol and complete 30-record evaluation logs.

## Inferred

H7 verdict is `PER_TASK_TEMPORAL_ROBOT_IGFA_NOT_SUPPORTED` under the preregistered descriptive thresholds.

## Not established

These three seeds do not establish statistical significance or performance outside LIBERO-OBJECT.

## Next action

If supported, proceed to method consolidation/ablation; otherwise inspect projection cost and task/stage localization before changing alpha or method scope.

## Exact successful commands

```bash
for seed in 42 43 44; do
  SEED=$seed bash scripts/run_object_mlr_per_task_temporal_robot_igfa.sh
  SEED=$seed bash test.sbatch mlr_per_task_temporal_igfa
done
python scripts/compute_per_task_temporal_robot_igfa_metrics.py
```
