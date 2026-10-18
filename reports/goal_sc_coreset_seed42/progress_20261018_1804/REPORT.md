# SC-Coreset GOAL seed42 progress — PARTIAL

Observation: 2026-10-18 18:04:20 CST metadata package; live process/input audit completed 18:05:20 CST. This is a progress audit, not final experimental acceptance.

Training completed all 400 epochs and 76,100 optimizer steps, train.exit=0. Four epoch99 checkpoint files exist and were hashed; full post-run independent strict loading remains pending. The Task6 epoch99 actual state_dict comparison with the parent Coreset is equal for every tensor, maximum absolute difference 0. Training took 2,483.60 seconds (41.39 minutes), including orchestration/storage costs.

The frozen source, configuration, input, feature and modal contract checks passed. Protected parent files and actual buffer payload SHA256 checks passed. Formal stored members/order exactly match preflight, quotas and minimal exchanges pass for every joint group, and each task retains all ten demonstrations.

| Task | Stored windows | Replaced original members | Original radius | SC radius |
|---|---:|---:|---:|---:|
| 6 | 499 | 70 | 0.278769 | 0.337656 |
| 7 | 445 | 38 | 0.261264 | 0.293265 |
| 8 | 452 | 38 | 0.290898 | 0.322380 |
| 9 | 985 | 39 | 0.241957 | 0.262762 |

Task6 positive final-command proportion: full population 41.4243%, original Coreset 54.7094%, SC 41.4830%. True endpoint targets: 8.0241%, 2.8056%, 8.0160%, respectively. The requested supervision-composition correction really entered the training memory; its cost is a larger geometric covering radius. This does not yet establish a closed-loop benefit.

Actual IFA selected pairs remain identical to the parent: Task6 [], Task7 [(6,7)], Task8 [(6,8)], Task9 []. Similarity values and memory payloads differ; equality of selected pairs is not equality of all IFA inputs.

Evaluation snapshot: 20/200 formal episodes, 2/20 completed vector attempts, one active vector in stage7/task6, no technical exceptions. RUN_STATUS records the last completed stage6/task6 cell. RTX3090 utilization 100%; evaluator and pipeline are live. System filesystem free ~20.86 GiB. No restart, extra optimizer update or rollout was requested by this check.

The first two completed ten-lane vectors took 288.32 and 278.06 seconds (both 600 steps). A simple remaining-18-vector extrapolation is 84.96 minutes, excluding initialization/exit audit and ignoring elapsed time in the current vector. Only two vectors support this estimate; later tasks may finish earlier. Working finish estimate including final audit: **19:20–19:45 CST on 2026-10-18**. No complete AUC/OldFinal or gate verdict is available yet. Seed43/44 remain unstarted; expansion requires AUC>78.75%, OldFinal>=68.333333% and all final engineering checks.

Commands (exit codes):

```sh
source /root/mlr_ifa/experiment_workspaces/SC_Coreset_GOAL_seed42_20261018/env.sh
/data/mlr_ifa/env/bin/python /root/mlr_ifa/experiment_workspaces/SC_Coreset_GOAL_seed42_20261018/monitor_snapshot.py  # 0
/data/mlr_ifa/env/bin/python /root/mlr_ifa/experiment_workspaces/SC_Coreset_GOAL_seed42_20261018/package_light.py  # 0
/data/mlr_ifa/env/bin/python /tmp/sc_readonly_progress_1804.py  # 0; archived here as remote_readonly_audit.py
python -X utf8 verify_progress.py  # 0; local, metadata-only recomputation
```

Remote outputs: `/root/mlr_ifa/experiments/SC_Coreset_GOAL_seed42_20261018`.
Package manifest verifies 56 immutable downloaded metadata files; real action vectors were removed from binding metadata by the existing packaging script. No weights, latent tensors, demonstrations, buffer payloads, RNG binary files or archive are published. Checkpoint hashes are metadata only.

Limits remain single-seed exploration, shared global action RNG reset per stage rather than isolated trajectory RNG, unknown original pretraining modality metadata, disclosed scheduler/final-epoch completion, and possible indirect IFA effects. Final verification and all 200 episodes are required before an efficacy claim.
