# SC-Coreset GOAL seed42 progress — PARTIAL

Observation: **2026-10-18 18:54:43 CST**. Training is complete (400 epochs, 76,100 optimizer steps, train.exit=0). Evaluation is still running on RTX3090; GPU utilization 100%. The pipeline/evaluator remain alive, and evaluate/audit/pipeline exit codes are not yet present. No technical exception or restart occurred.

**180/200 formal episodes, 18/20 vector attempts completed.** The active vector is stage9/task9/group0. All previously completed episodes were retained. This check only reads existing results and hashes, without training, model loading or environment execution.

| Stage | Task6 | Task7 | Task8 | Task9 |
|---|---:|---:|---:|---:|
| 6 | 65% | — | — | — |
| 7 | 70% | 100% | — | — |
| 8 | 60% | 100% | 70% | — |
| 9 | 60% | 100% | 50% | pending |

Each reported cell contains all 20 prescribed episodes. Final old-task mean is already determined: **70%**, versus parent Coreset **68.333333%**, a +1.666667pp difference. This satisfies only the OldFinal component of the preregistered gate. AUC and the full engineering acceptance are pending; do not infer an extension decision from partial results. Seed43/44 have not started.

The frozen source/configuration/input/features/modal contract, protected baseline files and actual buffer payload hashes were rechecked and passed (remote read-only command exit0). All four stored member lists still match their original preflight order exactly: quotas 499/445/452/985, minimum replacements 70/38/38/39. Task6 model tensor parity proof remains equal. The prior 18:04 report contains the full composition/geometry diagnostics; no selection algorithm or frozen code was changed.

Measured completed-vector mean is 202.85 seconds. The last four vectors took 58.97, 39.10, 283.10 and 280.87 seconds. Two remaining vectors imply approximately 5.5–6.8 minutes by these means, or about 9.5 minutes if both reach the 600-step cap. With model loading, shutdown and final audit overhead, revised expected finish is **19:05–19:15 CST**. Earlier ETA used only two full-length vectors; several later cells finished much sooner. This is an estimate, not a completion claim.

Commands: source the existing W/env.sh, then `/data/mlr_ifa/env/bin/python /tmp/sc_readonly_progress_1854.py` (exit0), followed by local `python -X utf8 verify_progress.py` (exit0). W is `/root/mlr_ifa/experiment_workspaces/SC_Coreset_GOAL_seed42_20261018`; output root is `/root/mlr_ifa/experiments/SC_Coreset_GOAL_seed42_20261018`. The remote script is archived here as `remote_readonly_audit.py`.

`PACKAGE_MANIFEST.json` binds the downloaded lightweight snapshot. The local verifier independently checks hashes, 400 epochs/76,100 steps, unique episodes, attempt accounting, cell counts and timings. Only metadata, episode outcomes and read-only scripts are published; no checkpoint, RNG binary, action payload, latent, demonstration or buffer payload is included.

Final acceptance still requires all 200 episodes, four strict checkpoint loads, the frozen audit PASS, pipeline exit0 and no remaining processes. Expansion requires AUC strictly above 78.75% and OldFinal at least 68.333333%, plus engineering PASS. No scientific setting or gate has changed.
