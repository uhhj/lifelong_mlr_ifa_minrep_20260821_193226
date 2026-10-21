# CAET-v2 OBJECT seed42

Read USER_METHOD.txt/PROTOCOL.md/IMPLEMENTATION_NOTES.md/RUN_STATUS.json. Gate1 independent PASS:gate1_20261021_200229. Actual sidecar precision parity PASS:sidecar_precision_20261021_201904. No extra seeds or baseline retrain/reeval. All COVR remains cancelled; LIBERO50 pending behind this request.

W=/data/mlr_ifa/experiment_workspaces/CAET_v2_OBJECT_seed42_20261021; O=/data/mlr_ifa/experiments/CAET_v2_OBJECT_seed42_20261021. SSH authorized root@jq1.9gpu.com:11920, password getpass/memory only. Source immutable W/source; source15 additions manifest launch_20261021_201227. Local frozen-source check PASS.

Initial supervisor66654 failed in LIBERO import, no optimizer/model/environment calls. Original FAILURE/log and supervisor.exit1 preserved. Separate recovery_environment01.py fills existing runtime environment only (LIBERO_CONFIG_PATH/HF/etc); no source/contract change. Actual recovery supervisor66931, pipeline66966. Gate2 exactly8 updates passed, no repeats. Its exit is admission/gpu_preflight_environment01.exit0. Final acceptance must use recovery_environment01.exit0 plus pipeline.exit0 and PROCESS_EXIT_ACCEPTANCE, while retaining original failed exit1. Never blindly relaunch deploy.py or either supervisor.

Check:python check_progress.py (getpass), then python audit_progress.py <snapshotdir>. Latest progress_20261021_202251 independently passed, Task6 72epoch11304step, formal0. Task6 at100epoch must pass actual same-hardware tensor/initialization equality. Then Task7 fallback;8/9 graph+heldout fitted predictor admission required, entire-stage fallback on failure. No scientific threshold changes. Four100epoch105500steps4ep99CP200formal20vector original global RNG protocol. Native BC a[u], auxiliary a[u+1] timing distinction remains explicit.

Checkpoint captures model/optimizer/scheduler/grads/modes/allRNG/step AND v2 mu/bounds/support/predictor hashes. Resume only complete epoch after stopped-process/contract checks, preserve failure. Partial global-RNG evaluation is not blindly restartable. No normal-result retries.

After actual recovery supervisor exit0/no workers:python package_final.py downloads only explicit metadata/episode allowlist and hashes; python audit_final.py <finaldir> independently verifies all400epochs105500steps/4CP proof/RA members+IFA/constraints/200episodes20vectors and compares exact metrics with RA and separately v1. These scripts have only been syntax checked so far, not final-executed. External read-only audit bugs may be repaired with evidence, never rerun environments for summarization. Preserve source and full original failed attempts.

Per check/gate publish reviewed code/report/light evidence using publish.py to publish_mlr_branch MLR, verify remoteSHA/receipt; no weights/raw actions/latent/buffer/noise/RNG binary. Existing same automation rctr-caet-covr-libero50 now prioritizes this v2 run every30min. Only meaningful progress/failure/completion notifications. Do not claim complete before local final audit. No automatic expansion after this single seed.


Update 2026-10-21T20:27:19.064850+08:00: Task6 completed100epochs/15700steps, actual100 tensor keys exactly match existing RA Task6, maximum difference0; initialization and781 ordered retained members/payloads match. All219 frozen source hashes pass. train6/retain6/labels7 exited0. Pipeline entered Task7 admission; its graph has no qualified edges and must fall back to RA+IFA. No formal evaluation has run. Evidence:handoff_20261021_202636/LOCAL_AUDIT.json.


## Progress 2026-10-21 20:33 CST

Observation completion: 2026-10-21T20:32:51.600035+08:00. The snapshot is not atomic: the epoch ledger was read before the hash sweep completed.

- Main training: 139/400 complete epochs, 24553/105500 updates. Task6 complete; Task7 39/100 epochs.
- Task7 predictor admission: FALLBACK_RA / NO_QUALIFIED_GRAPH, zero effect updates. Across all 8853 recorded Task7 policy updates, transfer and retention activation are both zero, as required by the frozen unsupported-stage fallback.
- Task6 actual tensor equality with RA remains PASS. All 518 source/input hash checks and local epoch/auxiliary accounting PASS. Four expected supervisor/pipeline/training processes remain active; no new recovery or duplicate launch.
- Formal evaluation: 0/200 records. Gate3 remains RUNNING_NOT_ACCEPTED. No performance metrics inferred from training or functionality tests.
- Task7 measured mean 6.455 seconds/epoch: remaining 61 epochs approximately 6.56 minutes from ledger sampling. Task8/9 predictor admission and active-gradient cost remain unmeasured; whole-run ETA is not yet reliable. Prior formal evaluation reference is approximately 1.3 hours, with final audit additional.
- Free disk: data 10.21 GiB; system 5.50 GiB.

The original import failure and supervisor.exit=1 remain preserved. Only recovery_environment01 and the eventual pipeline/process acceptance determine the active attempt's final outcome. No scientific source, contract, training budget, or evaluation protocol changed during this check.


## Progress 2026-10-21 21:46 CST

Full source/input snapshot: 2026-10-21T21:44:24.447451+08:00. Later read-only mechanism ledger: 2026-10-21T21:45:28.418907+08:00. These are sequential observations, not an atomic snapshot.

Main training has reached 394/400 complete epochs and 103328/105500 optimizer updates (Task9 94/100). Formal evaluation was 0/200 at the full snapshot. All 518 frozen source/input hashes pass; active recovery supervisor, pipeline and Task9 trainer are normal. Original failed import attempt remains preserved; no new recovery, training, test updates or rollouts were launched by this check.

| Check | Task8 (100 epochs) | Task9 (94 epochs) |
|---|---:|---:|
| Whole-demo heldout admission | PASS | PASS |
| Normalized heldout MAE | 0.269519 | 0.254410 |
| Training-mean baseline MAE | 0.479750 | 0.457105 |
| Fixed action-permutation MAE | 0.296241 | 0.288566 |
| Supported heldout times | 73 | 64 |
| Old task-demo support identities | 11 | 17 |
| Effect fitting updates (10 epochs) | 840 | 980 |
| Main updates observed | 30900 | 34028 |
| Transfer accepted updates | 7065 | 8372 |
| Transfer rejected for negative BC-gradient dot | 5955 | 6706 |
| Protection active, nonzero loss updates | 2983 | 3144 |
| Completed epochs ending with positive protection coefficient | 29 | 29 |
| Maximum protection coefficient observed | 0.638998 | 0.578394 |

Action gradients are finite and nonzero for both predictors. Local independent verification checked all downloaded per-update ledger entries for accepted/blocked transfer consistency, nonzero protection loss, complete sequential step identities, support hashes and every epoch's bounded dual update. MECHANISM_LOCAL_AUDIT.json is PASS. Protection counts mean at least one positive-weight violated group in that optimizer update, not simply that the module was enabled. Task7 correctly stayed in RA fallback, with no predictor updates, transfer or protection.

The last 10 observed Task9 epochs averaged 23.30 seconds. Six epochs remain in the later ledger, approximately 2.33 minutes at that rate. Training completion is expected around 21:48 CST, followed by storage, all 200 formal records, checkpoint/process audits and local independent recomputation. Using the earlier matched evaluation runtime of about 79 minutes, a provisional full-completion window is **23:05-23:35 CST on 2026-10-21**. This is not yet a measured CAET-v2 evaluation ETA.

Both mechanisms are executing, but their success-rate benefit and retention performance remain unknown until formal evaluation. Main BC still supervises a[u], whereas the causal auxiliary target is a[u+1]; these different targets and exploratory support thresholds remain disclosed. The run remains PARTIAL, with no automatic extra seeds.
