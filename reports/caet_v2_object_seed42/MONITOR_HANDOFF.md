# CAET-v2 OBJECT seed42

Read USER_METHOD.txt/PROTOCOL.md/IMPLEMENTATION_NOTES.md/RUN_STATUS.json. Gate1 independent PASS:gate1_20261021_200229. Actual sidecar precision parity PASS:sidecar_precision_20261021_201904. No extra seeds or baseline retrain/reeval. All COVR remains cancelled; LIBERO50 pending behind this request.

W=/data/mlr_ifa/experiment_workspaces/CAET_v2_OBJECT_seed42_20261021; O=/data/mlr_ifa/experiments/CAET_v2_OBJECT_seed42_20261021. SSH authorized root@jq1.9gpu.com:11920, password getpass/memory only. Source immutable W/source; source15 additions manifest launch_20261021_201227. Local frozen-source check PASS.

Initial supervisor66654 failed in LIBERO import, no optimizer/model/environment calls. Original FAILURE/log and supervisor.exit1 preserved. Separate recovery_environment01.py fills existing runtime environment only (LIBERO_CONFIG_PATH/HF/etc); no source/contract change. Actual recovery supervisor66931, pipeline66966. Gate2 exactly8 updates passed, no repeats. Its exit is admission/gpu_preflight_environment01.exit0. Final acceptance must use recovery_environment01.exit0 plus pipeline.exit0 and PROCESS_EXIT_ACCEPTANCE, while retaining original failed exit1. Never blindly relaunch deploy.py or either supervisor.

Check:python check_progress.py (getpass), then python audit_progress.py <snapshotdir>. Latest progress_20261021_202251 independently passed, Task6 72epoch11304step, formal0. Task6 at100epoch must pass actual same-hardware tensor/initialization equality. Then Task7 fallback;8/9 graph+heldout fitted predictor admission required, entire-stage fallback on failure. No scientific threshold changes. Four100epoch105500steps4ep99CP200formal20vector original global RNG protocol. Native BC a[u], auxiliary a[u+1] timing distinction remains explicit.

Checkpoint captures model/optimizer/scheduler/grads/modes/allRNG/step AND v2 mu/bounds/support/predictor hashes. Resume only complete epoch after stopped-process/contract checks, preserve failure. Partial global-RNG evaluation is not blindly restartable. No normal-result retries.

After actual recovery supervisor exit0/no workers:python package_final.py downloads only explicit metadata/episode allowlist and hashes; python audit_final.py <finaldir> independently verifies all400epochs105500steps/4CP proof/RA members+IFA/constraints/200episodes20vectors and compares exact metrics with RA and separately v1. These scripts have only been syntax checked so far, not final-executed. External read-only audit bugs may be repaired with evidence, never rerun environments for summarization. Preserve source and full original failed attempts.

Per check/gate publish reviewed code/report/light evidence using publish.py to publish_mlr_branch MLR, verify remoteSHA/receipt; no weights/raw actions/latent/buffer/noise/RNG binary. Existing same automation rctr-caet-covr-libero50 now prioritizes this v2 run every30min. Only meaningful progress/failure/completion notifications. Do not claim complete before local final audit. No automatic expansion after this single seed.


Update 2026-10-21T20:27:19.064850+08:00: Task6 completed100epochs/15700steps, actual100 tensor keys exactly match existing RA Task6, maximum difference0; initialization and781 ordered retained members/payloads match. All219 frozen source hashes pass. train6/retain6/labels7 exited0. Pipeline entered Task7 admission; its graph has no qualified edges and must fall back to RA+IFA. No formal evaluation has run. Evidence:handoff_20261021_202636/LOCAL_AUDIT.json.
