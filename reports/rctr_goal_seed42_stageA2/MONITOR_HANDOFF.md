# A2 handoff

Read USER_PROTOCOL.txt, PROTOCOL.md, RECOVERY_ADDENDUM.md, RECOVERY_DIFF.patch, RUN_STATUS.json and LATEST_PROGRESS.json first. This is independent of old historical Stage A, which remains BLOCKED.

SSH root@jq1.9gpu.com:11920 is authorized. Credentials only through memory/getpass; never store them. `check_progress.py` prompts through getpass, checks live processes/GPU/disk and both original/recovery frozen input hashes, then downloads only a fixed light allowlist. Run `audit_progress.py` afterwards. Do not download full states/actions/noise/history/checkpoints.

Remote W: `/data/mlr_ifa/experiment_workspaces/RCTR_GOAL_seed42_stageA2_20261020`.
Original O: `/data/mlr_ifa/experiments/RCTR_GOAL_seed42_stageA2_20261020`.
Current E: `O/engineering_recovery/dedicated_env_v1`.

Original run exited1 after135 prefix +48 engineering control steps; first screening branch's env.step failed when an old sensor callback used an MjSim destroyed by a later hard reset. Zero completed branches; one technical attempt, unknown partial internal physics substeps. Keep original logs/exits/contracts.

User separately approved another135 prefix +48 engineering steps and one technical retry:333 unique planned screening branches, at most334 attempts total including the original failure. Recovery has a dedicated live environment for each snapshot. Exact original physical/observation/noise/initial hashes must match. Gates are run after all three environments are constructed. Native1/4/8 history action equivalence and two8-step repeats at all three snapshots have passed with zero errors. Original source untouched; external recovery source/diff are frozen.

Recovery supervisor PID45986; worker45995 at launch (verify, never rely only on PID). `run_recovery.sh` automatically runs recovery_runner.py then analyze.py. Each branch is committed separately; LIVE_STEP.json marks start/end of latest env.step. Parent RA GOAL42 T6 and official frontend are frozen; no optimizer exists, no training or buffer changes. Fixed333-case grid,3×37×3, matches PROTOCOL.md. Independent discovery/confirmation or training not authorized. No extra seeds/initials/candidates/horizons or new paid resources.

Partial environment replay cannot be blindly restarted. If another engineering failure occurs, preserve all records and exits. Do not resample normal task failures. Do not silently change frozen source/thresholds or add more collection/retry budget. User already approved one specific recovery; additional unsafe/rebudgeting recovery is BLOCKED with concrete required decision.

Once runner/analyze/supervisor all exit0 and both processes are gone, download light artifacts, run independent analyze.py through audit_progress.py. Verify333 unique records, their matched noise/candidate grid, exact counts and complete attempt ledger, source/input protection, strict engineering gate and recollection hashes. Review all111 state-candidate summary rows and198 paired summary rows. Aggregate original and recovery acquisition/engineering cost (366 completed control steps), original partial failed step, and all screening steps. Do not call screening success a training certificate; significance/generalization is unproven with three exploratory states and multiple candidates.

Every check/audit: update reviewed REPORT.md, run publish.py to publish this directory only to publish_mlr_branch remote MLR, verify receipt. No forcepush or binary/private data. Root remote RUN_STATUS initially remains BLOCKED by design; current recovery status is under E. Recovery final RUN_STATUS says AWAITING_ANALYSIS; determine actual completion from execution/analysis/exit/process evidence and local audit.

Automation `rctr-stage-a2` is hourly. Normal checks should finish after saving progress and ETA; do not watch continuously. Notify only material progress/change/failure/completion. Delete this automation after final acceptance, complete publication and delivery. Do not start any further experiment.

## Latest user cancellation, 2026-10-20
A2 is STOPPED_BY_USER. Worker and supervisor exit143, 76 committed records retained, zero optimizer calls. Automation rctr-stage-a2 deleted. Do not resume or treat partial results as acceptance. See ../20261020_three_methods/STOP_ACCEPTANCE.json and stop_evidence. Supersedes earlier continuation instructions.
