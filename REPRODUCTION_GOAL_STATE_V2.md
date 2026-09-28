# GOAL seed42 MLR+IFA with explicit state

This branch publishes the completed `goal_seed42_mlr_ifa_paper_aligned_state_v2` experiment. The scientific source is `yfqi/lifelong_mlr_ifa` at `a721807f2f7fa7f9833dcc85c592d78e9accd00b` plus the exact [executed patch](reports/goal_seed42_state_v2/run/source.patch). The branch is based on that commit, rather than the destination repository's separate OBJECT experiment on `main`.

## Results and evidence

The completed run used one RTX 3090, training seed 42, tasks 6–9, ten demonstrations per task, 100 epochs per stage, batch 10, random storage rate 0.5, IFA weight 0.1 and gamma 0.7. All four final epoch99 checkpoints were evaluated in a separate process: 200 episodes, 20 per lower-triangular cell.

| Stage | Task 6 | Task 7 | Task 8 | Task 9 |
|---|---:|---:|---:|---:|
| 6 | 13/20 | — | — | — |
| 7 | 14/20 | 20/20 | — | — |
| 8 | 12/20 | 20/20 | 13/20 | — |
| 9 | 14/20 | 20/20 | 12/20 | 17/20 |

FWT **78.75%**, NBT **1.1111 percentage points**, AUC **78.4375%**. These are single-training-seed results. No rollouts or training were added during publication.

- [Final report](reports/goal_seed42_state_v2/REPORT.md)
- [Actual commands and exit codes](reports/goal_seed42_state_v2/run/logs/commands.log)
- Resolved configs: [export](reports/goal_seed42_state_v2/run/resolved_export.yaml), [training](reports/goal_seed42_state_v2/run/resolved_adaptation.yaml), [evaluation](reports/goal_seed42_state_v2/run/resolved_eval.yaml)
- [Modal contract](reports/goal_seed42_state_v2/run/modal_contract.json), [semantic tests](reports/goal_seed42_state_v2/run/modality_tests.json), [final acceptance](reports/goal_seed42_state_v2/run/final_acceptance.json)
- [200 episode records](reports/goal_seed42_state_v2/run/evaluation/rollouts.jsonl), [metrics](reports/goal_seed42_state_v2/run/metrics.json)
- [Data provenance](reports/goal_seed42_state_v2/run/data_manifest.json), [checkpoint provenance](reports/goal_seed42_state_v2/run/pretrained_manifest.json), [checkpoint hashes](reports/goal_seed42_state_v2/run/checkpoints/manifest.json)
- [Protocol audit](reports/goal_seed42_state_v2/run/protocol_audit.md), [retained v1 audit](reports/goal_seed42_state_v2/protocol_audit_v1_retained.md), [v2 execution document](reports/goal_seed42_state_v2/Runbook_v2.md)
- [Published file hashes](reports/goal_seed42_state_v2/publication_manifest.json)

Raw evidence preserves original server paths and bytes. The publication's `REPORT.md` changes only Markdown link targets; `run/report.md` retains the original report. The reproduction code is the tested snapshot. Model weights, demonstration HDF5 files, latent tensors and replay payloads are excluded. Existing upstream simulator assets belong to the upstream history and were not added by this publication.

## Reconstruct the exact execution checkout

The existing preflight intentionally checks that `HEAD` is the fixed upstream commit. Do not weaken this check to run directly from the publication commit. Reconstruct the original base-plus-patch layout in a **new directory**:

```bash
# Run from a clone of this MLR branch. Choose a new, unused data-disk directory.
publication_root="$PWD"
experiment_root=/data/mlr_ifa/experiments/goal_seed42_state_v2_reproduction
test ! -e "$experiment_root"
mkdir -p "$experiment_root/evidence"
git clone --no-hardlinks "$publication_root" "$experiment_root/source"
git -C "$experiment_root/source" checkout --detach a721807f2f7fa7f9833dcc85c592d78e9accd00b
cp "$publication_root/reports/goal_seed42_state_v2/run/source.patch" "$experiment_root/evidence/source.patch"
git -C "$experiment_root/source" apply --check "$experiment_root/evidence/source.patch"
git -C "$experiment_root/source" apply "$experiment_root/evidence/source.patch"
```

The parent `evidence/source.patch` location is required by the preserved driver. Do not replace it with a diff of publication-only reports or relabel old caches. Keep the existing completed experiment separate.

Use Python 3.8 and the recorded [environment](reports/goal_seed42_state_v2/run/environment.json) / [dependency versions](reports/goal_seed42_state_v2/run/requirements.lock.txt). The lock records the historical environment, including its original local paths. Configure actual dataset, BDDL, initial-state and CLIP cache locations before invoking the driver. The official checkpoint must hash to `241d3370acfc04d694df04256956a97aa13cfdc2e651cf258d338f8e72988be7`.

The completed server run used:

```bash
source /data/mlr_ifa/activate.sh
cd "$experiment_root/source"
args=(
  --pretrained /data/mlr_ifa/checkpoints/goal_multitask_model_ep96.pth
  --data-root /data/mlr_ifa/datasets
  --bddl-root /data/mlr_ifa/src/libero/libero/bddl_files
  --init-states-root /data/mlr_ifa/src/libero/libero/init_files
)
bash scripts/run_goal_seed42.sh --mode preflight "${args[@]}" --run-dir "$experiment_root/run"
bash scripts/run_goal_seed42.sh --mode smoke "${args[@]}" \
  --run-dir "$experiment_root/smoke_v2_release3" --smoke-dir "$experiment_root/smoke_v2_release3"
PYTHONPATH="$PWD:${PYTHONPATH:-}" python scripts/test_goal_buffer_and_updates.py \
  --run-dir "$experiment_root/smoke_v2_release3"
# Verify that both smoke test_gate.json and smoke/additional/result.json say PASS.
bash scripts/run_goal_seed42.sh --mode full "${args[@]}" \
  --run-dir "$experiment_root/run" --smoke-dir "$experiment_root/smoke_v2_release3"
```

These examples require the real authorized GPU/resources and are not executed by checking out this branch. The [historical release script](reports/goal_seed42_state_v2/evidence/release_pipeline3.sh) also merges additional test evidence into its test gate. `full` exports fresh features, trains the four stages, evaluates and summarizes. For an interrupted compatible run, inspect its checkpoint and process state before using `resume`; use `eval` only when training is complete. Never run a second process against the same output directory.

## Disclosed gaps

State is enabled according to the paper description. The official checkpoint's original `cfg.policy.use_state` is absent and remains **unknown**; existing state parameters and passing semantic tests do not establish their pretraining history. The original checkpoint is unchanged.

Training seed 42 comes from the official launch scripts, not a seed value enumerated in the paper. Complete six-task pretraining was not reimplemented. The 500-step warmup, optimizer-step scheduler repair, epoch99 checkpoint rule and evaluation RNG details are disclosed implementation decisions; this is a paper-aligned implementation with disclosed gaps, not a claim to recover the author's exact optimization trajectory.
