# Paired rollout failure diagnosis

Frozen source: `e65a64bfa3e7d7c829ff3ec840feeaa08f52a9f9`.
Branch: `codex/paired-rollout-failure-diagnosis`.
No training, new methods, optimization changes, or changes to original reports.

## Fixed scope

- seed43 stage7 task7; seed42 stage9 task7; seed43 stage9 task9.
- Each compares existing MLR, release-reference IFA and corrected-reference IFA.
- Fixed ep99;20 trials/cell;180 diagnostic trials plus20 instrumentation-control trials.
- GPU0 only, original official pretrained initialization followed by existing
  lifelong checkpoint, original test/evaluator/inference code.
-10 envs,2 rounds, initial-state indices0–19 (modulo file length), original env
  seeding, five settling steps,600 policy steps maximum, original GMM sampling,
  original success done accumulation and vector stopping rule. No trial omission.
- Each selected task is a separate test invocation with its run seed. This changes
  RNG call order versus the historical all-task matrix: new rates are diagnostic
  results, not replacements for historical formal rates. Initial states are paired;
  identical action noise for every cross-method trial is NOT claimed, because the
  native evaluator does not reset policy RNG at round boundaries.

## Passive recording and verification

Delegate the entire evaluation loop to unchanged `test.py` and `metric.py`. A
vector-env subclass only copies inputs/outputs and queries simulator state.
First run seed43/stage7/task7/MLR with light recording (actions/dones), then full
recording. Require exact equality of all actions, initial states, done flags and
step indices for all20 trials before continuing. No parity failure is ignored.

Full records: every action and done flag; all small numeric observations; simulator
state; agent-view images at step0 and every50 policy steps. Include settling steps
with nonpositive step indices. Preserve post-success native vector steps but truncate
individual diagnostic interpretation at first success. Raw arrays/images remain
under ignored `runs/`, never Git. Git receives small per-trial/paired reports only.

## Prespecified descriptive localization

Use task target object position, basket position and end-effector position from
observations. Record first successful step, min EEF-target distance, maximum target
height above settled start, minimum object-basket XY distance, and their time series.
Heuristic stages: target never rises3cm; rises3cm but never approaches basket XY
within10cm; reaches that neighborhood but no official success. These are geometric
descriptions, NOT verified grasp/contact/placement labels or causal explanations.
Thresholds fixed before outcomes, no threshold search. Report continuous values too.

Compare all paired trial outcomes. Inspect sparse images for discordant cases in
fixed increasing trial order, without excluding failures. If selected-task success
gaps do not persist under this evaluation invocation, explicitly report that and
do not assert the historical gap's cause has been located.
