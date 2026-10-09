# E1-OBJECT-v1

Historical MLR+IFA reused; no baseline or pretraining rerun.

|Seed|E1 AUC|IFA AUC|Delta AUC|E1 NBT|E1 FWT|
|---|---:|---:|---:|---:|---:|
|42|70.312000|77.708000|-7.396000|4.444000|72.500000|

## Mechanism and measured cost

|Seed|Checks|Triggers|Accepted|Capped|E1 overhead (s)|Peak GPU (GiB)|
|---|---:|---:|---:|---:|---:|---:|
|42|8980|4324|4320|0|340.56|0.835|

Per-stage rates and skip reasons: MECHANISM_seed*.json. Training/evaluation invocation wall times, including interrupted invocations: EXECUTION_COST.jsonl. No baseline speed ratio is asserted without matched timing evidence. Per-trial outcomes and initial-state IDs: seed*/trials_*.jsonl. Secondary selected-checkpoint metrics are in RESULT_seed*.json.

Primary gate: NO_OVERALL_BENEFIT

The single-seed gate determines additional E1 investment, not proof of stable improvement. All seeds share a fixed pretrained initialization. Secondary selected-checkpoint evaluation is retained separately and never used for the gate.

## Completion audit

Completed at 2026-10-09 01:33:05 Asia/Shanghai. Independently verified all 400 stage/epoch rows, 200 primary trials (10 cells), and 680 secondary trials (34 cells), including trial/initial-state identities and reconstruction of both metric matrices directly from trial successes. FINAL_VERIFICATION.json: 12/12 checks PASS. No E1 processes remain; seeds43/44 were not started.

Primary same-seed historical comparison: AUC 70.312 vs 77.708 (delta -7.396 percentage points), FWT 72.500 vs 77.500 (delta -5.000), NBT 4.444 vs -0.278 (delta +4.722; lower is better). The preregistered overall-benefit condition failed. The current E1 configuration is not supported by this single-seed screen; no additional E1 seeds are authorized by its conditional rule.

Secondary selected-checkpoint results: AUC 70.000, FWT 78.750, NBT 17.222. Selected epochs for tasks6/7/8/9: 70/95/99/85. These diagnostics do not replace the primary endpoint or reverse the gate.

Measured training invocation: 3240.11 seconds (54.00 minutes); successful primary evaluation: 4877.84 seconds (81.30 minutes); successful secondary evaluation: 15703.72 seconds (261.73 minutes). Cumulative internal E1 check/correction overhead: 340.56 seconds, about 10.51% of measured training invocation time, not a baseline-relative slowdown. Peak training allocated GPU memory: 0.835 GiB. All invocation costs, including failed bootstrap and the completed-training no-op resume, remain in EXECUTION_COST.jsonl.

The 4320 accepted corrections show reliable decrease on their sampled old-buffer check batch under the frozen acceptance rule; they did not translate into improved overall closed-loop AUC. This is one seed with a reused historical baseline, not evidence that every related correction method must fail. Task6 had an empty old buffer and no E1 checks, yet its evaluated success differed from the historical baseline (30% vs 35%); do not attribute that first-stage difference to an E1 correction or claim a bitwise paired full-run counterfactual. No baseline rerun or post-hoc retuning was performed.

The evaluation spawn-import failure was fixed by an additional bootstrap entrypoint only; frozen original implementation hashes remain unchanged. The failure evidence and repair provenance are retained. Historical experiments, protocols, and checkpoints were not overwritten.
