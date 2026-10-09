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
