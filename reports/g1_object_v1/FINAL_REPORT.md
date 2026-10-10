# G1-OBJECT-v1

Original mixed-batch BC and native weighted IFA, gated by decoder gradient dot sign. E1/P1 disabled.

|Seed|G1 AUC|Historical IFA AUC|Delta AUC|FWT|NBT|
|---|---:|---:|---:|---:|---:|
|42|74.166666667|77.708333333|-3.541666667|76.250000000|3.888888889|

Seed42 gate: NO_OVERALL_BENEFIT
Exact rational AUC gate from the raw trial matrix; rounded display never controls continuation.
Each seed: 400 epochs, four own ep99 checkpoints, 200 primary trials. No auxiliary evaluation or baseline rerun.
Task6 has no G1 intervention: any historical difference at task6 is not attributable to gating.
See MECHANISM_seed*.json for denominators, closure and own-trajectory IFA retention; this is not a baseline counterfactual.
EXECUTION_COST.jsonl and per-seed training_cost record wall time. Probe/dot timings synchronize CUDA. No unsupported baseline speedup claim.
A single-seed positive gate is an investment rule, not statistical significance. This does not prove shared knowledge or per-step forgetting prevention.
