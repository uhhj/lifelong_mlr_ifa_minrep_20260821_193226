# V1-OBJECT-TaskSelectiveReplay

Seed42 only. Original empty replay and 50% window caches retained by user-approved baseline adaptation; historical source counts 0/1/2/3.
Task6 and task7 retain native sampling; nontrivial source routing starts at task8. BC, IFA, batch slots and 105500 updates unchanged.

|Method|AUC|FWT|NBT|Final all-10 SR|
|---|---:|---:|---:|---:|
|Historical IFA|77.708333333|77.500000000|-0.277777778|not recorded|
|Paired baseline reevaluation|78.333333333|78.750000000|0.833333333|35.0|
|V1|72.291666667|77.500000000|12.500000000|30.0|

Screen: SCREEN_NOT_MET
Historical comparison uses the unrounded original four-task metric. Final all-10 uses paired evaluation; no baseline training was rerun.
See PAIRED_TRIALS.csv, SUMMARY.json, ROUTING_SUMMARY.json, per-epoch source counts, native IFA diagnostics, and EXECUTION_COST.jsonl.
Routing changes IFA exposure as well as BC exposure. Signed gradient similarity is not proof of positive transfer or shared knowledge.
Single-seed screening is not statistical confirmation. No temperature/coverage search, no additional seeds, no old queues resumed.

|Final task|V1|Paired baseline|Delta|
|---|---:|---:|---:|
|0|45|55|-10|
|1|0|0|+0|
|2|0|0|+0|
|3|0|0|+0|
|4|5|0|+5|
|5|0|0|+0|
|6|0|20|-20|
|7|90|95|-5|
|8|65|100|-35|
|9|95|80|+15|
