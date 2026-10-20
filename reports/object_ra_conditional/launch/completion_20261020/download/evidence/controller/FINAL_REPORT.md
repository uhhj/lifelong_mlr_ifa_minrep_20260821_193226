# Conditional RA / ZRv2 study — server final report

All authorized executed branches passed engineering acceptance. See FINAL_SERVER_ACCEPTANCE.json.

## stage1

|Seed|Method|AUC %|FWT %|NBT pp|OldFinal %|FinalAll %|
|---|---|---:|---:|---:|---:|---:|
|41|random|66.87500000|63.75000000|-4.16666667|58.33333333|61.25000000|
|41|coreset|70.10416667|67.50000000|-5.55555556|71.66666667|66.25000000|
|41|ra_coreset|73.02083333|75.00000000|4.16666667|78.33333333|70.00000000|
|42|random|66.14583333|66.25000000|-0.55555556|70.00000000|63.75000000|
|42|coreset|68.12500000|65.00000000|-5.83333333|73.33333333|66.25000000|
|42|ra_coreset|71.56250000|70.00000000|-3.05555556|65.00000000|63.75000000|
|43|random|71.87500000|70.00000000|-3.61111111|75.00000000|71.25000000|
|43|coreset|78.64583333|70.00000000|-16.94444444|86.66666667|78.75000000|
|43|ra_coreset|75.10416667|75.00000000|-1.38888889|81.66666667|75.00000000|

All task matrices, exact fractions and sample SD (ddof=1): EXPERIMENT_RESULTS.json.
Gate: {"passed": true, "mean_auc_delta_fraction": "71/1440", "mean_auc_delta_pp": 4.930555555555555, "positive_seeds": 3, "all_engineering_pass": true}
## stage2

|Seed|Method|AUC %|FWT %|NBT pp|OldFinal %|FinalAll %|
|---|---|---:|---:|---:|---:|---:|
|41|parent|73.02083333|75.00000000|4.16666667|78.33333333|70.00000000|
|41|new|68.43750000|66.25000000|-4.72222222|73.33333333|66.25000000|
|42|parent|71.56250000|70.00000000|-3.05555556|65.00000000|63.75000000|
|42|new|71.04166667|70.00000000|-3.05555556|63.33333333|62.50000000|
|43|parent|75.10416667|75.00000000|-1.38888889|81.66666667|75.00000000|
|43|new|76.66666667|73.75000000|-5.83333333|81.66666667|76.25000000|

All task matrices, exact fractions and sample SD (ddof=1): EXPERIMENT_RESULTS.json.
Gate: {"passed": false, "mean_auc_delta_fraction": "-17/1440", "mean_auc_delta_pp": -1.1805555555555556, "positive_seeds": 1, "all_engineering_pass": true}
Stop reason: STAGE2_METRIC_GATE_NOT_MET

Exploratory conditional sequence; historical hardware differs and episode action RNG is not isolated. Pretrain state metadata remains unknown. IFA pair choices may change indirectly with memory. ZRv2 incurs additional calibration and a fixed T0 inference branch. Task6 cross-hardware tensor differences are disclosed under the explicit user amendment; initialization and protocol checks remain mandatory. No significance or universal causal claim.
