# LIBERO-OBJECT seed 42 results

This is the completed minimal single-seed reproduction using the official pretrained OBJECT checkpoint, tasks 6–9, 100 epochs per stage, and 20 evaluation trials per reported matrix entry.

| Method | FWT | NBT | AUC | Paper three-seed mean (FWT / NBT / AUC) |
| --- | ---: | ---: | ---: | --- |
| MLR | 76.250 | 11.111 | 70.729 | 83.3 / 12.3 / 77.6 |
| MLR + IFA | 78.750 | 12.500 | 71.875 | 84.6 / 11.4 / 79.4 |

Selected checkpoints (best current-task success, latest epoch on ties):

- MLR tasks 6/7/8/9: epochs 85 / 75 / 75 / 80
- MLR+IFA tasks 6/7/8/9: epochs 85 / 80 / 80 / 80

Success-rate matrices `r[i,j]` in percent:

```text
MLR
stage 6: [45,  -,   -,  -]
stage 7: [50, 90,   -,  -]
stage 8: [30, 50, 100,  -]
stage 9: [30, 90,  95, 70]

MLR + IFA
stage 6: [45,   -,   -,  -]
stage 7: [15, 100,   -,  -]
stage 8: [20,  85, 100,  -]
stage 9: [10, 100, 100, 70]
```

IFA improves FWT and AUC for this seed, while NBT is 1.389 points worse because task 6 is retained less well. The result remains within the paper's general scale but is not expected to equal a three-seed mean. No result-directed tuning or rollout filtering was performed.

