# LIBERO-OBJECT three-seed baseline reconciliation

Values are percentages. Standard deviations are sample standard deviations over seeds 42, 43, and 44.

## Per-seed metrics

| Method | Seed | FWT | NBT | AUC |
| --- | ---: | ---: | ---: | ---: |
| mlr | 42 | 76.250 | 11.111 | 70.729 |
| mlr | 43 | 77.500 | -7.222 | 81.562 |
| mlr | 44 | 78.750 | 4.444 | 76.562 |
| mlr_ifa | 42 | 78.750 | 12.500 | 71.875 |
| mlr_ifa | 43 | 77.500 | -3.056 | 79.271 |
| mlr_ifa | 44 | 78.750 | 11.111 | 73.438 |

## Three-seed summary and paper comparison

| Method | Metric | Ours mean ± sample std | Paper | Gap | Standardized descriptive gap |
| --- | --- | ---: | ---: | ---: | ---: |
| mlr | FWT | 77.500 ± 1.250 | 83.3 ± 2.6 | -5.800 | -2.231 |
| mlr | NBT | 2.778 ± 9.279 | 12.3 ± 2.4 | -9.522 | -3.968 |
| mlr | AUC | 76.284 ± 5.422 | 77.6 ± 3.0 | -1.316 | -0.439 |
| mlr_ifa | FWT | 78.333 ± 0.722 | 84.6 ± 1.9 | -6.267 | -3.298 |
| mlr_ifa | NBT | 6.852 ± 8.608 | 11.4 ± 5.6 | -4.548 | -0.812 |
| mlr_ifa | AUC | 74.861 ± 3.898 | 79.4 ± 1.5 | -4.539 | -3.026 |

## MLR+IFA minus MLR

| Seed | ΔFWT | ΔNBT | ΔAUC |
| ---: | ---: | ---: | ---: |
| 42 | 2.500 | 1.389 | 1.146 |
| 43 | 0.000 | 4.166 | -2.291 |
| 44 | 0.000 | 6.667 | -3.124 |
| mean | 0.833 | 4.074 | -1.423 |
