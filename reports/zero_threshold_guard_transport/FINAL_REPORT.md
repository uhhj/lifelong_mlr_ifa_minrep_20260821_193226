# Zero-Threshold Candidate Directional Guard Transport Diagnosis

Verdict: `ZERO_THRESHOLD_DIRECTIONAL_GUARD_TRANSPORT_SUPPORTED`
Flags: `NONE`
Branch: `research/zero-threshold-directional-guard-transport`
Starting SHA: `d5d8759152563021483f570683aeaf7046a21201`
Analysis code SHA at report generation: `f725a66ecc21e34a4c4c709bdf9a50833aaf959c`
Final SHA: branch HEAD reported in the final handoff after committing this report.

## Protocol and threshold provenance

Temporal V1; seeds 42/43/44; stages 8/9; previous-stage ep95 entry proxy; 10 common warmup epochs; exact paired Control/Guard restore; 300 intervention updates per arm; H100 primary and H300 secondary; guard scale 0.5; old score probe 64/task with a fresh gradient every step; old label probe 256/task; current probe 256.

Threshold mode is `zero` and threshold is exactly `0.0`. No future-label-derived threshold artifact was read for guard decision.
Branch start parity: `PASS` (maximum absolute probe difference `0`). All primary metrics finite: `True`.

## H100 paired transport

| Seed | Stage | Fire rate | Control old drift | Guard old drift | Rel. old benefit | Rel. current harm |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 8 | 0.31 | -0.16999425 | -0.18690309 | 0.00066322937 | 0.00044367974 |
| 42 | 9 | 0.3 | -0.1659282 | -0.17017749 | 0.00017005063 | 0.00083250862 |
| 43 | 8 | 0.59 | 0.021075247 | -0.043222648 | 0.0025255274 | 0.0022295276 |
| 43 | 9 | 0.51 | -0.029329636 | -0.073105718 | 0.0017376211 | 0.0014358003 |
| 44 | 8 | 0.37 | -0.12828918 | -0.15099191 | 0.000908172 | 0.0008191933 |
| 44 | 9 | 0.58 | 0.17923115 | 0.10302425 | 0.0030593048 | 0.0027692329 |

Positive pairs: `6/6`; mean/median relative old benefit: `0.0015106509` / `0.0013228965`; mean/max relative current harm: `0.0014216571` / `0.0027692329`; pooled fire rate: `0.44333333`.

## Per-seed transport

| Seed | Positive pairs | Mean H100 rel. old benefit | Mean rel. current harm | Fire rate |
|---:|---:|---:|---:|---:|
| 42 | 2/2 | 0.00041664 | 0.00063809418 | 0.305 |
| 43 | 2/2 | 0.0021315743 | 0.0018326639 | 0.55 |
| 44 | 2/2 | 0.0019837384 | 0.0017942131 | 0.475 |

## H300 persistence

| Seed | Stage | Fire rate | Control old drift | Guard old drift | Rel. old benefit | Rel. current harm |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 8 | 0.48666667 | -0.21552697 | -0.20872617 | -0.00026675344 | 0.0041343816 |
| 42 | 9 | 0.43 | -0.19604327 | -0.20639105 | 0.00041410423 | 0.0045694158 |
| 43 | 8 | 0.51666667 | -0.20391117 | -0.25386105 | 0.0019619583 | 0.0049012063 |
| 43 | 9 | 0.43666667 | -0.13105575 | -0.16706786 | 0.0014294429 | 0.0032216917 |
| 44 | 8 | 0.47333333 | -0.2226889 | -0.20932071 | -0.00053476432 | 0.003603919 |
| 44 | 9 | 0.57333333 | 0.097466361 | 0.054807922 | 0.0017125113 | 0.0061721688 |

Positive pairs: `4/6`; mean relative old benefit: `0.00078608317`; mean relative current harm: `0.0044337972`; pooled fire rate: `0.48611111`.

## Mechanism, runtime, and per-task checks

Candidate cosine mean/median/q25/q75: `-0.0020438455` / `-0.0016944916` / `-0.011612626` / `0.0079735305`. Fired cosine mean/median/q25/q75: `0.010329535` / `0.0087296903` / `0.0041801227` / `0.014433004`.
Summed predicted first-order score reduction: `2.0341865`. Control/Guard intervention wall time: `142.23156` / `142.69362` seconds; mean sec/update: `0.079017535` / `0.079274232`.
Task-specific retention tradeoff: `NONE`; worst relative task harm: `0.00021017275`.

## Scientific conclusion

H16 is supported under the preregistered criteria. The label-free candidate-cosine > 0 boundary preserved paired causal retention benefit across all seeds, including seed44, without material H100 current-probe harm and without mean H300 reversal. Threshold calibration is closed for this mechanism. Next run Guard Gradient Refresh Feasibility Diagnosis, comparing fresh gradients with refresh every 10 updates; do not reduce probe size or start formal guard training.

## Not established

This remains a short paired BC-loss intervention diagnosis, not formal lifelong rollout evidence. Tau=0 removes future-label threshold calibration, but fresh old-task gradient estimation remains expensive. The fixed score probes may not perfectly represent rollout retention. The intervention full-commits AdamW moments while scaling only the parameter endpoint. H300 is a medium-horizon BC-loss probe, not proof of long-term continual-learning benefit. Ep95 remains a late-stage proxy.

## Exact successful commands

```bash
python -m pytest -q tests/test_zero_threshold_guard_transport.py tests/test_candidate_directional_guard.py
CUDA_VISIBLE_DEVICES=1 bash scripts/run_zero_threshold_guard_transport.sh smoke
CUDA_VISIBLE_DEVICES=1 bash scripts/run_zero_threshold_guard_transport.sh formal
```
