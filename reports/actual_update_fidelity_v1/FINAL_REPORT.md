# Actual AdamW update fidelity

Status: COMPLETE; starts=120; formal_training_resumed=false.

Classification: `FIRST_ORDER_ACTUAL_UPDATE_SUPPORTED`. Sample-QP local mechanism: `SUPPORTED`. AdamW-over-raw: `SUPPORTED`.

All thresholds are preregistered engineering interpretation, not proof of protection.

|alpha|Spearman median|sign accuracy|harm recall|harm precision|decisive coverage|all strata reliable|
|---:|---:|---:|---:|---:|---:|---|
|1.0|0.9843478260869564|0.9583333333333334|0.9|1.0|1.0|True|
|0.5|0.994782608695652|1.0|1.0|1.0|1.0|True|
|0.25|0.9978260869565216|1.0|1.0|1.0|1.0|True|
|0.125|0.9991304347826085|1.0|1.0|1.0|1.0|True|
|0.0625|0.9991304347826085|1.0|1.0|1.0|1.0|True|
|0.03125|0.9999999999999999|1.0|1.0|1.0|1.0|True|

1. Normal-scale prediction: see alpha1 and all six stratum gates above.
2. Reliable tested scales: [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]; no interpolation to untested alpha.
3. Paired actual-versus-raw conclusion: SUPPORTED; see same-target proxy separately.
4. Ranking, sign and magnitude errors are separately quantified in START_LEVEL_METRICS (Spearman, sign, slope/intercept/relative MAE).
5. Harm recall/precision include neutral errors in their respective denominators; undefined cases are not silently scored as correct.
6. Seed and transition consistency: ALPHA_SWEEP_REPORT contains every stratum; no pooled-window vote.
7. Sample-QP mechanism: SUPPORTED, only for the measured native directions. This never proves non-increasing finite loss for all QP-corrected displacements.
8. If only reduced steps pass, prioritize a measured trust region/line search; if none pass, audit numerical fidelity and use actual-loss probes before stronger protection claims.
9. True NLL HVP conditional trigger=False; completion=False. No Fisher substitution.
10. Formal lifelong training remains stopped; no automatic restart.

## Scope limitations
- development seeds
- epochs0-29 unavailable; only30-99; no early-phase inference
- task8->9 full distributed origins unavailable; use complete6->7 and7->8
- historical model-only checkpoints excluded
- origins on common carrier remain serially dependent; starts are not independent windows
