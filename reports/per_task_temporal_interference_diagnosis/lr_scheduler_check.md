# LR / scheduler note

The full trace is recorded in `reports/migration_followup/lr_scheduler_check.md`.

Result: **B**. AdamW starts at `1e-4`, `LambdaLR` construction sets the active LR to exactly `0`, and subsequent scheduler steps produce `2e-7`, `4e-7`, `2e-6`, and `2e-5` at steps 1, 2, 10, and 100. The release advances this 500-step warmup once per epoch. LR is therefore not permanently stuck at zero, and the offline diagnosis is unaffected because it performs no optimizer step.
