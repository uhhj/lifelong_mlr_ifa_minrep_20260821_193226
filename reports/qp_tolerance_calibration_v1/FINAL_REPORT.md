# QP tolerance calibration v1

## Scope
Offline only; `formal_training_resumed = false`. Formal solver/protocol/checkpoints were read only.

## F1
The real stored F1 dump (64x64) has original normalized KKT 1.5633119541946794e-8. The 10k continuation sweep leaves 1.0412605853937773e-8; 100k leaves 2.6401447000834466e-10. Thus the residual still decreases with budget; this is budget/path evidence, not a machine-precision claim. F1 would newly pass 2e-8 and 5e-8 but not strict 1e-8 on the original path.

## F2
The real traceback records KKT 1.0038755857838533e-8 at step 41530, but no matrix/solution dump exists. The only recorded state is the old recovery slot at step 40872; replay to 41530 was not silently substituted or completed in this offline run. Therefore F2 component decomposition, high-precision verification, and writeback comparison are unresolved.

## Decision
`REFERENCE_OR_DATA_INSUFFICIENT`. No tolerance candidate is supported. G1 is only partially auditable from source/traceback, G2 fails because F2 and an independent reference are missing, and G3 was not run because no real F2 writeback state was available. Preserve strict 1e-8 until F2 is captured and independently verified.
