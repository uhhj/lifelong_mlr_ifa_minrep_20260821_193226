# QP writeback representation forensic

Two immutable source states, zero preceding batches. Formal solver and original experiment remain unchanged.

|start|stage|fixed native normalized max|own normalized max|own-gate violations|
|---|---|---:|---:|---:|
|s45_t7_e98_b195|d64|7.14758422096e-12|7.1486962098e-12|0|
|s45_t7_e98_b195|d32|2.53992442423e-11|2.54031957419e-11|0|
|s45_t7_e98_b195|dscatter|2.53992442423e-11|2.54031957419e-11|0|
|s45_t7_e98_b195|dwritten|5.57503590622e-06|5.57588263106e-06|1|
|s45_t7_e98_b195|dwritten_float32_report|5.57503590209e-06|5.57588262693e-06|1|

s45_t7_e98_b195: first fixed-denominator violation=dwritten; first actual-norm gate violation=dwritten; solver KKT=7.147582872790714e-12; historical failure reproduced.
|s45_t8_e99_b94|d64|8.41503534274e-12|8.41751928809e-12|0|
|s45_t8_e99_b94|d32|2.17275143839e-11|2.1733927899e-11|0|
|s45_t8_e99_b94|dscatter|2.17275143839e-11|2.1733927899e-11|0|
|s45_t8_e99_b94|dwritten|8.7703276611e-06|8.7727747203e-06|9|
|s45_t8_e99_b94|dwritten_float32_report|8.77032765823e-06|8.77277471742e-06|9|

s45_t8_e99_b94: first fixed-denominator violation=dwritten; first actual-norm gate violation=dwritten; solver KKT=8.41504470272536e-12; historical failure reproduced.

dscatter is a layout-only reconstruction using actual parameter slice boundaries. It is not an alternative writeback implementation. dwritten is the exact difference of stored float32 parameters; the additional report stage isolates subtraction rounding.
Both normalization schemes are shown to avoid attributing a denominator change to rounding. Raw tiny positive inner products are reported separately from tolerance failures.
No threshold modification, repair, extra fallback, phase2, or formal training was executed.

## Independently verified root cause

- s45_t7_e98_b195: float32 parameter addition/subtraction rounding error norm 8.05709785061e-07, 0.0594579% of native displacement and 3.37112% of ideal correction. 80.148% of nonzero correction coordinates do not change the stored native parameter; this is a coordinate count, not energy loss.
- s45_t8_e99_b94: float32 parameter addition/subtraction rounding error norm 7.93161985515e-07, 0.0955737% of native displacement and 3.93438% of ideal correction. 83.441% of nonzero correction coordinates do not change the stored native parameter; this is a coordinate count, not energy loss.

Both QPs and flat float32 corrected displacements pass. The layout reconstruction is exact. Rounding theta_native-c32 into the float32 parameter lattice introduces the first above-gate residual. Computing the final parameter difference in binary64 still shows the same failure, ruling out a float32 subtraction-report artifact. Identical fixed-native normalization rules out denominator effects. Errors are individually within ordinary float32 rounding scale, but ideal active constraints have near-zero slack. This diagnosis does not relax thresholds, certify finite NLL protection, or authorize a writeback repair.
