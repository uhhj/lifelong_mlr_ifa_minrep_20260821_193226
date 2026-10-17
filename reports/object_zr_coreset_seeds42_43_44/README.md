# Coreset + IFA + zero-response residual: seeds42–44

Completed LIBERO-OBJECT experiment; 1,200 epochs, 316,500 optimizer steps, 12 strictly loaded final checkpoints, 600 formal episodes and 132 structural checks.

[Final report and all task matrices](execution/20261017_zr_coreset_extension/FINAL_REPORT.md) · [Exact results](execution/20261017_zr_coreset_extension/FINAL_RESULTS.json) · [Independent acceptance](execution/20261017_zr_coreset_extension/LOCAL_FINAL_ACCEPTANCE.json)

The primary comparison uses the second full seed43 baseline run at the user's request after results were observed. The first baseline remains a sensitivity comparison. Primary mean AUC difference: +2.7431 percentage points; first-seed43 sensitivity: −2.5000 points. This is an exploratory, hardware-confounded comparison; structural isolation does not guarantee retention by the shared network.

Recompute from this repository root:

```sh
python -X utf8 reports/object_zr_coreset_seeds42_43_44/execution/20261017_zr_coreset_extension/final_local_audit.py
```

This command reads recorded evidence only. It imports no ML framework, loads no model and executes no training or environment. The small seed42 acceptance archive contains only text evidence. No checkpoints, demonstrations, latent tensors, buffer payloads or credentials are included. Seed40/41 preparation and any future results are outside this publication.

`SOURCE_SHA256.json` records the copied lightweight evidence. Input paths in archived reports identify the original execution host; local recomputation resolves evidence relative to the script. Original failed seed42 exit evidence is retained alongside its successful recovery receipts. Structural/metric software checks are not counted as experimental episodes.
