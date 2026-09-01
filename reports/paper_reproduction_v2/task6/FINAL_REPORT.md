# Task6 paper-protocol reconciliation

Verdict: `PAPER_TASK6_PROTOCOL_RECONCILIATION_SUPPORTED`

All arms used LIBERO-OBJECT task6, seed42, the official pretrained checkpoint, 100 epochs, 20 evaluation trials per checkpoint, and isolated run/buffer directories.

| Arm | Batch | Scheduler unit | LR e0 | LR e10 | LR e50 | LR e99 | Task6 selected success |
|---|---:|---|---:|---:|---:|---:|---:|
| current legacy | 10 | epoch | 0 | 1.8e-6 | 9.8e-6 | 1.96e-5 | 40% (ep90) |
| release18 legacy | 18 | epoch | 0 | 1.8e-6 | 9.8e-6 | 1.96e-5 | 50% (ep95) |
| batch10 linear step | 10 | optimizer update | 0 | 9.30e-5 | 5.16e-5 | 1.03e-6 | 60% (ep90) |
| batch10 constant | 10 | none | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 85% (ep70) |

The official pretrained model scored 0% zero-shot on task6. The unit-consistent batch-step arm improved formal selected success by 20 percentage points over the current legacy arm and reduced late mean BC loss from -24.960 to -31.837. Its LR trace reaches 1e-4 after 500 optimizer updates and then follows the unchanged release linear decay. This supports `SCHEDULER_UNIT_MISMATCH_MATERIAL`.

The batch18 legacy difference was 10 percentage points, below the 15-point materiality threshold. Constant LR strongly confirms effective LR starvation, but it is not a paper claim and is not promoted. The source-supported full-MLR candidate is batch10 with the release linear schedule stepped once per optimizer update.
