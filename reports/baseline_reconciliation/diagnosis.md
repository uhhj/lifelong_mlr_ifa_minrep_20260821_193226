# Success-matrix diagnosis

All values are percentage points from the formal 20-trial evaluations. The six triangular matrices are in `matrices/`; aggregate values come from `object_3seed_metrics.json`.

## Finding

The original seed42 deficit is not one phenomenon. Its low AUC is partly seed-specific, but the low FWT is a stable, shared task6 effect. The remaining IFA AUC delta is a seed-dependent forgetting effect. The evidence therefore supports **D. mixed / unresolved**, not a pipeline failure and not a simple “bad seed42” explanation.

## Required questions

1. **Is seed42 clearly the lowest seed?** For MLR FWT it is the lowest (76.25 versus 77.50 and 78.75), but only 2.5 points below seed44; all three are below the paper mean. For MLR AUC, seed42 is clearly the lowest (70.729 versus 81.562 and 76.562). For MLR+IFA, seed42 is not the lowest FWT (78.75, tied with seed44) but is the lowest AUC (71.875 versus 79.271 and 73.438).
2. **Do seeds43/44 approach the paper?** Their MLR AUC values bracket the paper mean of 77.6, and seed43 MLR+IFA AUC=79.271 is close to 79.4. Their FWT values remain only 77.5–78.75, so additional seeds reconcile the MLR AUC gap but not the shared FWT gap.
3. **Are the same tasks low in both methods?** Yes. Task6 diagonal success is identical between MLR and MLR+IFA for each seed because no old-task IFA term exists at the first lifelong stage: 45, 20, and 55. Task7 and task8 are generally high, while task9 is 70/90/70 in both methods.
4. **Which tasks cause low FWT?** Task6 is dominant. Across seeds, mean task FWT for MLR is task6=40.0, task7=93.333, task8=100.0, task9=76.667. MLR+IFA is task6=40.0, task7=96.667, task8=100.0, task9=76.667. The overall FWT gap is therefore not broad environment failure.
5. **Why is AUC low?** Both low new-task learning and forgetting contribute, but at different places. Task6 starts low and has the lowest trajectory AUC (MLR 43.750; IFA 38.333). Seed42 additionally loses task7 at stage8. Seed44 IFA loses task7 from 75 at stage8 to 15 at stage9. Seed43 MLR instead exhibits positive backward transfer on task6 and reaches AUC=81.562.
6. **Does task6 cause seed42's slightly worse IFA NBT?** Yes. Seed42 task6 forgetting rises from 8.333 (MLR) to 30.0 (IFA), a +21.667-point deterioration. Improvements on task7 and task8 partly offset it, leaving aggregate Delta NBT=+1.389.
7. **Do selected IFA pairs map to a stable performance change?** No. For the task7 -> task6 pair, the stage7 task6 difference (IFA minus MLR) is -35, -30, and +5 for seeds42/43/44. Seed42's task8 -> task7 pair corresponds to +35 at stage8. Seeds43/44 selected no task8 pair; their stage8 task7 differences are 0 and +5 and reflect inherited trajectory differences, not a new task8 hinge term.
8. **What happens at task9 with no IFA pair?** New-task task9 FWT is exactly unchanged for every seed (Delta=0). Old-task behavior can still differ because the methods arrive at task9 with different parameters: the largest example is seed44 task7, 80 for MLR versus 15 for IFA after stage9.

## Attribution

- **Shared component:** three-seed FWT is tightly clustered but low for both methods, and task6 is identical across methods. The official-release scheduler is stepped once per epoch while its total-step scale is expressed in batches, producing a long warmup; stochastic GMM evaluation is also preserved. These release behaviors are plausible paper-versus-release contributors, not proven errors.
- **Seed component:** MLR AUC ranges from 70.729 to 81.562 and its three-seed mean is only 1.316 points below the paper mean.
- **IFA component:** IFA pair effects are not stable across seeds, and IFA mean AUC is 1.423 points below the local MLR mean despite +0.833 FWT. The implementation and selected-pair logging match the release, so the observed delta is behavioral rather than evidence of a coding mismatch.
- **Environment component:** all pinned software versions match the release. GPU/driver details were not published, so their numerical effect remains unknown but unsupported by a concrete incompatibility.

## Classification

**D. mixed / unresolved.** Seed42 explains part of the original AUC concern; a shared task6/release-protocol effect explains the stable FWT gap; and seed-dependent IFA forgetting explains the remaining method-specific AUC delta. The local baseline is internally reproducible and suitable for controlled follow-up comparisons, but the paper/local numerical gap should remain documented.
