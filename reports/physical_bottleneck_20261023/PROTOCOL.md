# Physical bottleneck v1 — preregistered 2026-10-23 after D0, before new fitting

Authorization: user requests same-unit diagnosis, then readout and spatial-patch controls, one factor at a time; engineering repair and reviewed MLR publication; only after A passes enter B. This new authorization supersedes the previous prohibition on further A variants, not the preserved original A failure. No old training queue restarts.

## D0: existing-checkpoint diagnosis

50 native checkpoints P0–P4 ×Task6/7×5fold, strictly selected in original A. Eval train/validation/test with the SAME 3D Euclidean mm, aperture mm, per-axis and demo/dynamic strata. Check original heldout predictions within1e-5 m. GPU forwards count separately, no optimization/environment/server writes. Average per-fold summaries only; overlapping train folds are not independent samples. D0 is complete before D1 launch; test outcomes used only to explain original failure, not to select new hyperparameters.

## Shared new-A conditions

OBJECT seed41, Task6/7, original official frozen frontend and original recorded-state labels/window population/state masks/demo splits, exact native preprocessing. No new demos, additional seeds, raw image augmentation, policy updates or environments. Original ten-demo outer folds already inspected in v1: this is explicitly exploratory follow-up, not a new untouched confirmatory holdout. Fixed prospective model list and gates, no open-ended trial until success. All original A evidence remains unchanged.

Each fresh fit uses original seed41+task*100+fold, batch128, FP32, AdamW3e-4 wd.01, same7D SmoothL1 scaling[.1m]x6+[.08m], original validation mean position rule, tie earliest. All fits run200epochs with no test-based stopping. For P0 and M record the validation-selected checkpoints within budgets50/100/200 from the SAME uninterrupted trajectory: only budget differs. Q uses200. Test each selected budget checkpoint once with all three fixed interventions. Differences to original50 results are checked, not hidden. Every diagnostic optimizer/forward/parameter count separately reported; not equal total compute to original A or B.

## D1: readout, then training-budget contrast

M is EXACT original P3, unchanged trainable parameters and initialization. P0 is EXACT original strong language/proprio history baseline. Q uses the same P3 projection, frame encoder, time encoder and head; the sole change is frame fusion: uniform modality mean becomes learned softmax attention using a single128D query initialized to zero. Extra128 active parameters disclosed; at initialization Q equals M. No temporal/history-length change in this round, keeping interpretation one factor. M50 vs100 vs200 diagnoses budget; M200 vsQ200 diagnoses fusion. Original frozen Temporal is bypassed in M and Q. No claim that P2 vsP3 isolated Temporal capacity.

Fits: P0/M/Q ×2tasks×5fold=30 fresh fits, each200epochs. Save all models and train/val/test same-unit metrics; use candidate M200/Q200 only for the prospective A gate, P0-200 comparator. Scalar errors and metadata only published.

## D2: spatial granularity with a matched bridge control

After D1 fitting completes, export frozen CLIP final patch grid using EXACT original RGB preprocess, weights (including loaded adaptation weights), camera order, and labels. Apply CLIP post-layernorm to patches, then official image_proj and language FiLM independently per patch. No visual fine-tuning. CLS/pooler path reconstructed through the same projection+FiLM must agree with existing native camera tokens within atol1e-4/rtol1e-5. Export shape and all SHA; all payload remains private. Same physical frames as original native export, no rollout.

G and S use identical new SpatialReadout architecture, initialization, optimizer, budget and sequence length. Each camera receives196tokens with fixed2D positions; G repeats its actual native global camera token across the grid, S supplies the actual spatial patch tokens. Same LN512→128, position projection, one learned128 query to pool each camera, Linear128→512, followed by EXACT P3 mean-fusion architecture. Language/proprio remain identical. The ONLY G/S difference is global-repeated vs patch token values. Query weights in G may have reduced useful degrees of freedom because contents repeat; total/active gradient counts reported, no promise of equal effective capacity. M/G is a bridge illustrating the added adapter, not attributed to spatial information. G/S is the spatial contrast. Fixed physical batch128; both G/S microbatch16 with loss weighted by microbatch/sample count before one AdamW step, dropout0, full FP32. Do not compare raw numerical trajectory with M as bit-identical. No precision or scientific batch reduction.

G/S each2tasks×5fold=20 fits×200epochs. No model variant chosen using test between D1 and D2; both phases always complete. Thus50fits/10000probeepochs; exact optimizer count from original masks and ceil(Ntrain/128), not guessed.

## Admission and scientific decision

Existing visibility gate must be valid. New Q/G/S separately check newest visual input sensitivity, causal prefix handling and strict checkpoint roundtrip; isolate smoke optimization from formal data fits and count it separately. Runtime/disk/hash/finite checks mandatory. No posthoc tolerance widening.

Native heldout A gate unchanged: both tasks×both position targets pooled median≤10mm/P90≤25mm; paired demo-bootstrap10000/seed410023 mean per-demo-median improvement over P0-200 CI95 lower>0; on dynamic samples mean per-demo(previous-target minus current-target error)>0 and latest-visual→previous-frame replacement increases current error>0. Fixed other-demo visual shuffle reported. For patches visual interventions act on both current global tokens and corresponding patch sequences so no second visual path leaks true current observations. P0 never consumes visual data. All failures/negative targets disclosed.

Predeclared candidate priority M200→Q200→G200→S200; first fully passing is selected AFTER ALL50fits and independent recomputation. This is a finite exploratory architecture screen, no population-significance claim. If none pass STOP_STATE_GATE; no B, no retries of science or weakened gates. If passed, implementation/admission of B proceeds autonomously in a separate frozen version with selected architecture, fresh weights, original legal current+RA populations and no A-fit weights. B itself must pass original effect validation gate before policy/300episodes. B code must exist and be audited, no placeholder launch. User-confirmed expansion: B2 final two-task mean>both controls and oldTask6≥both; then all three arms/interventions to42/43. No substitute RA-AUC gate.

## Engineering, capacity and publication

New isolated root PHYSICAL_BOTTLENECK_OBJECT41_20261023_v1, original assets read-only. Old frozen source/contracts never overwritten. Two-phase controller executes D1→export→spatial admission→D2→A audit; each phase's original exit/log retained. Each fit saves complete epoch model/optimizer/best model/epoch/cursor/allRNG/modes/requires_grad/grads; recovery only after confirmed stopped/hash-consistent failure, explicit versioned recovery from whole epoch (no automatic full restart). Test predictions immutable; scientific failed outcomes never retried. Storage: native labels reused read-only, new patch float32~2.45GB plus checkpoints/recovery and metadata planned≤3.3GiB, retain /data root2GiB and system ordinary1GiB. Stop on actual capacity bounds, do not delete old assets/pay/alterprecision. GPU single-controller lock and actual GPU-idle check; source/input SHA checks and no duplicate startup.

Every round publish exact reviewed code/report/hash/scalar error evidence via publish_mlr_branch, destination branchMLR; no rawstates/actions/prediction vectors/latents/patchcache/checkpoints/credentials. Ongoing status PARTIAL, local independent final audit before acceptance. Remote full queue may run after desktop closes; local repair/audit/publication requires active desktop executor and cannot be guaranteed while powered off.
