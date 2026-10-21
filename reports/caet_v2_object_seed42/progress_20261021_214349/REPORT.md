# CAET-v2 OBJECT seed42: PARTIAL, predictor and mechanism audit

Full source/input snapshot: 2026-10-21T21:44:24.447451+08:00. Later read-only mechanism ledger: 2026-10-21T21:45:28.418907+08:00. These are sequential observations, not an atomic snapshot.

Main training has reached 394/400 complete epochs and 103328/105500 optimizer updates (Task9 94/100). Formal evaluation was 0/200 at the full snapshot. All 518 frozen source/input hashes pass; active recovery supervisor, pipeline and Task9 trainer are normal. Original failed import attempt remains preserved; no new recovery, training, test updates or rollouts were launched by this check.

| Check | Task8 (100 epochs) | Task9 (94 epochs) |
|---|---:|---:|
| Whole-demo heldout admission | PASS | PASS |
| Normalized heldout MAE | 0.269519 | 0.254410 |
| Training-mean baseline MAE | 0.479750 | 0.457105 |
| Fixed action-permutation MAE | 0.296241 | 0.288566 |
| Supported heldout times | 73 | 64 |
| Old task-demo support identities | 11 | 17 |
| Effect fitting updates (10 epochs) | 840 | 980 |
| Main updates observed | 30900 | 34028 |
| Transfer accepted updates | 7065 | 8372 |
| Transfer rejected for negative BC-gradient dot | 5955 | 6706 |
| Protection active, nonzero loss updates | 2983 | 3144 |
| Completed epochs ending with positive protection coefficient | 29 | 29 |
| Maximum protection coefficient observed | 0.638998 | 0.578394 |

Action gradients are finite and nonzero for both predictors. Local independent verification checked all downloaded per-update ledger entries for accepted/blocked transfer consistency, nonzero protection loss, complete sequential step identities, support hashes and every epoch's bounded dual update. MECHANISM_LOCAL_AUDIT.json is PASS. Protection counts mean at least one positive-weight violated group in that optimizer update, not simply that the module was enabled. Task7 correctly stayed in RA fallback, with no predictor updates, transfer or protection.

The last 10 observed Task9 epochs averaged 23.30 seconds. Six epochs remain in the later ledger, approximately 2.33 minutes at that rate. Training completion is expected around 21:48 CST, followed by storage, all 200 formal records, checkpoint/process audits and local independent recomputation. Using the earlier matched evaluation runtime of about 79 minutes, a provisional full-completion window is **23:05-23:35 CST on 2026-10-21**. This is not yet a measured CAET-v2 evaluation ETA.

Both mechanisms are executing, but their success-rate benefit and retention performance remain unknown until formal evaluation. Main BC still supervises a[u], whereas the causal auxiliary target is a[u+1]; these different targets and exploratory support thresholds remain disclosed. The run remains PARTIAL, with no automatic extra seeds.
