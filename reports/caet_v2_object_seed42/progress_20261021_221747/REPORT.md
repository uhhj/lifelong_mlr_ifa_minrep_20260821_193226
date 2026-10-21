# CAET-v2 OBJECT seed42: PARTIAL evaluation

Observation: 2026-10-21T22:18:58.556389+08:00. Snapshot reads precede the final hash sweep and are not atomic.

All 400 epochs / 105500 main optimizer updates have completed. train6-9 and retain6-9 exit codes are all zero. Effect admission for Task8/9 passed; Task7 remained entirely RA fallback. Final Task8/9 transfer accepted counts are 7065/8878; conflicting transfer rejected counts 5955/7157; protection active counts 2983/5089. Auxiliary fitting cost remains 1820 extra updates. These are execution/functional counts, not performance gains.

Formal evaluation has 70/200 episode records, seven completed vectors and one in progress. The native evaluator PID68241 and ten spawned environment processes are active. No technical exceptions. Local recomputation validates unique episode identities, original initial-state indices/hashes, environment seeds, 600-step limits, attempt uniqueness and matched success vectors. Cell-summary validation, four strict checkpoint loads and process-exit acceptance remain for final packaging/audit; this progress report does not replace them.

All 518 frozen source/input hashes and local training/auxiliary step accounting PASS. Task6 actual tensor parity remains PASS. No recovery, repeated preflight, training launch or additional rollout was initiated by this check. The old initial import failure exit1 is preserved and is not the current run's status.

Seven completed vectors averaged 223.76 seconds. Remaining formal evaluation is approximately 48.48 minutes from the snapshot, with final server/local audits additional. Provisional completion window: **2026-10-21 23:10-23:35 CST**. Later-task duration can change this estimate. No partial AUC or performance gate has been computed.

Data disk free 9.79 GiB; system 5.50 GiB. Status stays PARTIAL; no seed expansion.
