# CAET-v2 OBJECT42 Gate1 — PASS

Zero training, zero policy/model forward, zero environment calls, zero server writes. Metadata evidence and local graph/causal-index recomputation passed.

5561/5921 original windows have legal unpadded unambiguous next-action targets. Task6–9 valid counts:1472/1390/1470/1229. Remaining360 are excluded only from auxiliary labels; all original BC windows remain.

|Stage|Qualified groups|Supported current physical times|Heldout supported times|Native batches with support per epoch|
|---|---:|---:|---:|---:|
|7|0|0|0|0/227|
|8|6|165|73|120–139/309|
|9|7|206|64|151–174/362|

Graph nodes deduplicate task/demo/real time. Every surviving endpoint has partners in at least3 independent demos; every retained group has at least3 demos and12 distinct times on each side. Holdout demos do not enter graph fitting. Task7 must fully return to RA/IFA. Task8/9 support does not establish learned prediction reliability or transfer benefit; those checks remain ahead.

Causal target is actions[u+1], transition states[u+1]→states[u+2], while the latest input is obs[u]. Recorded robot/gripper observations match states[u+1] exactly. Physical pose and basket frame are reconstructed from recorded free-joint state/XML, with no simulator replay or rendering. Cache visual/state timing provenance relies on the previously audited exporter and identity mapping; no fresh image re-encoding is claimed.

Support thresholds were written and hashed before reading this v2 graph, but were designed after observing v1/GOAL results. This is exploratory, not an independent preregistered confirmation. Main BC still targets a[u], so causal auxiliary supervision has a different time target; lag correction remains a possible explanatory factor without a separate ablation.
