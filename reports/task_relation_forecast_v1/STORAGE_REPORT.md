# Storage cleanup and prospective experiment preparation

User authorized removal of unused assets. Deleted pip download cache (2,519,887,249 bytes) and 202 completed seed42–44 evaluation candidates (17,491,034,442 bytes). The selected checkpoint and epoch95 checkpoint for every task/method were retained (86 unique files); all 86 verified present after deletion. Each deleted checkpoint has an archived SHA256 and evidence log in STORAGE_DELETIONS.jsonl. Logs/configurations/evaluation results, raw data, buffers, base weights, seed45–47 states and formal recovery slots were retained.

These unselected weights can no longer be directly re-evaluated without reconstruction. Historical published selections remain available. No claim that the deleted candidate weights were duplicates.

New large assets are isolated under diagnostic_storage/task_relation_forecast_v1 on /data, with a 2 GiB stop reserve. Frozen frontend is shared; source step300 is reused as target reference; only decoder/head endpoints and one active full recovery state are retained by the new source runner. No automatic historical cleanup is performed by runners.

CPU protocol tests: 6/6. Native cache smoke: PASS for one window per A/D/E role. Scientific results are not available yet. Source pilot prepares only seed45 tasks6 and7. Formal lifelong training remains stopped.

Base exposure audit: official checkpoint provenance and released pretraining recipe agree on tasks0–5; original author training-job manifest unavailable, recorded explicitly. Native LR step0 equals0, therefore first shadow-update signal is expected to be degenerate; no schedule change is made to improve that signal.
