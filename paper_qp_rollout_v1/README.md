# 20-step endpoints: paired closed-loop validation

This is a new evaluation of the existing `paper_active_set_v1` endpoints. It does not train a new carrier or change the QP rule. Original36origins are all retained, seeds45/46/47 remain development seeds. Only MLR and TASK_GRAD_QP are evaluated.

Run from the cloud repository after sourcing `.migration/activate.sh`:

```bash
python -m paper_qp_rollout_v1.freeze
nohup python -u -m paper_qp_rollout_v1.queue > reports/paper_qp_rollout_v1/launch.log 2>&1 < /dev/null &
```

The queue runs tests, exact6-origin recovery, a full20-trial identical-model logging parity check, full36-origin endpoint recovery, then180native evaluator calls with20trials each. Both arms are checked against original full-state hashes at1/5/20; endpoints are retained on the system disk and only inference parameters are exported. Full original anchors and original20-step schedules remain available to reconstruct moments.

Evaluation uses native600step limit,10lanes,2rounds,5zero-action warmup,8-observation history, original GMM sampling and native success OR. The only intentional RNG protocol change is explicit per-round reset shared by arms; this avoids cross-round termination-dependent RNG carry. These results are not mixed with historical evaluation streams. First20official initial states are fixed; reuse across origins must not be treated as independent replication.

The primary outcome is equal-task mean of seen task6..current within each origin, equal12origin mean within each seed, then equal3seed mean. Old/new and per-task success are mandatory. Predeclared practical decision thresholds are in PROTOCOL_FROZEN.json; no successful subset selection, extra trials, early outcome-based stopping, or best checkpoint selection.

Raw trial records plus action/done traces permit independent success reaggregation. Artifact hashes, environment, configs, parity evidence and seed metrics are written to reports/paper_qp_rollout_v1. A descriptive bootstrap conditions on these3seeds and resamples origins within seed; it is not an independent-seed significance claim. Technical failures stop the queue and preserve completed cells; a faithful resume must be audited explicitly.

Progress: inspect QUEUE_STATUS.json and the current log under logs/. Completion: FINAL_REPORT.md, seed_metrics.csv, task_metrics.csv, INDEPENDENT_AUDIT.json and DECISION.json. No complete lifelong training or task9 claim follows from this experiment.
