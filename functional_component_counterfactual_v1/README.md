# Functional component counterfactual v1

Independent diagnostic only. User protocol: reports/functional_component_counterfactual_v1/USER_PROTOCOL_v1.md.
No sample-QP imports, formal checkpoints/solver changes, or training continuation.

Run in the existing .migration/activate.sh environment:
```
python -m functional_component_counterfactual_v1.run --stage inventory
python -m functional_component_counterfactual_v1.run --stage smoke
python -m functional_component_counterfactual_v1.run --stage offline --resume
python -m functional_component_counterfactual_v1.run --stage behavior --resume
python -m functional_component_counterfactual_v1.run --stage report
```

The smoke stage keeps six independent shadow states in CPU RAM while calibrating numerical resolution on D only.
After all six calibrations, the common per-task numerical bands are frozen, D-only choices saved, and T opened.
GN/counterfactual candidates use native head FP32 values; BASE/R/M decoder caches are computed separately.
No physical feasibility gate from previous QP experiments applies.
The only native steps before measurement reconstruct missing historical states from their nearest complete anchors,
in this experiment's own rolling slot. Direct existing states are preferred. No formal training is resumed.

Completed per-start measurements are skipped. Behavior evaluates the six preselected points, twenty trials per task
and arm with the original evaluator, bounded at 480 rollouts. Behavior endpoints are model-only evaluation assets,
never optimizer restart checkpoints. Old and new losses use trajectory-equal averages.
