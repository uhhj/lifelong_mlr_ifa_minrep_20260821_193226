# EPR Stage A: frozen zero-update diagnostic

User authorization: existing RA checkpoints/data, GOAL seed42 and OBJECT seed41.
No training, optimizer construction/step, environment construction/step, action
sampling, new closed-loop evaluation, automatic Stage B, or changes to original runs.

Teacher: task-at-learning ep99 checkpoint. Students: every later ep99 stage.
Tasks 6/7/8, transitions 6->7/8/9, 7->8/9, 8->9. All original cached
windows are inspected, separately reporting RA selected and nonselected windows.
Both deterioration and improvement are reported.

Native observations: before prediction, fix up to two windows for each
task x demo x real-target-time quartile x selected/not-selected stratum.
Deduplicate real (demo,target) within each stratum, choose sorted quantiles
1/3 and 2/3 with fixed lexicographic filename ties. Exclude unknown identities
and nonunique starts from native checks; never guess from file numbers.
Maximum 160 native windows/task, 960 total. Native feature tensors remain only in
process memory and are not exported or reused for training.
Verify complete 8x7 cached expert action window against original HDF5 and dataset
output, including padding. Native preprocess/spatial/temporal/head follow eval
mode up to distribution parameters; no .sample or simulator. Frozen frontend is
verified identical across stages. Formal exporter remains unchanged.

Scale: exact original selector robust_scale over the task's original ten raw
demonstration action sequences; verify against stored selector scale.
Motion error = RMS first-six mean-action difference divided by fixed scales.
Also report expected standardized squared error including variance, gripper
sign correctness and analytic wrong-sign probability, winner margin, output
Gaussian W2, and nearest teacher-component match vs actual argmax.
Do not identify component-index changes with behavioral harm.

Frozen exploratory thresholds (not tuned on this diagnostic):
- teacher motion error <=0.5 robust-scale units and correct expert gripper sign;
- native eligibility additionally requires teacher valid in both views and
  teacher cache/native motion-mean RMS <=0.25, std RMS <=0.25;
- material motion harm: student error - teacher error >=0.1;
  material benefit: teacher error - student error >=0.1;
- gripper SF and FS counted in addition, with overlaps explicitly reported.
All continuous errors and per-demo summaries are also retained; thresholds are
diagnostic conventions, not physical-success boundaries.

Discovery demos0-4 and validation demos5-9 are fixed before inference. No fitting
or parameter choice uses validation. All demos were seen by historical models;
"validation" means diagnostic split, not full-policy unseen demonstrations.
Window and distinct-real-target summaries must not be treated as iid trials.

Each suite separately is worth considering B only if engineering PASS and:
in native validation RA-selected teacher-eligible rows, >=10 distinct harmful
(task,demo,target) across >=3 demos and >=2 task-stage cells; cache/native harm
Jaccard >=0.5 on the same eligible sampled rows. Report beneficial transfer and
whether it dominates. No opportunity / cache-only / inadequate support is
NO-GO or INCONCLUSIVE, never an automatic new experiment. A pass establishes
protection opportunity, not that EPR can realize it without harming learning.

Inference cost cap: only the above finite cached/native dataset scans; batch16
cache, batch4 native. One immutable parent set per suite, no hyperparameter scan.
Use eval mode and deterministic inference flags locally to diagnosis, explicitly
recording departure from historical benchmark=True. No global RNG in old process
exists or is altered. Hash all read checkpoints/data/config/source before/after.

Outputs: new standalone analysis directory, lightweight reduced statistics and
identity metadata only; no raw actions, means/std vectors, images, latent,
checkpoint, teacher training sidecar, or RNG binary publication.
Related code/report/evidence are published to MLR under standing authorization.
