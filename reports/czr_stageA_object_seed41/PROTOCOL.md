# CZR Stage A — frozen execution specification

2026-10-20. User attachment is the algorithm source. Execute only its first,
zero-training/zero-environment native-image feasibility experiment.

- Parent: RA+IFA OBJECT seed41 Task6 ep99, unchanged. Use its complete native
  eval image/preprocessing/temporal/head path, not augmented cached latents.
- Reuse independently verified full-action-window identities from previous
  Stage A. demo0–7 construct; demo8–9 once-only heldout. Four cross-demo ambiguous
  windows excluded. Multiple possible starts require exclusion rather than guessing.
  Validate selected original HDF5 actions against the complete cached window.
- Selected argmax-logits Gaussian mean gripper sign defines parent label class;
  report sampled-command results separately. Labels are real expert final actions.
- Center features with construction-only FP64 mean, normalize with
  `norm + 1e-12`. Radius exactly `0.45 * nearest_correct_distance`. Strict `<`.
  Empty error memory means exact fallback; missing correct protection means blocked.
  Zero-distance cases have zero radius and cannot activate.
- Support requires matching parent selected-component class; all matching support
  cases must agree on correction class. Conflict or absent support falls back.
- Organize neighboring errors by demo, consecutive real target step and direction.
  Keep original window centers within each event: no unprovided representative
  selection or radius expansion. Report both windows and independent events.
- No radius search. Eight leave-one-training-demo-out checks reconstruct mean,
  errors and correct protection from the other seven demos. These measure
  cross-demo support; they do not tune 0.45. Report insufficient support if no
  cross-demo error rescue. Fixed final memory uses demo0–7 only.
- Unique canonical normalized language match <=1e-6 and norm>1e-12 is the
  existing conservative language admission guard. Task6 admitted; other canonical,
  unknown, ambiguous and zero vectors rejected. No task ID input to action routing.
- Native get_action samples all seven original dimensions first, then applies
  a sparse expert-command override only for admitted supported cases.
  Motion and rejected actions remain bit-exact; global RNG consumption unchanged.
  No new optimizer, no environments, no rollouts, no changes to parent/buffer.
- Finite-correct-set protection verified on all construction windows. Heldout
  evaluated once with final memory. Actual get_action checked on every eligible
  image window; native features must match memory-query features. Additional
  history1/4/8 protected-language and empty/conflicting-support properties are
  interface tests, never experiment-success evidence.
- Prototype gate: heldout label repairs>=1 and harms=0, exact fallback/interface
  and input protection pass. Cross-demo-support sufficiency reported separately;
  absent support precludes recommending closure even if the heldout gate passes.
- Closure experiment B has no fixed initial states, episode budget or noise
  allocation in this attachment. Report gate first, do not launch B or expand.
  Parent trained on all ten demos: heldout is only relative to this new memory.
- Native features and error-bank payload remain server-only. Publish source,
  counts, distances, hashes and correctness metadata, not raw labels/actions,
  latent vectors, model weights, RNG or archive payload.
