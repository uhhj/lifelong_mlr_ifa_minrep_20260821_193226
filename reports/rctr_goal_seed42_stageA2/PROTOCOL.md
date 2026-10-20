# RCTR Stage A2 — current-state exploratory counterfactual screening

User approved a maximum of 333 continuations on 2026-10-20. This is an independent protocol. Historical Stage A remains BLOCKED and its thresholds and records are immutable. No historical demonstration reconstruction is required here.

## Frozen scope

- Frozen RA-Coreset GOAL seed42 Task6 epoch99 parent; native state=True policy, MuJoCo **3.2.3**, current installed assets and original GMM distribution.
- Three fixed official initial states **20, 21, 22**. These are exploratory reused states, not untouched confirmatory samples.
- For each state: env seed42, five zero actions, then exactly40 parent policy actions. Prefix noise is independent, seed `42030000+initial_index`. If success occurs before or at step40, that state is ineligible; do not replace it.
- Snapshot the actual observation after step40, the preceding seven raw observations, complete MjData, controller/gripper/robot caches, observable caches/timing, environment RNG, Python/NumPy/Torch/CUDA RNG, and policy latent queue. No resetting to historical coordinates.
- Three repeat noise tables per eligible snapshot, seeds `42040000+100*initial_index+repeat`, each600×7 independent standard Normal. Table rows refer to elapsed branch step and are matched across all37 candidates. This is a new exploratory matched-noise protocol, not the original stage-global evaluation protocol.
- Each repeat: one normal continuation; six interventions `±0.05` on each of the first three normalized continuous action coordinates at branch step0; for each intervention, six step1 recovery choices (original argmax-component Normal, or one of the five component means). All other steps use original argmax-component Normal. Component means are explicit one-step experimental interventions, never substituted for ordinary policy execution.
- The environment clips action commands. Actual clipped intervention differences must be nonzero; saturated directions are skipped before stepping and recorded, not replaced. Recovery means that are numerically similar are retained and their differences reported; no searching for better components.
- Each branch ends on first success or600 actions. Maximum333 branches. Report actual controller frequency and steps separately from wall-clock time.
- Prefix budget:3×(5 zero+40 policy)=135 control steps. Restore precheck:2×8 actions per eligible snapshot, at most48 engineering control steps. No retries or extra initial states based on outcomes.

## Strict engineering admission

All eligible snapshots must pass before **any** screening branch:

1. Complete native checkpoint loading and frozen input/source hashes.
2. Native get_action and extracted argmax Normal parameters/actions agree at history lengths1,4,8 (`atol=rtol=1e-5`).
3. Two repeats from the same snapshot, identical eight noise rows, produce physical state and all native numeric observations (including both RGB cameras) within absolute `1e-10`; GMM means/scales/logits and executed actions within `1e-5`. Restore error before action must be <=`1e-10`.
4. Eight-frame history and controller/observable timing are restored together. No extra forward/render/observation refresh inserted to reconstruct snapshot observations.

Failure preserves attempts and stops. No threshold relaxation, source patching or partial blind restart. Frozen inputs are rehashed after execution.

## Analysis and stop

Report all initial-state × direction × recovery × repeat outcomes, matched SS/SF/FS/FF and success differences against normal and against the same perturbation with original recovery; execution amplitude, candidate action differences, reachable-state descriptors, and actual interaction/storage cost.

Exploratory directional signal: a direction/recovery pair has at least two FS and zero SF against perturbed-original recovery over three matched repeats **at at least two eligible states**. This is a descriptive flag, not statistical significance or a training gate. Always report the complete grid and negative/zero effects. Failure to meet this flag does not prove RCTR impossible. Independent certification and a new predeclared effective-sample requirement must precede any training, and are outside this authorization.

No optimizer, training, buffer mutation, extra certification, new seeds or baseline reevaluation. Parent benchmark metrics remain unchanged. Only code and reviewed lightweight metadata/outcomes may be published. Raw snapshots/history/actions/noise remain private on server.
