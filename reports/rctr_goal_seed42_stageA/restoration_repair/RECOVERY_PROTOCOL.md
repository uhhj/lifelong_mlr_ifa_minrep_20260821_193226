# RCTR restoration repair — 2026-10-20

User explicitly authorizes repairing state restoration and action/observation indexing, preserving failure evidence and all acceptance thresholds, then continuing the original maximum 54 Stage A continuations only after validation. No optimization, training or B/C.

Previous probe, exit1 asset failure, successful diagnostic exit0 with BLOCKED result, contracts, reports and all 316 engineering transitions remain immutable. New files and outputs are under `restoration_repair`.

Unchanged acceptance: low-dimensional observed state atol1e-6/rtol0; recorded transition max absolute state-coordinate error <=1e-5 excluding time; repeated full restoration/transition <=1e-10; native/full vs latent/head GMM atol1e-5/rtol1e-5. No rounding, threshold expansion, arbitrary latent substitute or replacement window.

Official reference: https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/scripts/create_dataset.py . The code executes action[j] and records returned observations but stores states[valid_index], dropping the first five action/observation rows. It is supporting provenance, not proof that current data came from that exact Git revision. Actual selected demonstrations independently show states[i].joint == obs[i-1].joint exactly. Check other fields/rendered observations at corrected indices before accepting.

Engineering correction candidates are limited to reconstructing omitted simulator/controller state from recorded states and known past expert commands, not changing the physics, action scaling or controls:

1. Reset simulator data explicitly after XML reset, restore flat state and force controller state cache refresh. Do not overwrite controller initial posture with the middle-trajectory posture.
2. Reconstruct retained OSC goal orientation and accumulated gripper command by feeding the known prefix commands into their existing update methods, with each command's recorded pre-state; no physics or outcome-based fitting. Unknown pre-trim controller history remains a limitation and must be validated against physical transitions.
3. Test actual expert prefix continuation with synchronized controller and fresh simulator state, separately from direct reconstruction.

No scientific outcome is inspected during engineering repair. Record all real environment steps and simulator restores. Once exact reconstruction is established, implement complete same-state snapshot restoration (MuJoCo integration state, controller/gripper/observation history and RNG), verify repeated actions and full eight-frame native histories, then freeze branch implementation. If the strict recorded-state gate remains unsatisfied, stop and report; do not call inability to reproduce the demonstration an algorithmic RCTR failure.

The original 3 fixed samples, perturbation, matching noise seeds, 5 Gaussian candidates, 3 repeats, 600-step cap and certificate criterion remain unchanged. Engineering checks are separately accounted and never counted as policy continuations.
