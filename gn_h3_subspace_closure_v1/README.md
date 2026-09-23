# Directional closure and basis-invariant H3

Only two experiments, same seed45 original-MLR task8 ep99 checkpoint and original task8/task9 trajectory split. No optimizer, formal policy, buffer, solver, training or rollout changes.

A validates analytic derivatives on the real original GMM NLL using centered differences. 35 directions x two probe tasks x eight fixed spacings; each direction/task must close at two adjacent small spacings before B runs. Centered-difference closure does not make a finite one-sided update first-order accurate: even-order terms cancel in the former but not the latter.

B uses the entire projected new-task gradient. Its relation score is g_old.T P g_new, with P=U solve(U.T U,U.T) accounting for stored basis rounding. Actual probes have matched parameter norm, so loss-change prediction includes epsilon / ||P g_new||. This is not the unscaled score or an actual AdamW contribution. Ranks 4/8/16/21, mean-GN/model-Fisher/one fixed random basis, and two previously used scale endpoints only.

Existing mean basis is read-only. Existing Fisher top16 is read-only; the same frozen model draws extend the space with five columns. Within-span orthogonal rotation is explicitly verified. Different truncations/ranks remain different spans.

Run with the existing `.migration/activate.sh` environment:

```bash
python -m gn_h3_subspace_closure_v1.freeze
python -m unittest gn_h3_subspace_closure_v1.tests -v
python -m gn_h3_subspace_closure_v1.run
python -m gn_h3_subspace_closure_v1.report
python -m gn_h3_subspace_closure_v1.audit
```

Freeze and completed runs preserve existing results. Report/audit use saved CSV only. No package installation. Source dependencies are the already committed `gn_functional_geometry_v1.engine` and `gn_h3_scale_geometry_diag_v1.fisher`; loader report paths are explicitly redirected into this experiment, never formal runtime paths.

The post-measurement infinitesimal relation decomposition is an additional summary of already saved old-relation/probe projection columns, not another experiment or selection rule. It separates cross-trajectory first-order signs from finite-step errors. Four nested ranks and one random basis are not independent statistical trials.

Large Fisher extra-five basis stays on the server, hash-recorded in ARTIFACT_MANIFEST.json. No previous bases are overwritten. Final inference is local/descriptive, not a method-performance conclusion.
