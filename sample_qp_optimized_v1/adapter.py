"""Drop-in step adapter for the frozen 300-step experiment, with fresh references."""
import time
import numpy as np
from .core import Workspace, Timings, inner, project


def update(algo, ds, plan, spec, Pold, name, horizon, workspace, guard,
           profile=False):
    from sample_step_closedloop_v1.train import g
    start = time.perf_counter()
    t = Timings(profile)
    stage = spec['point']['stage_step']+horizon-1
    if horizon > 1:
        g.ensure_next_epoch_rng(algo, ds, plan, stage-1, stage)
    with t.phase('parameter_snapshot'):
        before = workspace.snapshot_parameters() if Pold else None
    H = K = None
    if Pold:
        tasks = sorted({task for half, task in Pold})
        batches = [data for task in tasks for half in ['P0','P1']
                   for data in Pold[half, task]]
        count = sum(len(data['feature']) for data in batches)
        assert count == 32*len(tasks) and count <= 64
        modes = {n:m.training for n,m in algo.policy.named_modules()}
        algo.policy.eval()
        try:
            with t.phase('sample_gradients'):
                def loss_fn(data):
                    return g.module(algo).policy_head.loss_fn(
                        g.gmm(algo, data), data['actions'].squeeze(1), 'none')
                H = workspace.sample_gradients(batches, loss_fn, count)
        finally:
            for n,m in algo.policy.named_modules():
                m.training = modes[n]
        with t.phase('gram'):
            K, _ = inner(H, gram=True)
    with t.phase('native_mlr'):
        g.native(algo, g.batch(ds, spec['batches'][horizon-1]), 8,
                 stage//plan['batches_per_epoch'])
    result = dict(point=name, arm='SAMPLE_STEP_QP_OPTIMIZED', horizon=horizon,
        stage_step=stage, lr=algo.optimizer.param_groups[0]['lr'],
        batch_indices=spec['batches'][horizon-1], trigger=False)
    if H is not None:
        with t.phase('native_displacement'):
            native = workspace.displacement(before)
        correction, sol = project(H, K, native, t)
        with t.phase('write_and_statistics'):
            workspace.subtract_correction(correction)
            actual = workspace.displacement(before)
            _, rates = inner(H, actual)
            native_rates = np.asarray(sol['f'])  # Reuse identical already-computed products.
            dn = float(native.double().norm())
            an = float(actual.double().norm())
            hn = np.sqrt(np.maximum(np.diag(K), 0))
            result.update(trigger=bool(any(x > 0 for x in sol['lam'])),
                constraints=H.shape[1], active=len(sol['active']), kkt=sol['kkt_error'],
                projection_residual=sol['written_normalized_rate_max'],
                correction_norm=float(correction.double().norm()),
                actual_correction_norm=float((actual-native).double().norm()),
                native_norm=dn, actual_norm=an, actual_norm_ratio=an/max(dn,1e-30),
                native_positive=int((native_rates>5e-6*hn*dn+1e-15).sum()),
                actual_positive=int((rates>5e-6*hn*an+1e-15).sum()))
    result['seconds'] = time.perf_counter()-start
    if profile:
        result['profile_seconds'] = t.seconds
    guard.check(force=True)
    return result
