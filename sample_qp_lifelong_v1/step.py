"""Full-lifecycle displacement QP; original MLR moments and scheduler are retained."""
import time
import numpy as np
from sample_qp_optimized_v1.core import Workspace, inner, project

def update(algo, data, task, epoch, references, workspace, guard=None):
    from functional_geometry_diagnostic_v1 import geometry as g
    start = time.perf_counter()
    before = workspace.snapshot_parameters() if references else None
    H = K = None
    if references:
        count = sum(len(x['feature']) for x in references)
        assert count == 32*(task-6) and count in (32, 64, 96)
        modes = {n:m.training for n,m in algo.policy.named_modules()}
        algo.policy.eval()
        try:
            def loss(data):
                return g.module(algo).policy_head.loss_fn(g.gmm(algo,data),
                                                        data['actions'].squeeze(1), 'none')
            H = workspace.sample_gradients(references, loss, count)
        finally:
            for n,m in algo.policy.named_modules(): m.training = modes[n]
        K, _ = inner(H, gram=True)
    lr = float(algo.optimizer.param_groups[0]['lr'])
    loss = float(g.native(algo, data, task, epoch))
    assert np.isfinite(loss)
    rec = dict(mixed_bc=loss, lr=lr, lr_after=float(algo.optimizer.param_groups[0]['lr']),
               trigger=False, constraints=0 if H is None else H.shape[1],
               optimizer_step=algo._optimizer_step_count, scheduler_step=algo._scheduler_step_count)
    if H is not None:
        native = workspace.displacement(before)
        c, sol = project(H, K, native)
        workspace.subtract_correction(c)
        actual = workspace.displacement(before)
        _, rates = inner(H, actual)
        dn = float(native.double().norm()); an = float(actual.double().norm())
        hn = np.sqrt(np.maximum(np.diag(K), 0))
        actual_positive = int((rates > 5e-6*hn*an+1e-15).sum())
        # Record the physical write; project() retains the original stopping tolerance.
        correction = float((actual-native).double().norm())
        rec.update(trigger=any(v > 0 for v in sol['lam']), active=len(sol['active']),
            kkt=sol['kkt_error'], projection_residual=sol['written_normalized_rate_max'],
            native_norm=dn, actual_norm=an, actual_correction_norm=correction,
            correction_ratio=correction/max(dn,1e-30), actual_positive=actual_positive,
            native_positive=int((np.asarray(sol['f']) > 5e-6*hn*dn+1e-15).sum()))
    rec['seconds'] = time.perf_counter()-start
    if guard: guard.check()
    return rec
