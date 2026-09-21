"""v3 batch185 single-step writeback validation.

Restores the real pre-failure state (task8/epoch11/batch185, global step 41984)
and executes exactly one update through the v3 chain
(step.update -> core.project -> solver.solve with L-BFGS-B fallback).

Verifies:
  - protocol verify() passes and the fresh build matches the official init hashes
  - state restore parity (model/optimizer/scheduler/RNG)
  - the historical QP instance reappears (standard path KKT = 4.864608e-8)
  - the fallback rescues it (KKT <= 1e-8) and project() writeback gate passes
  - optimizer/scheduler advance exactly once; LR follows the frozen schedule
  - two independent runs are bit-identical (determinism)
  - a post-step recovery snapshot reloads exactly
Writes WRITEBACK_VALIDATION.json. Does not touch the formal v3 run directory.
"""
import json
import time
from pathlib import Path

from sample_qp_lifelong_v1_continuation_v3 import common as c
from sample_qp_lifelong_v1_continuation_v3.step import update, Workspace

ROOT = Path('/data/lifelong_mlr_ifa_minrep_20260912')
V2_OUT = ROOT / 'reports/sample_qp_lifelong_v1_continuation_v2'
TARGET = ROOT / 'reports/qp_later_failure_diag_v1/t8_e11_b180'
OUTDIR = ROOT / 'reports/sample_qp_lifelong_v1_continuation_v3/regression/b185_writeback'
EXPECTED_STANDARD_KKT = 4.864608038612772e-08
EXPECTED_LR_PRE = 8.985526315789474e-05


def build_and_run(g, torch, run_id):
    import contextlib
    import gc
    import io
    from cb_ifa_experiment.gpu_runtime import bootstrap_train, build_train
    cfg = c.config(OUTDIR / ('checkpoints_run%d' % run_id))
    bootstrap_train(cfg)
    algo = build_train(cfg)
    expected = json.loads((V2_OUT / 'INITIALIZATION_EXPECTED.json').read_text())
    assert g.digest(algo.policy.state_dict()) == expected['model_hash'], 'init model mismatch'
    plan, ds, sampler, loader = c.dataset_loader(8)
    with contextlib.redirect_stdout(io.StringIO()):
        algo.start_task(-1, loader)
    saved = torch.load(TARGET / 'PRE_FAILURE_STATE.pt', map_location='cpu')
    g.restore(algo, saved['state'])
    assert all(g.compare_states(saved['state'], g.capture(algo)).values()), 'restore parity'
    assert algo._optimizer_step_count == 3584, algo._optimizer_step_count
    assert algo._scheduler_step_count == 3584, algo._scheduler_step_count
    data = g.batch(ds, saved['record_indices'])
    ids = [int(x) for x in data[1]]
    assert ids == saved['source_ids'], (ids, saved['source_ids'])
    assert c.sha(c.OUT / 'plans/references_task8.json') == saved['reference_sha256'], \
        'reference plan changed'
    refs = c.reference_batches(algo, 8)
    w = Workspace(g.params(algo))
    guard = c.Guard()
    rec = update(algo, data, 8, 11, refs, w, guard)
    final = g.capture(algo)
    digests = {k: g.digest(v) for k, v in final.items()}
    del refs, w, ds, loader, plan
    gc.collect()
    return algo, rec, digests, final


def main():
    from functional_geometry_diagnostic_v1 import geometry as g
    import gc
    torch = g.torch
    c.verify()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    g.init_ddp(OUTDIR / 'ddp')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    report = dict(target='task8/epoch11/batch185', pre_global_step=41984,
                  formal_training_resumed=False)
    try:
        algo1, rec1, dig1, _ = build_and_run(g, torch, 1)
        del algo1
        gc.collect(); torch.cuda.empty_cache()
        algo2, rec2, dig2, final2 = build_and_run(g, torch, 2)

        from libero.lifelong.utils import make_linear_schedule_with_warmup
        lr_fn = make_linear_schedule_with_warmup(500, 30900)
        expected_lr_post = 1e-4 * lr_fn(3585)

        checks = dict(
            fallback_triggered=bool(rec1['fallback_triggered']),
            standard_kkt_matches_history=bool(
                abs(rec1['standard_kkt'] - EXPECTED_STANDARD_KKT) <= 1e-14),
            fallback_kkt_within_threshold=bool(rec1['kkt'] <= 1e-8),
            projection_residual_within_gate=bool(rec1['projection_residual'] <= 5e-6),
            optimizer_step_exactly_once=bool(rec1['optimizer_step'] == 3585),
            scheduler_step_exactly_once=bool(rec1['scheduler_step'] == 3585),
            lr_pre_matches=bool(abs(rec1['lr'] - EXPECTED_LR_PRE) <= 1e-16),
            lr_post_matches_schedule=bool(abs(rec1['lr_after'] - expected_lr_post) <= 1e-14),
            actual_positive_zero=bool(rec1['actual_positive'] == 0),
            deterministic_two_runs=bool(dig1 == dig2),
        )
        report.update(rec1={k: rec1[k] for k in (
            'mixed_bc', 'lr', 'lr_after', 'trigger', 'constraints', 'optimizer_step',
            'scheduler_step', 'active', 'kkt', 'projection_residual', 'native_norm',
            'actual_norm', 'actual_correction_norm', 'correction_ratio',
            'actual_positive', 'native_positive', 'seconds', 'fallback_triggered',
            'standard_kkt', 'fallback_kkt', 'fallback_nit', 'fallback_seconds')},
            expected_lr_post=expected_lr_post, checks=checks)

        # Post-step recovery snapshot and exact reload.
        import contextlib
        import io
        from cb_ifa_experiment.gpu_runtime import bootstrap_train, build_train, atomic_save
        post_path = OUTDIR / 'post_step_state.pt'
        atomic_save(post_path, dict(
            state=final2, task=8, epoch=11, batch=185, global_step=41985,
            optimizer_step=rec2['optimizer_step'], scheduler_step=rec2['scheduler_step'],
            protocol_sha256=c.sha(c.OUT / 'PROTOCOL_FROZEN.json')))
        saved_post = torch.load(post_path, map_location='cpu')
        cfg = c.config(OUTDIR / 'checkpoints_reload')
        bootstrap_train(cfg)
        algo3 = build_train(cfg)
        plan3, ds3, sampler3, loader3 = c.dataset_loader(8)
        with contextlib.redirect_stdout(io.StringIO()):
            algo3.start_task(-1, loader3)
        g.restore(algo3, saved_post['state'])
        reload_ok = all(g.compare_states(saved_post['state'], g.capture(algo3)).values())
        checks['post_step_snapshot_reloads_exactly'] = bool(reload_ok)
        checks['reload_counters'] = bool(algo3._optimizer_step_count == 3585
                                         and algo3._scheduler_step_count == 3585)
        report['post_step_state_sha256'] = c.sha(post_path)
        report['checks'] = checks
        report['status'] = 'PASS' if all(checks.values()) else 'FAIL'
        out = OUTDIR / 'WRITEBACK_VALIDATION.json'
        tmp = out.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        tmp.replace(out)
        print(json.dumps(report, indent=2, allow_nan=False))
        print('WRITEBACK_VALIDATION', report['status'])
    finally:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
