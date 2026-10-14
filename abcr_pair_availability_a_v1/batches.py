"""Supplement the supplied script with paired random/retrieved batch statistics."""
import random
import torch
from .original_probe import _similarities
from .settings import FROZEN_PARAMETERS, validate


def labels(v, d, grip, threshold, action_share=FROZEN_PARAMETERS['action_share'], action_bif=FROZEN_PARAMETERS['action_bif']):
    assert action_share < action_bif
    match = v >= threshold
    share = match & (d <= action_share) & ~grip
    bif = match & ((d >= action_bif) | grip)
    assert not (share & bif).any()
    return match, share, bif


def compare(nv, ov, na, oa, scale, old_tasks, seed, trials=None, parameters=None):
    """Same five anchors per trial across strategies and visual thresholds.

    Retrieval returns five neighbors per anchor (up to 25 unique old windows),
    whereas random uses five shared old windows. Both score 25 pairs, but the
    retrieval search inspects 5*n_old cosines; costs and uniqueness are explicit.
    Trajectory deduplication is unavailable from original feature/actions files.
    """
    rng = random.Random(seed)
    parameters = dict(FROZEN_PARAMETERS if parameters is None else parameters)
    validate(parameters)
    if trials is not None:
        assert trials == parameters['batch_trials'], 'BATCH_TRIALS_PARAMETER_MISMATCH'
    trials = parameters['batch_trials']
    k = parameters['topk']; bn = parameters['batch_new']; bo = parameters['batch_old']
    chunk = parameters['chunk']; thresholds = parameters['visual_thresholds']
    n, m = len(nv), len(ov)
    assert n >= bn and m >= max(bo, k)
    top_v, top_d, top_g, top_ids = [], [], [], []
    for start in range(0, n, chunk):
        v, d, grip, _ = _similarities(nv[start:start+chunk], ov, na[start:start+chunk], oa, scale)
        ids = torch.topk(v, k=k, dim=1).indices
        top_ids.append(ids)
        top_v.append(v.gather(1, ids))
        top_d.append(d.gather(1, ids))
        top_g.append(grip.gather(1, ids))
    top_ids, top_v, top_d, top_g = [torch.cat(x) for x in (top_ids, top_v, top_d, top_g)]
    rows = []
    for trial in range(trials):
        ci = torch.tensor(rng.sample(range(n), bn))
        oi = torch.tensor(rng.sample(range(m), bo))
        rv, rd, rg, _ = _similarities(nv[ci], ov[oi], na[ci], oa[oi], scale)
        for threshold in thresholds:
            for strategy, v, d, g, ids in (
                ('RANDOM_%d_PLUS_%d' % (bn, bo), rv, rd, rg, oi[None, :].expand(bn, -1)),
                ('RETRIEVE_TOP%d_PER_ANCHOR' % k, top_v[ci], top_d[ci], top_g[ci], top_ids[ci]),
            ):
                match, share, bif = labels(v, d, g, threshold, parameters['action_share'], parameters['action_bif'])
                useful = share | bif
                cols = ids[useful]
                anchors = ci[:, None].expand_as(ids)[useful]
                unique_pairs = len(set(zip(anchors.tolist(), cols.tolist())))
                rows.append(dict(trial=trial, visual_threshold=threshold, strategy=strategy,
                    current_indices=';'.join(map(str, ci.tolist())),
                    old_indices=';'.join(map(str, ids.flatten().tolist())),
                    share_pairs=int(share.sum()), bif_pairs=int(bif.sum()),
                    uncertain_pairs=int((match & ~useful).sum()), matched_pairs=int(match.sum()),
                    has_share=bool(share.any()), has_bif=bool(bif.any()),
                    has_both=bool(share.any() and bif.any()),
                    unique_old_windows=int(ids.unique().numel()),
                    unique_useful_old_windows=int(cols.unique().numel()),
                    unique_useful_current_windows=int(anchors.unique().numel()),
                    unique_useful_window_pairs=unique_pairs,
                    old_tasks_sampled=int(old_tasks[ids].unique().numel()),
                    old_tasks_useful=int(old_tasks[cols].unique().numel()),
                    evaluated_pairs=int(v.numel()), search_cosines=bn*m if strategy.startswith('RETRIEVE') else bn*bo,
                    unique_trajectory_pairs=None))
    summaries = []
    for threshold in thresholds:
        for strategy in ('RANDOM_%d_PLUS_%d' % (bn, bo), 'RETRIEVE_TOP%d_PER_ANCHOR' % k):
            group = [r for r in rows if r['visual_threshold'] == threshold and r['strategy'] == strategy]
            out = dict(visual_threshold=threshold, strategy=strategy, trials=trials)
            for src, dst in [('has_share', 'fraction_batches_with_share'), ('has_bif', 'fraction_batches_with_bif'),
                ('has_both', 'fraction_batches_with_both'), ('share_pairs', 'mean_share_pairs'),
                ('bif_pairs', 'mean_bif_pairs'), ('unique_old_windows', 'mean_unique_old_windows'),
                ('unique_useful_window_pairs', 'mean_unique_useful_window_pairs'),
                ('old_tasks_useful', 'mean_useful_old_task_coverage'), ('search_cosines', 'mean_search_cosines')]:
                out[dst] = sum(r[src] for r in group) / trials
            out['trajectory_deduplication'] = 'UNAVAILABLE_NOT_IN_PT'
            summaries.append(out)
    return rows, summaries
