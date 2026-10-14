"""Supplement the supplied script with paired random/retrieved batch statistics."""
import random
import torch
from .original_probe import _similarities


def labels(v, d, grip, threshold):
    match = v >= threshold
    share = match & (d <= 0.5) & ~grip
    bif = match & ((d >= 1.5) | grip)
    assert not (share & bif).any()
    return match, share, bif


def compare(nv, ov, na, oa, scale, old_tasks, seed, trials=500):
    """Same five anchors per trial across strategies and visual thresholds.

    Retrieval returns five neighbors per anchor (up to 25 unique old windows),
    whereas random uses five shared old windows. Both score 25 pairs, but the
    retrieval search inspects 5*n_old cosines; costs and uniqueness are explicit.
    Trajectory deduplication is unavailable from original feature/actions files.
    """
    rng = random.Random(seed)
    n, m = len(nv), len(ov)
    assert n >= 5 and m >= 5
    top_v, top_d, top_g, top_ids = [], [], [], []
    for start in range(0, n, 128):
        v, d, grip, _ = _similarities(nv[start:start+128], ov, na[start:start+128], oa, scale)
        ids = torch.topk(v, k=5, dim=1).indices
        top_ids.append(ids)
        top_v.append(v.gather(1, ids))
        top_d.append(d.gather(1, ids))
        top_g.append(grip.gather(1, ids))
    top_ids, top_v, top_d, top_g = [torch.cat(x) for x in (top_ids, top_v, top_d, top_g)]
    rows = []
    for trial in range(trials):
        ci = torch.tensor(rng.sample(range(n), 5))
        oi = torch.tensor(rng.sample(range(m), 5))
        rv, rd, rg, _ = _similarities(nv[ci], ov[oi], na[ci], oa[oi], scale)
        for threshold in (0.7, 0.8, 0.9, 0.95):
            for strategy, v, d, g, ids in (
                ('RANDOM_5_PLUS_5', rv, rd, rg, oi[None, :].expand(5, -1)),
                ('RETRIEVE_TOP5_PER_ANCHOR', top_v[ci], top_d[ci], top_g[ci], top_ids[ci]),
            ):
                match, share, bif = labels(v, d, g, threshold)
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
                    evaluated_pairs=25, search_cosines=5*m if strategy.startswith('RETRIEVE') else 25,
                    unique_trajectory_pairs=None))
    summaries = []
    for threshold in (0.7, 0.8, 0.9, 0.95):
        for strategy in ('RANDOM_5_PLUS_5', 'RETRIEVE_TOP5_PER_ANCHOR'):
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
