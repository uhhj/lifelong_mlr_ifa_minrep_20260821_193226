"""RA-Coreset CPU reference: original K-Center + uniform anchors + exact min-radius exchange.

Input IDs are stable original train-window file basenames (unique within a task).
Distances come ONLY from the frozen original visual-action Distance implementation;
this standalone module does not compute embeddings, gradients, scores, or rollouts.
No global random state is consumed. The real integration must verify the parent
Coreset and Random manifests, bytes and parent distance calibration first.
"""
from __future__ import annotations

import random
from typing import Sequence, Mapping, Any
import numpy as np

TIE_TOL = 1e-12


def anchor_ids(all_ids: Sequence[str], quota: int, seed: int) -> list[str]:
    """Exact prefix of the parent Random arm's size-quota draw."""
    ids = sorted(all_ids)
    if len(set(ids)) != len(ids) or not (1 <= quota <= len(ids)):
        raise ValueError('Invalid population or quota')
    return random.Random(seed).sample(ids, quota)[: quota // 2]


def _deletion_scores(D: np.ndarray, augmented: list[int]) -> tuple[np.ndarray,np.ndarray]:
    """Exact max/mean nearest-neighbor distance after deleting each center.

    Handles nearest-center ties without approximation: if two nearest centers are
    tied, second-nearest distance equals first and deletion penalty is zero.
    """
    sub = D[:, augmented]
    first_at = np.argmin(sub, axis=1)
    first = sub[np.arange(len(sub)), first_at]
    second = np.partition(sub, 1, axis=1)[:, 1]
    summed = np.bincount(first_at, weights=second-first, minlength=len(augmented))
    worse = np.zeros(len(augmented), dtype=np.float64)
    np.maximum.at(worse, first_at, second)
    radius = np.maximum(float(first.max()), worse)
    mean = float(first.mean()) + summed / len(sub)
    return radius, mean


def select_anchor_repair(
    all_ids: Sequence[str], parent_centers: Sequence[str],
    D: np.ndarray, *, seed: int, expected_random: Sequence[str] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Enforce uniform anchors via exactly necessary swaps from original Coreset.

    For each missing anchor, choose the removable non-anchor center minimizing
    (maximum coverage distance, average coverage distance, fixed filename ID).
    No threshold tuned on success rates or supervised loss is used.
    """
    ids = list(all_ids)
    n = len(ids)
    if ids != sorted(ids) or len(set(ids)) != n:
        raise ValueError('IDs must be unique and sorted')
    centers = list(parent_centers)
    quota = len(centers)
    if not (1 <= quota <= n) or len(set(centers)) != quota:
        raise ValueError('Invalid parent Coreset')
    index = {name: i for i, name in enumerate(ids)}
    if any(x not in index for x in centers):
        raise ValueError('Parent centers missing from population')
    D = np.asarray(D, dtype=np.float64)
    if D.shape != (n, n) or not np.isfinite(D).all() or np.any(D < -TIE_TOL):
        raise ValueError('Invalid frozen Distance matrix')
    if np.max(np.abs(np.diag(D))) > TIE_TOL:
        raise ValueError('Center self-distances must be zero')
    anchors = anchor_ids(ids, quota, seed)
    random_draw = random.Random(seed).sample(ids, quota)
    if expected_random is not None and list(expected_random) != random_draw:
        raise ValueError('Random parent selection does not match same-seed draw')
    protected = set(anchors)
    chosen = [index[x] for x in centers]
    replaced = []
    for name in anchors:
        x = index[name]
        if x in chosen:
            continue
        augmented = chosen + [x]
        radii, means = _deletion_scores(D, augmented)
        # No anchor may ever be deleted, including anchors previously in S_C.
        removable = [(p, ids[old]) for p, old in enumerate(chosen) if ids[old] not in protected]
        if not removable:
            raise AssertionError('Anchor insertion has no legal removable center')
        # Parent-style absolute tolerance. Stable tie order by original IDs.
        winner = None
        for p, old_name in sorted(removable, key=lambda it: it[1]):
            r, m = float(radii[p]), float(means[p])
            if winner is None or r < winner[0]-TIE_TOL or (
                abs(r-winner[0]) <= TIE_TOL and m < winner[1]-TIE_TOL
            ) or (
                abs(r-winner[0]) <= TIE_TOL and abs(m-winner[1]) <= TIE_TOL
                and old_name < winner[2]
            ):
                winner = (r, m, old_name, p)
        _, _, removed, position = winner
        chosen[position] = x
        replaced.append({'incoming': name, 'outgoing': removed,
                         'parent_center_slot': position,
                         'radius_after': winner[0], 'mean_after': winner[1]})
    result = [ids[i] for i in chosen]
    if len(set(result)) != quota or not protected.issubset(result):
        raise AssertionError('Quota, uniqueness or anchor requirement failed')
    if len(replaced) != len(set(anchors)-set(centers)):
        raise AssertionError('Not the minimum possible number of anchor additions')
    baseline = D[:, [index[x] for x in centers]].min(axis=1)
    final = D[:, chosen].min(axis=1)
    report = dict(version='RA_v1_half_quota_uniform_anchor_minimax_exchange',
                  population=n,quota=quota,anchor_count=len(anchors),
                  random_draw_ids=random_draw,anchor_ids=anchors,
                  intersection_with_parent=len(protected & set(centers)),
                  swaps=replaced,swap_count=len(replaced),
                  original_ids=centers,selected_ids=result,
                  max_before=float(baseline.max()),max_after=float(final.max()),
                  mean_before=float(baseline.mean()),mean_after=float(final.mean()))
    return result, report


def stable_slots(parent_sorted_buffer_ids: Sequence[str],
                 candidate_buffer_ids: Sequence[str]) -> tuple[list[str],list[dict[str,str|int]]]:
    """Preserve every common original member at the same pre-shuffle index.

    parent_sorted_buffer_ids are source identities in EXACT parent filename sort
    order, which may be different from parent K-Center selection order. Incoming
    identities occupy only the deleted parent slots; no files are renamed.
    """
    old = list(parent_sorted_buffer_ids)
    new = list(candidate_buffer_ids)
    if len(old) != len(new) or len(set(old)) != len(old) or len(set(new)) != len(new):
        raise ValueError('Population must be unique and quota-preserving')
    if old != sorted(old):
        raise ValueError('Parent slot order must match the parent sorted-file order')
    incoming = sorted(set(new)-set(old))
    removed_slots = [i for i, name in enumerate(old) if name not in set(new)]
    if len(incoming) != len(removed_slots):
        raise AssertionError('Symmetric set difference must be equal-size')
    result = old.copy()
    replacements=[]
    for i, name in zip(removed_slots,incoming):
        replacements.append(dict(slot=i,removed=old[i],incoming=name))
        result[i] = name
    if set(result) != set(new) or any(result[i] != x for i,x in enumerate(old) if x in set(new)):
        raise AssertionError('Common-member slot invariance failed')
    return result, replacements
