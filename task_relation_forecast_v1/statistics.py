"""Frozen fold-local ridge and dependence-preserving outcome reconstruction."""
import numpy as np
from .common import TASKS, SEEDS

FAMILIES = {
    'M0': ['b_t', 'b_s'],
    'M_IFA': ['b_t', 'b_s', 'S_visual', 'S_language'],
    'M_A': ['b_t', 'b_s', 'A'],
    'M_LA': ['b_t', 'b_s', 'S_lat', 'A'],
    'M_GA': ['b_t', 'b_s', 'S_GN', 'A'],
    'M_probe': ['b_t', 'b_s', 'R_old', 'R_new'],
}


def folds(rows, mode='task'):
    result = []
    for held in TASKS if mode == 'task' else SEEDS:
        if mode == 'task':
            train = [i for i, r in enumerate(rows) if held not in (r['source'], r['target'])]
            test = [i for i, r in enumerate(rows) if r['target'] == held]
        else:
            train = [i for i, r in enumerate(rows) if r['seed'] != held]
            test = [i for i, r in enumerate(rows) if r['seed'] == held]
        result.append(dict(held=held, train=train, test=test))
    return result


def design(rows, family, train, test):
    keys = FAMILIES[family]
    x = np.asarray([[r[k] for k in keys] for r in rows], dtype=float)
    assert np.isfinite(x).all(), 'Unavailable signals must not be silently imputed or rows dropped'
    mean, scale = x[train].mean(0), x[train].std(0)
    zero = scale == 0
    safe = np.where(zero, 1., scale)
    x = (x - mean) / safe
    x[:, zero] = 0
    meta = dict(keys=keys, mean=mean.tolist(), scale=scale.tolist(), zero_variance=zero.tolist())
    if family in ('M_LA', 'M_GA'):
        inter = x[:, -2] * x[:, -1]
        mu, sd = inter[train].mean(), inter[train].std()
        inter = (inter - mu) / sd if sd > 0 else np.zeros_like(inter)
        x = np.column_stack((x, inter))
        meta['interaction'] = dict(mean=float(mu), scale=float(sd))
    return x[train], x[test], meta


def predict_fold(rows, family, y, train, test):
    a, b, meta = design(rows, family, train, test)
    y = np.asarray(y, dtype=float)
    mu, sd = y[train].mean(), y[train].std()
    meta.update(y_mean=float(mu), y_scale=float(sd), constant_target=bool(sd == 0))
    if sd == 0:
        return np.full(len(test), mu), meta
    # Intercept unpenalized; mean-MSE + 1.0 * ||beta||^2.
    center = a.mean(0)
    ac = a - center
    yc = (y[train] - mu) / sd
    beta = np.linalg.solve(ac.T @ ac / len(train) + np.eye(a.shape[1]), ac.T @ yc / len(train))
    return ((b - center) @ beta) * sd + mu, meta


def oof(rows, family, y, mode='task'):
    result = np.full(len(rows), np.nan)
    metadata = []
    for fold in folds(rows, mode):
        pred, meta = predict_fold(rows, family, y, fold['train'], fold['test'])
        result[fold['test']] = pred
        metadata.append(dict(fold, transform=meta))
    assert np.isfinite(result).all()
    return result, metadata


def reconstruct_targets(rows, success, indices=None):
    """success[(seed, model_id, eval_task)] is the paired 20-trial array.

    indices is keyed ONLY by (seed, eval_task), shared across every model.
    """
    def p(seed, model, task):
        v = np.asarray(success[(seed, model, task)], dtype=float)
        assert len(v) == 20
        return float(v.mean() if indices is None else v[indices[seed, task]].mean())
    targets = []
    for row in rows:
        seed, s, t = row['seed'], row['source'], row['target']
        ps = p(seed, 'source%d' % s, s)
        pst = p(seed, 'source%d' % s, t)
        pt = p(seed, 'base', t)
        ref = p(seed, 'reference%d' % t, t)
        finalt = p(seed, 'transfer%d_%d' % (s, t), t)
        finals = p(seed, 'transfer%d_%d' % (s, t), s)
        transfer0 = pst - pt
        transfer = finalt - ref
        targets.append(dict(TRANSFER_0=transfer0, TRANSFER_300=transfer,
                            ACQUISITION_DIFF_300=transfer-transfer0, DAMAGE_300=ps-finals))
    return targets


def bootstrap(rows, success, count=5000, seed=2026092501):
    rng = np.random.RandomState(seed)
    results = []
    for repeat in range(count):
        sampled = rng.choice(SEEDS, len(SEEDS), replace=True)
        # Each sampled seed block is an independent replicate; references are
        # shared within each block, including when one original seed repeats.
        rr, yy = [], []
        for block, original in enumerate(sampled):
            subset = [r for r in rows if r['seed'] == original]
            indices = {(int(original), t): rng.randint(0, 20, 20) for t in TASKS}
            labels = reconstruct_targets(subset, success, indices)
            rr.extend(dict(r, bootstrap_block=block) for r in subset)
            yy.extend(labels)
        one = dict(repeat=repeat)
        for target in ('TRANSFER_300', 'DAMAGE_300'):
            y = np.array([r[target] for r in yy])
            for family in FAMILIES:
                pred, _ = oof(rr, family, y)
                one[target + '/' + family] = float(np.mean((pred-y)**2))
        results.append(one)
    return results
