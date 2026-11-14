"""Build disjoint, paired training and common probe memories from one collection."""
import collections
import contextlib
import io

from evaluate import load
from coreset_exp.model import data_setup
from memory import tensor_sha
from runtime import *


def match(online, experts, base):
    """Return parallel rows, with deterministic choices and pair-preserving holdout."""
    order_rng = np.random.RandomState(base + 1)
    selection_online = np.random.RandomState(base + 2)
    selection_expert = np.random.RandomState(base + 3)
    probe_rng = np.random.RandomState(base + 4)
    remaining = list(order_rng.permutation(sorted(experts)))
    pairs = []
    for traj, rows in sorted(online.items(), key=lambda item: (-len(item[1]), item[1][0]['initial_index'])):
        grouped = collections.defaultdict(list)
        for row in rows:
            grouped[row['length']].append(row)

        def compatible(demo):
            return sum(min(len(values), len(experts[demo].get(length, [])))
                       for length, values in grouped.items())

        if not remaining:
            break
        demo = max(remaining, key=compatible)
        if compatible(demo) == 0:
            continue
        remaining.remove(demo)
        for length in sorted(grouped):
            a = sorted(grouped[length], key=lambda row: row['step'])
            b = sorted(experts[demo].get(length, []), key=lambda row: row['u'])
            count = min(len(a), len(b))
            ia = sorted(selection_online.choice(len(a), count, replace=False))
            ib = sorted(selection_expert.choice(len(b), count, replace=False)) if count else []
            pairs.extend((traj, a[i], b[j]) for i, j in zip(ia, ib))
    counts = collections.Counter(t for t, _, _ in pairs)
    held = set()
    for i in probe_rng.permutation(len(pairs)):
        if len(held) == 4:
            break
        traj = pairs[i][0]
        if counts[traj] > 1:
            counts[traj] -= 1
            held.add(int(i))
    assert all(v > 0 for v in counts.values())
    assert all(a['length'] == b['length'] for _, a, b in pairs)
    return pairs, held, sorted(counts)


def expert_candidates(ds, task):
    # A legal full window ending at observation 7 starts at observation 0.
    # Its frozen per-frame features supply the genuine 1..7 prefix histories.
    binding = js(O / 'bindings' / ('task%d.json' % task))['windows']
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    seen = set()
    for i, row in enumerate(ds.rows):
        if row['task'] != task or not row['bc_valid']:
            continue
        identity = binding[Path(ds.man[i]['path']).name]
        u, demo = int(row['u']), row['demo']
        assert identity.get('identity_unique', True) and identity['sha256'] == row['sample_sha256']
        assert identity['target'] == u and identity['demo_id'] == demo
        assert set(identity['possible_starts']) == {u - 7}
        if (demo, u) in seen:
            continue
        seen.add((demo, u))
        for length in (range(1, 9) if u == 7 else (8,)):
            groups[demo][length].append(dict(demo=demo, u=(length - 1 if length < 8 else u),
                source_index=i, source_sha256=row['sample_sha256'], length=length))
    return groups


@torch.no_grad()
def expert_reference(policy, ds, task, traj, descriptor):
    x = ds.native[descriptor['source_index']][0]['feature'].squeeze(0).float()
    assert tuple(x.shape) == (8, 4, 512)
    feature = x[:descriptor['length']].clone()
    mu, sd, logits = policy.policy_head.forward_fn(policy.temporal_encode(feature[None].cuda())[:, -1])
    k = int(logits[0].argmax())
    return dict(descriptor, source='demonstration', task=task, trajectory=traj,
                feature=feature, mu=mu[0, k].cpu(), sd=sd[0, k].cpu(), mode=k)


def lightweight(row):
    return {**{k: v for k, v in row.items() if k not in ('feature', 'mu', 'sd')},
            **{k + '_sha256': tensor_sha(row[k]) for k in ('feature', 'mu', 'sd')}}


@torch.no_grad()
def main():
    verify()
    capacity()
    assert (ACQUISITION / 'acquire.exit').read_text() == '0'
    assert not MATCHED_MEMORY.exists(), 'Existing matched cache must be preserved'
    control_seed(SEED)
    cfg = EasyDict(js(O / 'resolved_eval.json'))
    cfg.training = False
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, shape = data_setup(cfg, False)
    cfg.shape_meta = shape
    output = {'E': {}, 'S': {}, 'probe': {}}
    summary = {}
    for task in (6, 7, 8):
        teacher = TEACHERS[str(task)]
        assert sha(teacher['path']) == teacher['sha256']
        dataset = Dataset(task)
        source = acquisition_path(task)
        meta = js(ACQUISITION / ('MEMORY_S_%d.json' % task))
        assert sha(source) == meta['cache_sha256'] and meta['source_sha256'] == teacher['sha256']
        raw = torch.load(source, map_location='cpu')
        online = collections.defaultdict(list)
        assert len(raw['rows']) == meta['count'] <= MEMORY_PER_TASK
        for row, proof in zip(raw['rows'], meta['rows']):
            assert lightweight(row) == proof
            online[row['trajectory']].append(row)
        assert len(online) == meta['collection_successes']
        base = 20261112 + SEED * 1000 + task * 100
        pairs, held, ids = match(online, expert_candidates(dataset, task), base)
        policy = load('S', task, cfg, shape)
        expert, success, probe, records = [], [], [], []
        for i, (traj, a, b) in enumerate(pairs):
            assert a['trajectory'] == traj and a['length'] == b['length']
            reference = expert_reference(policy, dataset, task, traj, b)
            if i in held:
                probe.extend((reference, a))
            else:
                expert.append(reference)
                success.append(a)
            records.append(dict(pair=i, probe=i in held, expert=lightweight(reference), online=lightweight(a)))
        assert len(expert) == len(success) and len(expert) + len(held) <= MEMORY_PER_TASK
        assert len(probe) == 2 * len(held) <= 8
        assert {row['trajectory'] for row in expert} == {row['trajectory'] for row in success} == set(ids)
        for arm, rows in [('E', expert), ('S', success), ('probe', probe)]:
            output[arm][task] = dict(rows=rows, scales=raw['scales'], support=len(ids)/10.,
                                     support_trajectories=ids, source_sha256=teacher['sha256'])
        summary[task] = dict(raw_trajectories=len(online), raw_windows=len(raw['rows']),
            paired_trajectories=len(ids), train_per_source=len(expert), probe_per_source=len(held),
            dropped_online_trajectories=sorted(set(online) - set(ids)),
            dropped_online_windows=len(raw['rows'])-len(pairs), support=len(ids)/10.,
            original_online_cache_sha256=meta['cache_sha256'], teacher_sha256=teacher['sha256'],
            rng_base=base, pair_records=records)
        del policy, dataset
    save(MATCHED_MEMORY, output)
    write(R / 'MATCHED_MEMORY.json', dict(status='PASS_MATCHED_HISTORY_CACHE',
          sha256=sha(MATCHED_MEMORY), tasks=summary, optimizer_calls=0, environments=0))


if __name__ == '__main__':
    main()
