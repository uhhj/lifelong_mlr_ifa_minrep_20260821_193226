"""Audit recorded trajectories and recompute the manuscript comparison."""
import collections
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics

from common import ROOT, read, sha, write

WORKSPACE = ROOT.parents[1]
IDENTITY = ('stage', 'task', 'initial_index', 'initial_file_sha256', 'initial_sha256', 'noise_sha256', 'env_seed', 'attempt')
KEYS = [(stage, task, index) for stage in range(6, 10) for task in range(6, stage + 1) for index in range(20)]


def key(row):
    return row['stage'], row['task'], row['initial_index']


def audit(snapshot, contract, previous=None):
    spec = read(ROOT / 'SPEC.json')
    assert snapshot['contract'] == contract
    assert snapshot['failure'] is None and snapshot['controller_exit'] in (None, '0'), snapshot['failure']
    assert set(snapshot['cases']) == set(spec['evaluation_cases'])
    counts = {}
    for name, actual in snapshot['cases'].items():
        model = spec['evaluation_cases'][name]
        rows = actual.get('OUTCOMES', [])
        assert len(rows) <= 200 and len(rows) % 10 == 0
        assert list(map(key, rows)) == KEYS[:len(rows)]
        if previous is not None:
            old = previous['cases'][name].get('OUTCOMES', [])
            assert old == rows[:len(old)]
        if rows:
            assert actual['EVAL_ADMISSION']['status'] == 'PASS'
        for row in rows:
            stage, task, index = key(row)
            assert row['attempt'] == 1 and row['env_seed'] == model['seed'] + index // 10
            assert row['cp_sha256'] == model['models'][str(stage)]['sha256']
            assert [row['initial_file_sha256'], row['initial_sha256']] == model['initials'][f'{task}_{index}']
            assert 1 <= row['steps'] <= 600 and isinstance(row['success'], bool)
            assert (row['success'] and row['termination'] == 'success') or (not row['success'] and row['termination'] == 'timeout600' and row['steps'] == 600)
        if 'EVALUATION_COMPLETE' in actual:
            assert actual['EVALUATION_COMPLETE']['episodes'] == len(rows) == 200
        counts[name] = len(rows)
    if snapshot['controller_exit'] == '0':
        assert all(count == 200 for count in counts.values())
    return counts


def reference_rows(source, arm):
    filename, light_key = source.split('#light/', 1)
    path = Path(filename)
    if not path.is_absolute():
        path = WORKSPACE / path
    entry = read(path)['light'][light_key]
    assert hashlib.sha256(entry['text'].encode()).hexdigest() == entry['sha256']
    return [row for row in json.loads(entry['text']) if row['arm'] == arm]


def reference_cases(sources):
    arms = {'C_normal': 'C', 'Q': 'Q', 'S_latest': 'S'}
    records = {method: reference_rows(source, arms[method]) for method, source in sources.items()}
    expected = [tuple(row[field] for field in IDENTITY) for row in sorted(records['S_latest'], key=key)]
    assert len(expected) == 200
    for rows in records.values():
        assert sorted(map(key, rows)) == KEYS
        assert [tuple(row[field] for field in IDENTITY) for row in sorted(rows, key=key)] == expected
    return records


def summarize(rows):
    assert len(rows) == 200 and sorted(map(key, rows)) == KEYS
    cells = collections.Counter((row['stage'], row['task']) for row in rows if row['success'])
    rates = {(s, t): Fraction(cells[s, t], 20) for s in range(6, 10) for t in range(6, s + 1)}
    values = dict(
        AUC=sum(sum(rates[s, t] for s in range(t, 10)) / (10 - t) for t in range(6, 10)) / 4,
        FWT=sum(rates[t, t] for t in range(6, 10)) / 4,
        NBT=sum(sum(rates[t, t] - rates[s, t] for s in range(t + 1, 10)) / (9 - t) for t in range(6, 9)) / 3,
        OldFinal=sum(rates[9, t] for t in range(6, 9)) / 3,
        NewFinal=rates[9, 9], FinalAll=sum(rates[9, t] for t in range(6, 10)) / 4,
    )
    return dict(matrix_percent={f'{s}_{t}': float(rate * 100) for (s, t), rate in rates.items()},
                metrics_percent={name: float(value * 100) for name, value in values.items()})


def finalize(evidence, contract):
    snapshot = read(evidence)
    counts = audit(snapshot, contract)
    assert snapshot['controller_exit'] == '0' and not snapshot['processes'] and sum(counts.values()) == 2400
    spec = read(ROOT / 'SPEC.json')
    assert set(snapshot['cpu']) == set(counts)
    all_results = {}
    changes = {}
    pairs = {}
    old_audit_file = WORKSPACE / 'execution/20261111_main_ablation_protocol_audit/AUDIT.json'
    assert sha(old_audit_file) == spec['table_audit_sha256']
    old = read(old_audit_file)['cases']
    for suite in ('OBJECT', 'GOAL'):
        for seed in (41, 42, 43):
            references = spec['comparison_outcomes'][f'{suite}_{seed}']
            combined = reference_cases(references)
            for method in ('MLR_IFA', 'RA'):
                name = f'{suite}_{method}_{seed}'
                model = spec['evaluation_cases'][name]
                cpu = snapshot['cpu'][name]
                assert cpu['status'] == 'PASS_FINAL_CASE' and cpu['episodes'] == 200 and cpu['initial_bytes_verified'] == 80
                assert cpu['new_optimizer'] == cpu['new_environment'] == 0
                assert len(cpu['checkpoints']) == 4
                for stage, row in cpu['checkpoints'].items():
                    assert row['keys'] == 100 and row['sha256'] == model['models'][stage]['sha256']
                rows = snapshot['cases'][name]['OUTCOMES']
                encoded = (json.dumps(rows, indent=2, allow_nan=False) + '\n').encode()
                assert hashlib.sha256(encoded).hexdigest() == cpu['outcomes_sha256'] == snapshot['outcomes_sha256'][name]
                combined[method] = rows
            identities = [tuple(row[k] for k in IDENTITY) for row in sorted(combined['S_latest'], key=key)]
            for method, rows in combined.items():
                assert [tuple(row[k] for k in IDENTITY) for row in sorted(rows, key=key)] == identities, (suite, seed, method)
                name = f'{suite}_{method}_{seed}'
                all_results[name] = summarize(rows)
                if method in ('MLR_IFA', 'RA'):
                    former = old[f'{method}/{suite}/{seed}']['metrics_percent']
                    changes[name] = {metric: value - former[metric] for metric, value in all_results[name]['metrics_percent'].items()}
            pairs[f'{suite}_{seed}'] = {}
            for method in ('MLR_IFA', 'RA', 'C_normal', 'Q'):
                paired = collections.defaultdict(lambda: [0, 0])
                for baseline, debr in zip(sorted(combined[method], key=key), sorted(combined['S_latest'], key=key)):
                    cell = f"{debr['stage']}_{debr['task']}"
                    paired[cell][0] += int(debr['success'] and not baseline['success'])
                    paired[cell][1] += int(baseline['success'] and not debr['success'])
                pairs[f'{suite}_{seed}'][method] = dict(paired)
    metrics = ('FWT', 'NBT', 'AUC', 'OldFinal', 'NewFinal', 'FinalAll')
    methods = {'MLR_IFA': 'MLR+IFA', 'RA': 'RA-Coreset', 'C_normal': 'C', 'Q': 'Q', 'S_latest': '完整 DEBR'}
    table = {}
    lines = ['| 套件 | 方法 | FWT↑ | NBT↓ | AUC↑ | OldFinal↑ | NewFinal↑ | FinalAll↑ |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for suite in ('OBJECT', 'GOAL'):
        table[suite] = {}
        for method, label in methods.items():
            runs = [all_results[f'{suite}_{method}_{seed}']['metrics_percent'] for seed in (41, 42, 43)]
            stats = {metric: dict(mean=statistics.mean(r[metric] for r in runs), sd=statistics.stdev(r[metric] for r in runs)) for metric in metrics}
            table[suite][method] = stats
            cells = [f"{stats[m]['mean']:.3f} ± {stats[m]['sd']:.3f}" for m in metrics]
            lines.append('| ' + ' | '.join([suite, label] + cells) + ' |')
    write(ROOT / 'FINAL_RESULTS.json', dict(status='COMPLETE_ACCEPTED', observation=snapshot['observed'], evidence_sha256=sha(evidence),
          formal=2400, checkpoints=48, new_optimizer=0, table=table, cases=all_results, historical_protocol_delta_pp=changes,
          paired_discordances_debr_only_then_comparator_only=pairs))
    report = ('# 论文组件消融统一评估\n\nMLR+IFA 和 RA-Coreset 的 48 份原阶段模型已完成 2,400 个正序评估回合。'
              'C、Q 和完整 DEBR 沿用已验收结果。所有比较逐回合核对初态、动作噪声和环境种子。'
              '下表为三个种子的均值和样本标准差，单位为百分点。\n\n' + '\n'.join(lines) +
              '\n\n旧协议与本轮的分差包含实际运行环境差异。完整矩阵、逐种子分差及配对结果见 FINAL_RESULTS.json。\n')
    (ROOT / 'REPORT.md').write_text(report, encoding='utf-8')
