#!/usr/bin/env python3
"""Read-only analysis of the registered GOAL state_v2 episode evidence.

Python >=3.8, standard library only. No training, environment creation, torch,
checkpoint loading, network requests, or changes to the input directory.
New diagnostic rates are descriptive; they do not identify a causal replay effect.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

RUN_ID = 'goal_seed42_mlr_ifa_paper_aligned_state_v2'
PUBLICATION_COMMIT = 'eb15698ae8ee356f6098b9777e5c7dad21450b0e'
TASKS = (6, 7, 8, 9)
REQUIRED = (
    'evaluation/rollouts.jsonl', 'metrics.json', 'final_acceptance.json',
    'checkpoints/manifest.json', 'modal_contract.json', 'source.patch',
)
OPTIONAL = ('evaluation/success_matrix.csv', 'resolved_eval.yaml',
            'data_manifest.json', 'pretrained_manifest.json', 'logs/commands.log')


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _unique_keys(pairs):
    value = {}
    for k, v in pairs:
        require(k not in value, 'Duplicate JSON key: ' + k)
        value[k] = v
    return value


def _nonfinite(value):
    raise AuditError('Non-standard nonfinite JSON value: ' + value)


def parse_json(text):
    return json.loads(text, object_pairs_hook=_unique_keys, parse_constant=_nonfinite)


def read_json(path):
    return parse_json(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, obj):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def write_csv(path, rows):
    require(bool(rows), 'Cannot write an empty CSV without a schema')
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def is_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def close(left, right, label):
    require(type(left) in (int, float) and type(right) in (int, float),
            'Non-numeric value: ' + label)
    require(math.isfinite(left) and math.isfinite(right) and
            math.isclose(left, right, rel_tol=0, abs_tol=1e-12),
            'Value mismatch: %s: %r != %r' % (label, left, right))


def paper_metrics(cells):
    rates = {(s, t): sum(r['success'] for r in rows) / 20.0
             for (s, t), rows in cells.items()}
    nbt = [statistics.mean(rates[(t, t)] - rates[(s, t)]
                           for s in range(t + 1, 10)) for t in TASKS[:-1]]
    auc = [statistics.mean(rates[(s, t)] for s in range(t, 10)) for t in TASKS]
    return rates, {
        'FWT': statistics.mean(rates[(t, t)] for t in TASKS),
        'NBT': statistics.mean(nbt), 'AUC': statistics.mean(auc),
        'FinalAvgSR': statistics.mean(rates[(9, t)] for t in TASKS),
        'NBT_per_task': nbt, 'AUC_per_task': auc,
    }


def validate_records(records, acceptance, checkpoints, contract):
    require(len(records) == 200, 'Expected exactly 200 formal records; do not deduplicate or impute')
    require(acceptance.get('status') == 'PASS', 'Upstream acceptance must report PASS')
    require(set(checkpoints) == {'6', '7', '8', '9'}, 'Four stage checkpoint manifests required')
    require(contract.get('use_state') is True, 'Expected state_v2 modal contract')
    require(contract.get('token_order') == ['language', 'agentview', 'eye_in_hand', 'state'],
            'Unexpected token order')
    body = {k: v for k, v in contract.items() if k != 'fingerprint'}
    fingerprint = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'),
                                             ensure_ascii=True).encode()).hexdigest()
    require(fingerprint == contract.get('fingerprint') == acceptance.get('modal_fingerprint'),
            'Modal contract fingerprint mismatch')
    cells = {(s, t): [] for s in TASKS for t in range(6, s + 1)}
    seen_ids, seen_hashes = set(), set()
    names = {}
    for n, row in enumerate(records, 1):
        label = 'record %d' % n
        required = ('kind', 'run_id', 'training_seed', 'stage', 'task_id', 'task_name',
                    'checkpoint_sha', 'eval_seed', 'env_batch_seed', 'worker_seed',
                    'init_state_id', 'init_state_hash', 'success', 'policy_steps',
                    'termination_reason', 'exception', 'modal_fingerprint')
        require(all(k in row for k in required), label + ': missing fields')
        require(row['kind'] == 'FORMAL' and row['run_id'] == RUN_ID, label + ': wrong scope')
        for k in ('training_seed', 'stage', 'task_id', 'eval_seed', 'env_batch_seed',
                  'worker_seed', 'init_state_id', 'policy_steps'):
            require(type(row[k]) is int, label + ': not an integer: ' + k)
        stage, task, init = row['stage'], row['task_id'], row['init_state_id']
        require((stage, task) in cells and 0 <= init < 20, label + ': invalid cell or initial ID')
        require(row['training_seed'] == row['eval_seed'] == 42, label + ': wrong seed')
        require(row['env_batch_seed'] == 42 + init // 10, label + ': wrong environment batch seed')
        require(row['worker_seed'] == row['env_batch_seed'] + init % 10, label + ': wrong worker seed')
        require(type(row['success']) is bool, label + ': success must be JSON boolean')
        require(row['exception'] is None, label + ': exception recorded')
        require(1 <= row['policy_steps'] <= 600, label + ': invalid policy steps')
        expected_reason = 'success' if row['success'] else 'max_policy_steps'
        require(row['termination_reason'] == expected_reason, label + ': inconsistent termination')
        require(row['success'] or row['policy_steps'] == 600, label + ': failed episode must reach horizon')
        require(is_sha(row['init_state_hash']) and is_sha(row['checkpoint_sha']), label + ': invalid SHA')
        require(row['modal_fingerprint'] == fingerprint, label + ': wrong modal fingerprint')
        require(checkpoints[str(stage)]['task'] == stage and
                checkpoints[str(stage)]['epoch_index'] == 99, 'Wrong final checkpoint metadata')
        require(row['checkpoint_sha'] == checkpoints[str(stage)]['sha256'] ==
                acceptance['stages'][str(stage)]['checkpoint_sha256'], label + ': checkpoint mismatch')
        key_id, key_hash = (stage, task, init), (stage, task, row['init_state_hash'])
        require(key_id not in seen_ids and key_hash not in seen_hashes, label + ': duplicate initial state')
        seen_ids.add(key_id)
        seen_hashes.add(key_hash)
        require(isinstance(row['task_name'], str) and bool(row['task_name']), label + ': empty name')
        require(task not in names or names[task] == row['task_name'], label + ': task name changed')
        names[task] = row['task_name']
        cells[(stage, task)].append(row)
    for (stage, task), rows in cells.items():
        require(len(rows) == 20 and {r['init_state_id'] for r in rows} == set(range(20)),
                'Incomplete cell %s' % ((stage, task),))
        rows.sort(key=lambda r: r['init_state_id'])
        baseline = {r['init_state_id']: r for r in cells[(task, task)]}
        for row in rows:
            original = baseline[row['init_state_id']]
            for k in ('init_state_hash', 'eval_seed', 'env_batch_seed', 'worker_seed',
                      'task_name', 'modal_fingerprint', 'training_seed'):
                require(row[k] == original[k], 'Cross-stage initial-state mismatch: task=%d id=%d field=%s'
                        % (task, row['init_state_id'], k))
        count = sum(r['success'] for r in rows)
        require(count == acceptance['success_counts_out_of_20']['%d,%d' % (stage, task)],
                'Counts disagree with upstream acceptance')
    return cells


def comparisons():
    # Views overlap. They are not independent experiments or extra rollouts.
    adjacent = [('adjacent', t, s - 1, s) for s in (7, 8, 9) for t in range(6, s)]
    from_learned = [('from_learned', t, t, s) for t in (6, 7, 8) for s in range(t + 1, 10)]
    return adjacent + from_learned


def summarize_pairs(rows):
    counts = Counter(row['transition'] for row in rows)
    n = len(rows)
    ss, sf, fs, ff = [counts[k] for k in ('SS', 'SF', 'FS', 'FF')]
    before, after = ss + sf, ss + fs
    step_changes = [row['after_policy_steps'] - row['before_policy_steps']
                    for row in rows if row['transition'] == 'SS']
    return {
        'n_pairs': n, 'SS': ss, 'SF': sf, 'FS': fs, 'FF': ff,
        'before_successes': before, 'after_successes': after,
        'observed_loss_pp': 100.0 * sf / n,
        'observed_gain_pp': 100.0 * fs / n,
        'net_success_change_pp': 100.0 * (fs - sf) / n,
        'observed_flip_pp': 100.0 * (sf + fs) / n,
        'conditional_retention': ss / before if before else None,
        'conditional_loss': sf / before if before else None,
        'ss_step_change_n': len(step_changes),
        'ss_median_after_minus_before_steps': statistics.median(step_changes) if step_changes else None,
    }


def paired_views(cells):
    detail, summaries = [], []
    for view, task, before, after in comparisons():
        # Hash join within task; ordering and initial IDs are only checked metadata.
        old = {r['init_state_hash']: r for r in cells[(before, task)]}
        new = {r['init_state_hash']: r for r in cells[(after, task)]}
        require(set(old) == set(new) and len(old) == 20, 'Cannot pair all 20 initial states')
        rows = []
        for h in sorted(old, key=lambda key: old[key]['init_state_id']):
            a, b = old[h], new[h]
            row = {
                'view': view, 'task_id': task, 'before_stage': before, 'after_stage': after,
                'init_state_id': a['init_state_id'], 'init_state_hash': h,
                'before_success': a['success'], 'after_success': b['success'],
                'transition': ('S' if a['success'] else 'F') + ('S' if b['success'] else 'F'),
                'before_policy_steps': a['policy_steps'], 'after_policy_steps': b['policy_steps'],
                'before_checkpoint_sha': a['checkpoint_sha'], 'after_checkpoint_sha': b['checkpoint_sha'],
                'eval_seed': a['eval_seed'], 'env_batch_seed': a['env_batch_seed'],
                'worker_seed': a['worker_seed'],
            }
            rows.append(row)
        detail.extend(rows)
        summaries.append(dict(view=view, task_id=task, before_stage=before, after_stage=after,
                              **summarize_pairs(rows)))
    return detail, summaries


def state_histories(cells):
    result = []
    for task in TASKS:
        lookup = {s: {r['init_state_hash']: r for r in cells[(s, task)]} for s in range(task, 10)}
        for initial in cells[(task, task)]:
            h = initial['init_state_hash']
            observations = [lookup[s][h] for s in range(task, 10)]
            values = [r['success'] for r in observations]
            observed_later = len(values) > 1
            flips = sum(a != b for a, b in zip(values[:-1], values[1:])) if observed_later else None
            row = {'task_id': task, 'task_name': initial['task_name'],
                   'init_state_id': initial['init_state_id'], 'init_state_hash': h}
            row.update({'success_stage%d' % s: lookup[s][h]['success'] if s >= task else None for s in TASKS})
            row.update({
                'observed_sequence': ''.join('1' if v else '0' for v in values),
                'observed_checkpoints': len(values), 'later_checkpoints': len(values) - 1,
                'diagonal_success': values[0], 'final_success': values[-1],
                'pattern': ('single_observation' if not observed_later else
                            'all_success' if all(values) else 'all_failure' if not any(values) else 'mixed'),
                'adjacent_flips': flips,
                'initial_success_ever_observed_failure': (not all(values[1:])) if observed_later and values[0] else None,
                'initial_success_lost_then_final_success': (not all(values[1:]) and values[-1])
                    if observed_later and values[0] else None,
            })
            result.append(row)
    return result


def nbt_decomposition(summaries):
    per_task = []
    for task in TASKS[:-1]:
        rows = [r for r in summaries if r['view'] == 'from_learned' and r['task_id'] == task]
        loss = statistics.mean(r['observed_loss_pp'] for r in rows)
        gain = statistics.mean(r['observed_gain_pp'] for r in rows)
        per_task.append({'task_id': task, 'mean_observed_loss_pp': loss,
                         'mean_observed_gain_pp': gain, 'signed_NBT_pp': loss - gain})
    loss = statistics.mean(r['mean_observed_loss_pp'] for r in per_task)
    gain = statistics.mean(r['mean_observed_gain_pp'] for r in per_task)
    return {'definition': 'Within task: mean over all later stages from its learned checkpoint; then mean over tasks 6,7,8.',
            'interpretation': 'Descriptive success-outcome decomposition, not causal forgetting or validated new benchmark metrics.',
            'per_task': per_task, 'macro_observed_loss_pp': loss,
            'macro_observed_gain_pp': gain, 'signed_NBT_pp': loss - gain}


def _text(value):
    return 'NA' if value is None else str(value)


def report_text(metrics, summaries, histories, decomp):
    lines = ['# E1：既有闭环记录同初态配对审计', '',
             '**范围：仅分析既有200条记录；未新增训练或环境执行。**', '',
             '## 1. 原指标复算', '',
             '| 指标 | 结果 |', '|---|---:|']
    for k in ('FWT', 'NBT', 'AUC', 'FinalAvgSR'):
        lines.append('| %s | %.6f %s |' % (k, metrics[k]*100, '个百分点' if k == 'NBT' else '%'))
    lines += ['', '## 2. 初态配对结果', '',
              'SS=成功→成功；SF=成功→失败；FS=失败→成功；FF=失败→失败。每行20个配对初态。', '',
              '| 视角 | 任务 | 阶段 | SS | SF | FS | FF | 净变化(pp) | 翻转份额(%) |',
              '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append('| %s | %d | %d→%d | %d | %d | %d | %d | %.2f | %.2f |' %
                     (r['view'], r['task_id'], r['before_stage'], r['after_stage'],
                      r['SS'], r['SF'], r['FS'], r['FF'], r['net_success_change_pp'], r['observed_flip_pp']))
    lines += ['', '## 3. NBT 的有符号分解', '',
              '按论文NBT的相同权重先任务内、再任务间平均，观测成功→失败份额 %.6f pp，'
              '观测失败→成功份额 %.6f pp；二者相减 = %.6f pp。' %
              (decomp['macro_observed_loss_pp'], decomp['macro_observed_gain_pp'], decomp['signed_NBT_pp']),
              '净变化小不能单独判定两种变化都小；本表不把结果更替等同于回放造成的因果错误转移。', '',
              '## 4. 初态跨阶段历史', '',
              '| 任务 | 刚学完失败数/20 | 有后续观测 | 原成功初态中曾观察到失败 | 其中最终又成功 |',
              '|---|---:|---|---:|---:|']
    for t in TASKS:
        rows = [r for r in histories if r['task_id'] == t]
        has_later = t < 9
        lost = sum(r['initial_success_ever_observed_failure'] is True for r in rows) if has_later else None
        recovered = sum(r['initial_success_lost_then_final_success'] is True for r in rows) if has_later else None
        lines.append('| %d | %d | %s | %s | %s |' % (t, sum(not r['diagonal_success'] for r in rows),
                     '是' if has_later else '否', _text(lost), _text(recovered)))
    lines += ['', '## 5. 证据边界与停止条件', '',
              '- 两种比较视角共240行，其中部分重复；只有180个去重后的“任务/前后阶段/初态”配对。'
              '它们复用同一批episode，不是240或180次新增独立试验。',
              '- 每任务20个初态、每个checkpoint仅一次动作采样执行。相同初态与记录的seed不保证每个episode获得相同随机动作流；'
              '不作显著性、因果回放伤害或模型不确定性推断。',
              '- task9只有一个checkpoint观察，遗忘、恢复和后续稳定性均记NA，而不是0。',
              '- step变化仅在同一配对两端均成功的SS子集上描述；不把失败的600步当成成功耗时。'
              '该子集不能代表全部初态，duration_seconds不作为策略效率指标。',
              '- 日志没有抓取/搬运/释放阶段事件，不能定位具体动作阶段、动作分支或责任训练窗口；不调用模拟器补造证据。',
              '- 本分析已使用原测试结果；逐初态输出仅限分析，不能生成训练采样权重或用原测试结果选择超参数。',
              '- 下一阶段实验方案必须另行冻结协议；本脚本不会自动执行训练、评估、重跑或push。', '',
              '本次PASS只表示轻量证据内部一致、计算和文件只读检查通过，不表示已重新验证服务器权重二进制或独立复现训练。', '']
    return '\n'.join(lines)


def audit(evidence_root, out_dir):
    root, out = Path(evidence_root).resolve(), Path(out_dir).resolve()
    require(root.is_dir(), 'Evidence root not found: ' + str(root))
    require(out != root and root not in out.parents and out not in root.parents,
            'Output must be a separate directory, not inside/above evidence root')
    require(not out.exists(), 'Output already exists; preserve it and choose a new directory')
    out.mkdir(parents=True, exist_ok=False)
    try:
        for relative in REQUIRED:
            require((root / relative).is_file(), 'Missing required evidence: ' + relative)
        inputs = REQUIRED + tuple(p for p in OPTIONAL if (root / p).is_file())
        before = {p: sha256(root / p) for p in inputs}
        acceptance = read_json(root / 'final_acceptance.json')
        original_metrics = read_json(root / 'metrics.json')
        checkpoints = read_json(root / 'checkpoints/manifest.json')
        contract = read_json(root / 'modal_contract.json')
        require(before['source.patch'] == acceptance['source_patch_sha256'] == contract['code_patch_sha256'],
                'Source patch hash mismatch')
        records = [parse_json(line) for line in (root / REQUIRED[0]).read_text(encoding='utf-8-sig').splitlines()
                   if line.strip()]
        cells = validate_records(records, acceptance, checkpoints, contract)
        rates, metrics = paper_metrics(cells)
        for k in ('FWT', 'NBT', 'AUC', 'FinalAvgSR'):
            close(metrics[k], original_metrics[k], 'metrics.json/' + k)
            close(metrics[k], acceptance['metrics'][k], 'final_acceptance/' + k)
        for k in ('NBT_per_task', 'AUC_per_task'):
            require(len(metrics[k]) == len(original_metrics[k]), 'Length mismatch: ' + k)
            for i, (a, b) in enumerate(zip(metrics[k], original_metrics[k])):
                close(a, b, '%s[%d]' % (k, i))
        detail, summaries = paired_views(cells)
        histories = state_histories(cells)
        decomposition = nbt_decomposition(summaries)
        close(decomposition['signed_NBT_pp'], metrics['NBT'] * 100, 'Paired NBT identity')
        unique_pairs = {(r['task_id'], r['before_stage'], r['after_stage'], r['init_state_hash']) for r in detail}
        require(len(detail) == 240 and len(summaries) == 12 and len(histories) == 80 and len(unique_pairs) == 180,
                'Unexpected analysis dimensions')
        for row in summaries:
            require(row['SS']+row['SF']+row['FS']+row['FF'] == 20, 'Invalid transition partition')
            close(row['net_success_change_pp'], 100*(rates[(row['after_stage'], row['task_id'])] -
                                                   rates[(row['before_stage'], row['task_id'])]), 'Net SR identity')
        matrix_rows = [dict(stage=s, **{'task_%d' % t: rates[(s, t)] if t <= s else 'NaN' for t in TASKS}) for s in TASKS]
        write_csv(out / 'success_matrix_recomputed.csv', matrix_rows)
        write_json(out / 'metrics_recomputed.json', metrics)
        write_csv(out / 'paired_transitions.csv', detail)
        write_csv(out / 'paired_summary.csv', summaries)
        write_csv(out / 'initial_state_histories.csv', histories)
        write_json(out / 'nbt_decomposition.json', decomposition)
        with (out / 'REPORT.md').open('x', encoding='utf-8') as stream:
            stream.write(report_text(metrics, summaries, histories, decomposition))
        after = {p: sha256(root / p) for p in inputs}
        require(before == after, 'Input evidence changed during analysis')
        artifacts = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
        result = {
            'status': 'PASS', 'analysis_id': 'goal_seed42_state_v2_paired_audit_v1',
            'scope': 'Lightweight evidence validation and descriptive paired analysis only; no checkpoint binary verification.',
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'evidence_root': str(root), 'expected_publication_commit': PUBLICATION_COMMIT,
            'commit_note': 'Source reference; actual origin of a local evidence copy must be recorded by the operator. No Git checkout was performed.',
            'script_sha256': sha256(Path(__file__)), 'input_hashes_before': before, 'input_hashes_after': after,
            'input_files_unchanged': True, 'formal_episodes': 200, 'cells': 10,
            'adjacent_pairs': 120, 'from_learned_pairs': 120,
            'pair_rows_total': 240, 'unique_stage_pair_initial_states': 180,
            'initial_state_histories': 80, 'new_training_steps': 0, 'new_rollouts': 0,
            'NBT_identity_passed': True, 'output_hashes': artifacts,
        }
        write_json(out / 'audit_acceptance.json', result)
        return result
    except Exception as exc:
        with (out / 'BLOCKED.md').open('x', encoding='utf-8') as stream:
            stream.write('# BLOCKED\n\n%s: %s\n\nNo input data was repaired or overwritten.\n' %
                         (type(exc).__name__, exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.evidence_root, args.out_dir)
    except (AuditError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print('BLOCKED: %s' % exc, file=sys.stderr)
        return 2
    print(json.dumps({'status': result['status'], 'out_dir': str(args.out_dir.resolve()),
                      'new_training_steps': 0, 'new_rollouts': 0}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
