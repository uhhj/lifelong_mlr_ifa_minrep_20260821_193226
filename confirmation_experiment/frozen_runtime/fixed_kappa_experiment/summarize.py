"""Strict three-seed soft-vs-existing-endpoints summary and paired differences."""
import argparse
import csv
import json
from pathlib import Path

from metrics import aggregate, read_fixed_matrix, score_matrix
from protocol import endpoint_dirs, read, require, sha
from run import arm_name
from validate_run import validate


def summarize(repo, tag, kappa, output):
    result = {'protocol': 'LIBERO_OBJECT tasks6-9, ep99, 20 trials, seeds42/43/44; original FWT/NBT/AUC definitions',
              'kappa': kappa, 'runs': {'mlr': {}, 'soft': {}, 'hard': {}}, 'aggregate': {}, 'paired': {}}
    rows = []
    for seed in (42, 43, 44):
        soft = repo / 'runs/fixed_kappa' / tag / 'formal' / arm_name(kappa) / ('seed%d' % seed)
        validation = validate(soft, kappa)
        require((soft / 'PRIMARY_EVAL_COMPLETE').is_file(), 'Missing completed evaluation')
        completed = read(soft / 'evaluation_complete.json')
        require(completed['log_sha256'] == sha(soft / 'primary_ep99.log'), 'Soft evaluation log changed')
        require(completed['protocol_sha256'] == sha(soft / 'evaluation_protocol.json'), 'Soft evaluation protocol changed')
        manifest = read(soft / 'experiment.json')
        for arm, directory in dict(endpoint_dirs(repo, seed), soft=soft).items():
            log = directory / 'primary_ep99.log'
            if arm != 'soft':
                ref = manifest['references']['endpoints'][arm]
                require(sha(log) == ref['log_sha256'], 'Reference results changed after training')
                require(sha(directory / 'run_001/config.json') == ref['config_sha256'], 'Reference recipe changed')
            values = score_matrix(read_fixed_matrix(log))
            values['source_log'] = str(log)
            if arm == 'soft':
                values['diagnostics'] = validation['diagnostics']
                values['training_wall_seconds'] = read(soft / 'train_execution.json')['wall_seconds']
            result['runs'][arm][str(seed)] = values
            for stage in range(4):
                for task in range(stage+1):
                    rows.append({'arm': arm, 'seed': seed, 'stage': stage+6, 'task': task+6,
                                 'success_percent': values['r'][stage][task]})
    names = ('fwt', 'nbt', 'auc', 'final_mean_success', 'worst_old_final_success', 'worst_peak_to_final_forgetting')
    for arm, seeds in result['runs'].items():
        result['aggregate'][arm] = {n: aggregate([s[n] for s in seeds.values()]) for n in names}
    for baseline in ('mlr', 'hard'):
        result['paired']['soft_minus_'+baseline] = {n: dict(aggregate([
            result['runs']['soft'][str(s)][n]-result['runs'][baseline][str(s)][n] for s in (42,43,44)]),
            per_seed={str(s): result['runs']['soft'][str(s)][n]-result['runs'][baseline][str(s)][n]
                      for s in (42,43,44)}) for n in names}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    with (output / 'success_matrices.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines = ['# 固定系数软控制实验结果', '', '固定 κ=%g；三seed均值 ± 样本标准差。' % kappa,
             'FWT沿用本项目对角线均值定义，并非独立零样本前向迁移指标。', '',
             '| 方法 | FWT ↑ | NBT ↓ | AUC ↑ |', '| --- | ---: | ---: | ---: |']
    for arm in ('mlr', 'soft', 'hard'):
        scores = result['aggregate'][arm]
        lines.append('| %s | %s |' % (arm, ' | '.join('%.2f ± %.2f' % (scores[n]['mean'], scores[n]['sample_std']) for n in ('fwt','nbt','auc'))))
    lines += ['', '## 配对差值：soft − 对照', '', '| 对照 | ΔFWT | ΔNBT | ΔAUC |', '| --- | ---: | ---: | ---: |']
    for arm in ('mlr', 'hard'):
        stats = result['paired']['soft_minus_'+arm]
        lines.append('| %s | %s |' % (arm, ' | '.join('%+.2f' % stats[n]['mean'] for n in ('fwt','nbt','auc'))))
    lines += ['', '执行完成与效果改善分开判断。本报告不按成功率过滤seed或选择checkpoint；三seed不足以支持强显著性结论。',
              '若只恢复MLR性能且保持收益消失，不能认定软控制有净收益。完整逐seed结果与触发统计见results.json。', '']
    (output / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True, type=Path)
    p.add_argument('--tag', default='fixed_v1')
    p.add_argument('--kappa', type=float, default=.5)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    out = a.output or a.repo / 'reports/fixed_kappa' / a.tag / arm_name(a.kappa)
    print(json.dumps(summarize(a.repo.resolve(), a.tag, a.kappa, out)['aggregate'], indent=2))
