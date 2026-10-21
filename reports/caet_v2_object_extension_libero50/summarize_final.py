"""Read-only per-task/cost/dual-chain audit after full independent acceptance."""
import json
import sys
from fractions import Fraction as F
from pathlib import Path

p = Path(sys.argv[1])
def read(path):
    return json.loads(path.read_text(encoding='utf8'))
a = read(p/'LOCAL_FINAL_AUDIT.json')
assert a['status'] == 'COMPLETED_LOCAL_INDEPENDENT_PASS'
mat = {(r['stage'], r['task']): r for r in a['matrix']}
tasks = {}
for method in ('ra', 'v1', 'v2'):
    tasks[method] = {}
    for t in range(6, 10):
        diag, final = F(mat[t,t][method]), F(mat[9,t][method])
        peak = max(F(mat[s,t][method]) for s in range(t,10))
        tasks[method][str(t)] = dict(diagonal_percent=float(100*diag),
            final_percent=float(100*final), diagonal_minus_final_pp=float(100*(diag-final)),
            peak_minus_final_pp=float(100*(peak-final)))
cost, constraints = {}, {}
for t in (7, 8, 9):
    root = p/'current/effects/models'/f'task{t}'
    fit = read(root/'ACCEPTANCE.json')
    if fit['status'] != 'PASS':
        assert fit['effect_updates'] == 0
        cost[str(t)] = dict(status=fit['status'], optimizer_updates=0)
        continue
    h = read(root/'epochs.json')
    assert len(h) == 10 and sum(r['updates'] for r in h) == fit['effect_updates']
    cost[str(t)] = dict(optimizer_updates=fit['effect_updates'],
        fit_seconds_raw=fit['seconds'], fit_seconds_valid=fit['seconds']>=0,
        fit_seconds=fit['seconds'] if fit['seconds']>=0 else None,
        anchor_exposures=sum(r['anchor_windows'] for r in h),
        total_window_exposures=sum(r['total_window_exposures'] for r in h),
        pair_exposures=sum(r['graph_pair_exposures'] for r in h))
    previous, peak, violations = None, 0., 0
    for ep in range(100):
        c = read(p/'current/ra_coreset/constraints'/f'task{t}_epoch{ep:02d}.json')
        assert (all(v==0 for v in c['mu_before'].values()) if previous is None
                else previous == c['mu_before'])
        previous = c['mu_after']
        peak = max(peak, max(previous.values(), default=0))
        violations += sum(c['values'][k]>c['bounds'][k]+.05 for k in c['bounds'])
    constraints[str(t)] = dict(max_mu=peak, violated_group_epochs=violations,
        final_violated_groups=sum(c['values'][k]>c['bounds'][k]+.05 for k in c['bounds']),
        groups=len(c['bounds']))
result = dict(status='TASK_AND_COST_RECOMPUTATION_PASS', seed=a['seed'], tasks=tasks,
    effect_cost=cost, constraints=constraints,
    main_epoch_seconds=a['main_training_seconds'],
    evaluation_vector_seconds=a['formal']['mean_vector_seconds']*20,
    historical_reference_scoring_seconds='Not separately timed',
    effect_optimizer_updates=sum(v['optimizer_updates'] for v in cost.values()))
(p/'TASK_COST_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
lines=['\n## 逐任务与成本复算\n',
    'FWT沿用各任务刚训练完成的对角线平均成功率；不是未训练任务的零样本迁移。NBT沿用原遗忘定义，越低越好。\n',
    '|任务|RA 对角线|v1 对角线|v2 对角线|RA 最终|v1 最终|v2 最终|',
    '|---|---:|---:|---:|---:|---:|---:|']
for t in range(6,10):
    values=[tasks[m][str(t)]['diagonal_percent'] for m in ('ra','v1','v2')]
    values += [tasks[m][str(t)]['final_percent'] for m in ('ra','v1','v2')]
    lines.append('|'+str(t)+'|'+'|'.join(f'{x:.1f}' for x in values)+'|')
lines += ['\n旧组NLL约束可能在最终仍超限；有限双对偶惩罚不是严格可行性保证，更不是闭环成功率保证。迁移/保护激活、预测器留出通过不能替代实际结果。',
    f"\n主更新105500；隔离预检8；效果模型额外{result['effect_optimizer_updates']}更新。主epoch累计{result['main_epoch_seconds']/60:.2f}分钟，正式vector累计{result['evaluation_vector_seconds']/60:.2f}分钟。参考评分无单独计时。负墙钟字段原样保留、标无效，不改成0、不重跑。",
    '\n```json\n'+json.dumps(dict(effect_cost=cost,constraints=constraints),indent=2)+'\n```\n']
with (p/'REPORT.md').open('a',encoding='utf8') as f:
    f.write('\n'.join(lines))
print(json.dumps(result,indent=2))
