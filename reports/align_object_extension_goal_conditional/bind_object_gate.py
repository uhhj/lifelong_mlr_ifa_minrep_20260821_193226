"""Bind accepted local OBJECT packages to the exact server gate; no remote work."""
import argparse
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('aggregate')
ap.add_argument('snapshot')
ap.add_argument('--seed41', required=True)
ap.add_argument('--seed42', required=True)
ap.add_argument('--seed43', required=True)
a = ap.parse_args()
out = Path(a.aggregate)
aggregate = json.loads((out/'LOCAL_AGGREGATE.json').read_text(encoding='utf8'))
snapshot = json.loads((Path(a.snapshot)/'EVIDENCE.json').read_text(encoding='utf8'))
gate = snapshot['queue_state']['OBJECT_GATE.json']
assert aggregate['status'] == 'LOCAL_THREE_SEED_AGGREGATE_PASS'
assert gate['status'] == 'SERVER_EXACT_GATE_PASS'
assert not snapshot['frozen_hashes']['failures']
assert snapshot['frozen_hashes']['count'] == 1209
assert F(gate['mean_auc']) == F(aggregate['metrics']['AUC']['ALIGN']['mean_fraction'])
assert F(gate['mean_baseline_auc']) == F(aggregate['metrics']['AUC']['RA']['mean_fraction'])
assert F(gate['delta']) == F(aggregate['metrics']['AUC']['paired_delta']['mean_fraction'])
assert gate['passed'] == aggregate['strict_mean_AUC_gate_passed'] is True
rows = []
negative = []
costs = {}
for seed in (41, 42, 43):
    p = Path(getattr(a, 'seed'+str(seed)))
    raw = (p/'LOCAL_FINAL_AUDIT.json').read_bytes()
    v = json.loads(raw)
    assert v['status'] == 'COMPLETED_LOCAL_INDEPENDENT_PASS'
    assert hashlib.sha256(raw).hexdigest() == aggregate['proofs'][str(seed)]
    assert v['metrics']['AUC']['fraction'] == gate['auc'][str(seed)]
    baseline = v.get('baseline', v.get('ra'))
    assert baseline['AUC']['fraction'] == gate['baseline_auc'][str(seed)]
    if seed != 41:
        receipt = json.loads((p/'launcher/receipt.json').read_text(encoding='utf8'))
        assert receipt == snapshot['runs']['OBJECT_'+str(seed)]['receipt']
        assert receipt['independent_audit_sha256'] == gate['proofs']['OBJECT_'+str(seed)]
        assert receipt['acceptance_sha256'] == v['acceptance_sha256']
        assert hashlib.sha256((p/'launcher/server_independent.json').read_bytes()).hexdigest() == receipt['independent_audit_sha256']
    else:
        assert snapshot['prefix_independent']['metrics'] == v['metrics']
        assert snapshot['prefix_independent']['baseline'] == baseline
    costs[str(seed)] = dict(counts=v['counts'], main_training_seconds=v['main_training_seconds'], formal=v['formal'])
    for cell in v['matrix']:
        parent = cell.get('parent', cell.get('ra'))
        delta = F(cell['current'])-F(parent)
        row = dict(seed=seed, stage=cell['stage'], task=cell['task'], ALIGN=cell['current'], RA=parent, delta_fraction=str(delta), delta_pp=float(100*delta))
        rows.append(row)
        if delta < 0:
            negative.append(row)
assert len(rows) == 30
result = dict(status='LOCAL_GATE_RECEIPT_MATRIX_BINDING_PASS', server_gate=gate,
              local_aggregate_sha256=hashlib.sha256((out/'LOCAL_AGGREGATE.json').read_bytes()).hexdigest(),
              snapshot_sha256=hashlib.sha256((Path(a.snapshot)/'EVIDENCE.json').read_bytes()).hexdigest(),
              all_30_cells=rows, negative_cells=negative, per_seed_cost=costs,
              no_significance_claim=True, GOAL_automatic_branch=snapshot['queue_state']['RUN_STATUS.json'])
(out/'GATE_AND_TASK_AUDIT.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
lines=['', '## 所有逐任务负结果', '', '保留三seed全部30个单元；以下列出相对同seed RA下降的单元，不通过挑选任务解释总体收益。', '', '|seed|阶段|任务|ALIGN %|RA %|差 pp|', '|---|---|---|---:|---:|---:|']
for r in negative:
    lines.append(f"|{r['seed']}|{r['stage']}|{r['task']}|{float(F(r['ALIGN'])*100):.1f}|{float(F(r['RA'])*100):.1f}|{r['delta_pp']:+.1f}|")
lines += ['', 'NBT使用有符号数值、越小越好，但以刚学完时的成功率为锚点。它不能代替最终旧任务成功率，也不构成无遗忘保证。', '', '三seed均值AUC通过用户冻结门槛；OldFinal均值仍比RA低0.555556pp。seed43的FWT、OldFinal和FinalAll分别低3.75、5和2.5pp。这里只作探索性描述，不作显著性结论。', '', '服务器门槛与本地精确分数/receipt/30个任务单元绑定PASS。GOAL自动分支已进入GOAL_41；后续仍须各seed完整独立验收。']
with (out/'REPORT.md').open('a', encoding='utf8') as f:
    f.write('\n'.join(lines)+'\n')
print(json.dumps(dict(status=result['status'], negative_cells=len(negative), gate_mean=gate['mean_auc'], delta=gate['delta']), indent=2))
