"""Bind completed local GOAL results and prior receipts; no server mutation."""
import argparse
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('aggregate')
ap.add_argument('snapshot')
ap.add_argument('next_snapshot')
ap.add_argument('--seed41', required=True)
ap.add_argument('--seed42', required=True)
ap.add_argument('--seed43', required=True)
a = ap.parse_args()
root = Path(__file__).resolve().parent
out = Path(a.aggregate)

def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

v = read(out/'LOCAL_AGGREGATE.json')
s = read(Path(a.snapshot)/'EVIDENCE.json')
n = read(Path(a.next_snapshot)/'EVIDENCE.json')
assert v['status'] == 'LOCAL_THREE_SEED_AGGREGATE_PASS' and v['suite'] == 'GOAL'
assert not v['automatic_branch_authorized_by_this_GOAL_comparison']
assert s['frozen_hashes']['count'] == 1209 and not s['frozen_hashes']['failures']
assert n['frozen_hashes']['count'] == 1462 and not n['frozen_hashes']['failures']
q = s['queue_state']
assert q['controller.exit'].strip() == '0' and not q['FAILURE.json']
keys = ['OBJECT_42', 'OBJECT_43', 'GOAL_41', 'GOAL_42', 'GOAL_43']
assert q['RUN_STATUS.json']['completed'] == keys
assert q['RUN_STATUS.json']['status'] == 'SERVER_FINISHED_AWAITING_LOCAL_AUDIT'
assert q['RUN_STATUS.json']['object_gate_passed'] is True
assert not s['processes'], 'Prior queue still has processes'
prefix = n['queue_state']['PRIOR_PREFIX_ACCEPTANCE.json']
assert prefix['status'] == 'SERVER_PREFIX_ACCEPTED_LOCAL_AUDIT_STILL_REQUIRED'
assert prefix['completed'] == keys and prefix['old_gate_passed']
assert prefix['prior_contract_sha256'] == q['DISPATCH_RECEIPT.json']['contract_sha256']
gate = q['GOAL_AGGREGATE.json']
assert gate['suite'] == 'GOAL' and gate['seeds'] == [41, 42, 43]
for field, group in [('mean_auc', 'ALIGN'), ('mean_baseline_auc', 'RA'), ('delta', 'paired_delta')]:
    assert F(gate[field]) == F(v['metrics']['AUC'][group]['mean_fraction'])
above = F(gate['mean_auc']) > F(gate['mean_baseline_auc'])
assert gate['passed'] == v['mean_AUC_above_RA_descriptive'] == above
assert gate['status'] == ('SERVER_EXACT_GATE_PASS' if above else 'SERVER_EXACT_GATE_NOT_PASSED')
packages = dict(OBJECT_42=root/'final_OBJECT_42_20261023_012252',
                OBJECT_43=root/'final_OBJECT_43_20261023_032427')
packages.update({'GOAL_'+str(seed): Path(getattr(a, 'seed'+str(seed))) for seed in (41, 42, 43)})
bindings = {}
cells = []
declines = []
for key, p in packages.items():
    audit = read(p/'LOCAL_FINAL_AUDIT.json')
    receipt = read(p/'launcher/receipt.json')
    assert audit['status'] == 'COMPLETED_LOCAL_INDEPENDENT_PASS'
    assert receipt == s['runs'][key]['receipt']
    assert receipt['key'] == key
    assert receipt['acceptance_sha256'] == audit['acceptance_sha256']
    assert receipt['contract_sha256'] == audit['contract_sha256']
    assert receipt['orchestration_contract_sha256'] == prefix['prior_contract_sha256']
    assert sha(p/'launcher/receipt.json') == prefix['receipts'][key]
    assert sha(p/'launcher/server_independent.json') == receipt['independent_audit_sha256']
    if audit['suite'] == 'GOAL':
        assert read(p/'QUEUE_TASK_COST_AUDIT.json')['local_final_audit_sha256'] == sha(p/'LOCAL_FINAL_AUDIT.json')
    else:
        prior = read(root/'aggregate_OBJECT_20261023_0326/LOCAL_AGGREGATE.json')
        binding = read(root/'aggregate_OBJECT_20261023_0326/GATE_AND_TASK_AUDIT.json')
        assert prior['proofs'][str(audit['seed'])] == sha(p/'LOCAL_FINAL_AUDIT.json')
        assert binding['local_aggregate_sha256'] == sha(root/'aggregate_OBJECT_20261023_0326/LOCAL_AGGREGATE.json')
    bindings[key] = dict(package=str(p.resolve()), local_final_audit_sha256=sha(p/'LOCAL_FINAL_AUDIT.json'), receipt=receipt)
    if audit['suite'] != 'GOAL':
        continue
    seed = audit['seed']
    assert v['proofs'][str(seed)] == sha(p/'LOCAL_FINAL_AUDIT.json')
    assert F(gate['auc'][str(seed)]) == F(audit['metrics']['AUC']['fraction'])
    assert F(gate['baseline_auc'][str(seed)]) == F(audit['baseline']['AUC']['fraction'])
    assert gate['proofs'][key] == receipt['independent_audit_sha256']
    for r in audit['matrix']:
        d = F(r['current'])-F(r['parent'])
        cells.append(dict(seed=seed, stage=r['stage'], task=r['task'], ALIGN=r['current'], RA=r['parent'], delta_fraction=str(d), delta_pp=float(100*d)))
    for task in (6, 7, 8):
        trajectory = [F(r['current']) for r in audit['matrix'] if r['task'] == task]
        declines.append(dict(seed=seed, task=task, peak_fraction=str(max(trajectory)), final_fraction=str(trajectory[-1]), decline_from_peak_pp=float(100*(max(trajectory)-trajectory[-1]))))
assert len(cells) == 30
negative = [r for r in cells if F(r['delta_fraction']) < 0]
result = dict(status='LOCAL_GOAL_AGGREGATE_RECEIPTS_AND_QUEUE_CLOSURE_PASS',
              local_aggregate_sha256=sha(out/'LOCAL_AGGREGATE.json'),
              snapshot_sha256=sha(Path(a.snapshot)/'EVIDENCE.json'),
              next_snapshot_sha256=sha(Path(a.next_snapshot)/'EVIDENCE.json'),
              server_GOAL_comparison_descriptive_only=gate, prior_prefix=prefix,
              completed_local_packages=bindings, prior_controller_exit=0,
              prior_residual_processes=[], all_30_cells=cells,
              negative_cells=negative, decline_from_own_peak=declines,
              automatic_GOAL_branch_authorized=False,
              next_queue_still_requires_independent_acceptance=True,
              no_significance_or_no_forgetting_claim=True)
(out/'QUEUE_CLOSURE_AND_TASK_AUDIT.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
lines = ['', '## 全任务负结果与队列闭合', '',
         '全部30个单元均绑定已验收包和服务器原始记录。以下保留相对同seed RA下降的全部单元。', '',
         '|seed|阶段|任务|ALIGN %|RA %|差 pp|', '|---|---|---|---:|---:|---:|']
for r in negative:
    lines.append(f"|{r['seed']}|{r['stage']}|{r['task']}|{float(F(r['ALIGN'])*100):.1f}|{float(F(r['RA'])*100):.1f}|{r['delta_pp']:+.1f}|")
lines += ['', 'NBT为有符号数值，越小越好；受刚学完时的锚点影响，不能代替最终旧任务成功率或从自身峰值下降的检查。', '',
          'GOAL三seed比较仅描述。冻结服务器文件沿用gate命名，但没有依此授权后续分支；TAC-R/ST的扩展仍分别以自身OBJECT seed41严格高于同seed RA41为准。', '',
          '原队列controller自然exit0、五组receipt、1209冻结hash和零残留与下游接续证明绑定PASS；下游实验仍须自身完整本地独立验收。保留负结果，不作显著性或无遗忘保证。']
with (out/'REPORT.md').open('a', encoding='utf8') as f:
    f.write('\n'.join(lines)+'\n')
print(json.dumps(dict(status=result['status'], negative_cells=len(negative), GOAL_mean=gate['mean_auc'], RA_mean=gate['mean_baseline_auc'], delta=gate['delta']), indent=2))
