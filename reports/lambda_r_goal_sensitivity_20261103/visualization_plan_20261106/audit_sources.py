"""Read-only inventory for visualization planning; no models or environments loaded."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import datetime

HERE = Path(__file__).resolve().parent
EXEC = HERE.parents[1]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

sources = {
    'Q_vs_S': EXEC/'20261031_full_q_vs_latest_s_two_suites/final_readonly_20261101_032229/PRIMARY_S_COMPARISON.json',
    'C_equal_optimizer_vs_S': EXEC/'20261030_equal_optimizer_budget_cs_two_suites/final_readonly_20261031_065009/PRIMARY_S_COMPARISON.json',
    'GOAL_lambda': EXEC/'20261103_lambda_r_goal_sensitivity/final_readonly_20261104_122613/PRIMARY_S_COMPARISON.json',
    'OBJECT_lambda': EXEC/'20261101_lambda_r_object_sensitivity/final_acceptance_20261103_v1/final_readonly_20261103_124306/PRIMARY_S_COMPARISON.json',
}
inventory = {}
for label, p in sources.items():
    data = json.loads(p.read_text(encoding='utf8'))
    cases = {}
    for name, case in data['cases'].items():
        rows = case['paired_outcomes']
        counts = Counter((r['stage'], r['task']) for r in rows)
        assert set(counts) == {(s,t) for s in range(6,10) for t in range(6,s+1)}
        assert len(rows) == 200 and set(counts.values()) == {20}
        for stage, task in counts:
            rr = [r for r in rows if (r['stage'],r['task']) == (stage,task)]
            assert sorted(r['initial_index'] for r in rr) == list(range(20))
            assert all(r['initial_sha256'] and r['noise_sha256'] for r in rr)
        cases[name] = dict(paired_rows=len(rows), observed_cells=len(counts), episodes_per_cell=20)
    inventory[label] = dict(path=str(p), sha256=sha(p), cases=cases)

light = sources['GOAL_lambda'].parent/'light/L005_41'
gates = []
gate_sources = {}
for stage in (7,8,9):
    p = light/f'GATES_stage{stage}.jsonl'
    rows = [json.loads(line) for line in p.read_text(encoding='utf8').splitlines() if line]
    for r in rows:
        c,q = r['C'],r['Q']
        checks = dict(bc_new=q['bc_new']<=c['bc_new'],exec_new=q['exec_new']<=c['exec_new'],retain_improved=q['retain_support']<c['retain_support'])
        assert checks == r['checks']
        assert r['accepted'] == ('Q' if all(checks.values()) else 'C')
    gate_sources[p.name] = dict(sha256=sha(p), blocks=len(rows))
    gates.extend(rows)
assert len(gates)==6700
grad_path = light/'GRADIENTS_S.json'
gradients = json.loads(grad_path.read_text(encoding='utf8'))
assert len(gradients)==12

paper_sources = []
for folder in ['8WVD3WFZ','2N5P24ZM']:
    p = next((Path('C:/Users/Lenovo/Zotero/storage')/folder).glob('*.pdf'))
    paper_sources.append(dict(filename=p.name,sha256=sha(p)))
result = dict(
    status='PASS_LOCAL_VISUALIZATION_DATA_READINESS',
    observed=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    comparisons=inventory,
    example_gate_audit=dict(case='GOAL_L005_41',sources=gate_sources,blocks=6700,decisions_recomputed=True),
    gradient_example=dict(sha256=sha(grad_path),records=len(gradients),meaning='Sparse selected training diagnostics, not dense epochs or independent validation'),
    paper_sources=paper_sources,
    ready=['stage_task_success_heatmaps','success_curves','lambda_sensitivity','gate_candidate_tradeoffs','sparse_gradient_diagnostics'],
    pending=['Common fixed-input manifest and checkpoint availability verification for new hidden-feature/output-drift extraction'],
    limitations=['No remote asset availability check in this audit','No model forward passes or feature extraction performed','No new environment, training, optimizer calls or monitor changes','Case gate audit is explicitly one example, not a new all-run scientific acceptance'],
)
(HERE/'READINESS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=result['status'],comparison_sources=len(inventory),cases=sum(len(x['cases']) for x in inventory.values()),example_gate_blocks=len(gates)),ensure_ascii=False))
