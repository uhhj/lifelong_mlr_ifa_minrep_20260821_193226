"""Read-only decomposition of recorded five-seed metrics; no ML imports."""
import json, statistics
from pathlib import Path
R = Path(__file__).resolve().parent
s = json.loads((R / 'FINAL_RESULTS.json').read_text(encoding='utf8'))
out = {'source': 'FINAL_RESULTS.json', 'primary_seed43': 'repeat1_post_result_choice', 'tasks': {}}
for task in range(6, 10):
    row = {}
    for arm in ('primary_baseline', 'ZR'):
        row[arm] = {
            'at_learning': statistics.mean(v[arm][f'{task},{task}'] for v in s['matrices_percent'].values()),
            'final': statistics.mean(v[arm][f'9,{task}'] for v in s['matrices_percent'].values()),
            'task_average': statistics.mean(statistics.mean(v[arm][f'{stage},{task}'] for stage in range(task, 10)) for v in s['matrices_percent'].values()),
        }
    row['total_AUC_contribution_pp'] = (row['ZR']['task_average'] - row['primary_baseline']['task_average']) / 4
    out['tasks'][task] = row
assert abs(sum(v['total_AUC_contribution_pp'] for v in out['tasks'].values()) + 3.375) < 1e-10
print(json.dumps(out, indent=2))
