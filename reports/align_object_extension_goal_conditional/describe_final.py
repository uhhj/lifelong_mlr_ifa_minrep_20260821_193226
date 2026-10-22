"""Describe accepted seed outcomes, costs and queue receipt without new rollouts."""
import argparse
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('package')
ap.add_argument('snapshot')
a = ap.parse_args()
p = Path(a.package)
v = json.loads((p/'LOCAL_FINAL_AUDIT.json').read_text(encoding='utf8'))
assert v['status'] == 'COMPLETED_LOCAL_INDEPENDENT_PASS'
key = v['suite']+'_'+str(v['seed'])
snapshot = json.loads((Path(a.snapshot)/'EVIDENCE.json').read_text(encoding='utf8'))
receipt = json.loads((p/'launcher/receipt.json').read_text(encoding='utf8'))
assert receipt == snapshot['runs'][key]['receipt']
assert receipt['acceptance_sha256'] == v['acceptance_sha256']
assert receipt['contract_sha256'] == v['contract_sha256']
assert receipt['independent_audit_sha256'] == hashlib.sha256((p/'launcher/server_independent.json').read_bytes()).hexdigest()
assert not snapshot['frozen_hashes']['failures']
cells = []
for r in v['matrix']:
    delta = F(r['current'])-F(r['parent'])
    cells.append(dict(stage=r['stage'], task=r['task'], ALIGN=r['current'], RA=r['parent'], delta_fraction=str(delta), delta_pp=float(100*delta)))
assert len(cells) == 10
finals = [r for r in cells if r['stage'] == 9]
declines = []
for task in (6, 7, 8):
    trajectory = [r for r in cells if r['task'] == task]
    peak = max(F(r['ALIGN']) for r in trajectory)
    last = F(trajectory[-1]['ALIGN'])
    declines.append(dict(task=task, peak_fraction=str(peak), final_fraction=str(last), decline_from_peak_pp=float(100*(peak-last))))
cost = dict(main_updates=v['counts']['main_updates'], BC_valid_exposures=v['counts']['bc_exposures'], BC_masked_exposures=v['counts']['invalid_bc_exposures'], main_epoch_minutes=v['main_training_seconds']/60, formal_vector_minutes=v['formal']['mean_vector_seconds']*20/60, preparation_and_audits_not_in_these_times=True, extra_auxiliary_updates=0)
result = dict(status='QUEUE_RECEIPT_TASK_COST_LOCAL_PASS', suite=v['suite'], seed=v['seed'], local_final_audit_sha256=hashlib.sha256((p/'LOCAL_FINAL_AUDIT.json').read_bytes()).hexdigest(), snapshot_sha256=hashlib.sha256((Path(a.snapshot)/'EVIDENCE.json').read_bytes()).hexdigest(), receipt=receipt, metrics=v['metrics'], baseline=v['baseline'], delta=v['delta'], all_cells=cells, negative_cells=[r for r in cells if F(r['delta_fraction']) < 0], final_cells=finals, decline_from_own_peak=declines, cost=cost, no_significance_claim=True)
(p/'QUEUE_TASK_COST_AUDIT.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
lines=['', '## 全任务单元与成本', '', '下表包含所有单元，包括全部负结果；同seed原RA只读复用，不重评。', '', '|阶段|任务|ALIGN %|RA %|差 pp|', '|---|---|---:|---:|---:|']
for r in cells:
    lines.append(f"|{r['stage']}|{r['task']}|{float(F(r['ALIGN'])*100):.1f}|{float(F(r['RA'])*100):.1f}|{r['delta_pp']:+.1f}|")
lines += ['', f"主epoch用时{cost['main_epoch_minutes']:.6f}分钟；正式vector用时{cost['formal_vector_minutes']:.6f}分钟。准备、导出与审计另计。主更新{cost['main_updates']}；BC有效/屏蔽曝光{cost['BC_valid_exposures']}/{cost['BC_masked_exposures']}；无辅助更新。", '', 'NBT使用有符号数值，越小越好，锚点是各任务刚学完的成功率；不能替代最终旧任务成功率或峰值后下降，也不是无遗忘保证。单seed探索性，不作显著性结论。', '', '本地验收文件与队列receipt/服务器独立审计SHA、同轮只读快照绑定PASS；不代表其他seed完成。']
with (p/'REPORT.md').open('a', encoding='utf8') as f:
    f.write('\n'.join(lines)+'\n')
print(json.dumps(dict(status=result['status'], key=key, negative_cells=len(result['negative_cells']), cost=cost), indent=2))
