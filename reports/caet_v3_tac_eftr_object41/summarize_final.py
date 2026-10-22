"""Bind final local audit to queue receipt; report task differences and costs."""
import json,hashlib,sys
from pathlib import Path
from fractions import Fraction as F
p=Path(sys.argv[1]);snapshot=Path(sys.argv[2]);variant=sys.argv[3]
def js(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=js(p/'LOCAL_FINAL_AUDIT.json');assert a['status']=='COMPLETED_LOCAL_INDEPENDENT_PASS'
d=js(snapshot/'SNAPSHOT.json');r=d['runs'][variant+'_OBJECT_41'];receipt=r['receipt']
assert receipt['seed']==41 and receipt['method']==a['method']
assert receipt['acceptance_sha256']==sha(p/'current/PROCESS_EXIT_ACCEPTANCE.json')
assert receipt['contract_sha256']==sha(p/'current/CONTRACT.json')
assert r['worker']['supervisor.exit'].strip()=='0'
assert not any('/'+variant+'/source'==x['cwd'][-len('/'+variant+'/source'):] for x in d['processes'])
rows=a['matrix'];cost={}
history=js(p/'current/ra_coreset/epochs.json')
for t in range(6,10):
 stats=[s for f in sorted((p/'current/ra_coreset/auxiliary').glob('task%d_epoch*.json'%t)) for s in js(f)]
 cost[str(t)]=dict(main_updates=len(stats),epoch_seconds=sum(x['seconds'] for x in history if x['task']==t),bc_exposures=sum(x['bc_valid'] for x in stats),invalid_bc_exposures=sum(x['bc_invalid'] for x in stats),auxiliary_head_updates=sum(x.get('head_updates',0) for x in stats),optimizer_trials=sum(x['optimizer_trials'] for x in stats),effect_exposures=sum(x.get('effect_samples',0) for x in stats),cross_anchor_exposures=sum(x.get('cross_anchors',0) for x in stats),partner_forwards=sum(x.get('extra_partner_forwards',0) for x in stats))
out=dict(status='QUEUE_AND_TASK_COST_AUDIT_PASS',receipt=receipt,local_final_sha256=sha(p/'LOCAL_FINAL_AUDIT.json'),snapshot_sha256=sha(snapshot/'SNAPSHOT.json'),cost_by_stage=cost,formal_vector_minutes=a['formal']['mean_vector_seconds']*20/60)
(p/'QUEUE_TASK_COST_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
lines=['','## 原始成功率矩阵（%）','','|阶段|任务|RA|旧v2|本组|本组−RA pp|','|---|---|---:|---:|---:|---:|']
for row in rows:
 values=[100*float(F(row[k])) for k in ('ra','v2_baseline','current')]
 lines.append('|'+str(row['stage'])+'|'+str(row['task'])+'|'+'|'.join('%.6f'%v for v in [*values,values[2]-values[0]])+'|')
lines+=['','## 成本与解释边界','',f'主epoch累计 {sum(x["epoch_seconds"] for x in cost.values())/60:.3f} 分钟，正式vector累计 {out["formal_vector_minutes"]:.3f} 分钟；额外准备、存储、加载和验收另计。辅助/候选计算详见 QUEUE_TASK_COST_AUDIT.json，不称等算力比较。','TAC的跨任务锚点曝光为重复训练次数，不是独立样本数。TAC/Align改变BC目标及有效监督量；TAC相对RA的差异不能在Align完成前归因于效果模块。单seed探索性，不宣称统计显著。NBT按有符号数值越小越好，须同时看OldFinal与各任务绝对成功率。']
with (p/'REPORT.md').open('a',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps(out,indent=2))
