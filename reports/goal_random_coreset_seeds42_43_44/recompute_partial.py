"""Recompute the completed seed42/43 records only; never executes policy code."""
import hashlib, json
from fractions import Fraction as F
from pathlib import Path
R = Path(__file__).resolve().parent
manifest = json.loads((R/'partial_manifest.json').read_text())
for p,v in manifest.items():
    assert hashlib.sha256((R/'partial_evidence'/p).read_bytes()).hexdigest() == v['sha256'], p
out = {'status':'PARTIAL','observed_CST':'2026-10-18 12:32','completed_seeds':[42,43],'seed44':'training completed, evaluation underway; not included','seeds':{}}
for seed in (42,43):
    root = R/'partial_evidence'/f'seed{seed}'
    saved = json.loads((root/'SUMMARY.json').read_text())
    assert (root/'pipeline.exit').read_text().strip() == (root/'supervisor.exit').read_text().strip() == '0'
    arms={}
    for arm in ('random','coreset'):
        r={}
        for stage in range(6,10):
            for task in range(6,stage+1):
                rows=[json.loads(x) for x in (root/arm/'evaluation'/f'trials_stage{stage}_task{task}.jsonl').read_text().splitlines()]
                assert len(rows)==20 and {x['initial_state_index'] for x in rows}==set(range(20))
                assert all((x['seed'],x['stage'],x['task'],x['epoch'])==(seed,stage,task,99) and type(x['success']) is bool for x in rows)
                r[stage,task]=F(sum(x['success'] for x in rows),20)
        metrics={'AUC':sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,'FWT':sum(r[t,t] for t in range(6,10))/4,'NBT':sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3}
        for k,v in metrics.items():assert F(saved['arms'][arm]['metrics'][k]['fraction'])==v
        arms[arm]={'metrics':{k:{'fraction':str(v),'percent':float(100*v)} for k,v in metrics.items()},'matrix':[[float(100*r[s,t]) if t<=s else None for t in range(6,10)] for s in range(6,10)]}
    out['seeds'][seed]=arms
print(json.dumps(out,indent=2))
