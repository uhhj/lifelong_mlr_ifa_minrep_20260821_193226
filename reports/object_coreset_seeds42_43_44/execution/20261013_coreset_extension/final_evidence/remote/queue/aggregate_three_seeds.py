"""Aggregate only completed, independently accepted runs; never execute environments."""
import json,statistics,hashlib,time
from pathlib import Path
from fractions import Fraction as F
W=Path(__file__).resolve().parent
allseeds={};inputs={}
for seed in (42,43,44):
    O=Path(f'/data/mlr_ifa/experiments/Coreset_paper_OBJECT_seed{seed}_20261013')
    for name in ('pipeline.exit','supervisor.exit'):assert (O/name).read_text().strip()=='0'
    assert json.loads((O/'INDEPENDENT_ACCEPTANCE.json').read_text())['status']=='PASS'
    summary=json.loads((O/'SUMMARY.json').read_text());assert summary['seed']==seed and summary['formal_records']==400
    seeds={}
    for arm in ('random','coreset'):
        r={}
        for i in range(6,10):
            for j in range(6,i+1):
                p=O/arm/'evaluation'/f'trials_stage{i}_task{j}.jsonl'
                inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
                rows=[json.loads(x) for x in p.read_text().splitlines()]
                assert len(rows)==20 and {x['trial'] for x in rows}==set(range(20))
                assert all(x['seed']==seed for x in rows)
                r[i,j]=F(sum(x['success'] for x in rows),20)
        m={'FWT':sum(r[j,j] for j in range(6,10))/4,
           'NBT':sum(sum(r[j,j]-r[i,j] for i in range(j+1,10))/(9-j) for j in range(6,9))/3,
           'AUC':sum(sum(r[i,j] for i in range(j,10))/(10-j) for j in range(6,10))/4}
        assert all(str(v)==summary['arms'][arm]['metrics'][k]['fraction'] for k,v in m.items())
        seeds[arm]={'metrics_percent':{k:float(v*100) for k,v in m.items()},'matrix':summary['arms'][arm]['matrix']}
    allseeds[str(seed)]=seeds
stats={}
for group in ('random','coreset','paired_C_minus_R'):
    stats[group]={}
    for k in ('AUC','FWT','NBT'):
        vals=[v[group]['metrics_percent'][k] if group!='paired_C_minus_R' else v['coreset']['metrics_percent'][k]-v['random']['metrics_percent'][k] for v in allseeds.values()]
        stats[group][k]={'values':vals,'mean':statistics.mean(vals),'sample_sd_ddof1':statistics.stdev(vals),'population_sd_ddof0':statistics.pstdev(vals)}
out=dict(status='AGGREGATE_PASS_AWAITING_QUEUE_EXIT',seeds=allseeds,stats=stats,formal_records=1200,epochs=2400,optimizer_steps=633000,inputs=inputs,unix=time.time(),limits='Exploratory extension authorized after observing seed42; n=3 descriptive results, not an independent confirmatory study. Pretrain modality unknown, hardware and scheduler disclosures persist.')
(W/'THREE_SEED_SUMMARY.json').write_text(json.dumps(out,indent=2)+'\n')
lines=['# OBJECT Random vs Coreset: seeds42/43/44','', 'Descriptive three-seed results; sample SD uses ddof=1. Extension was authorized after seed42 results.','', '| Group | AUC mean ± SD | FWT mean ± SD | NBT mean ± SD |','|---|---:|---:|---:|']
for group,d in stats.items():lines.append('| '+group+' | '+' | '.join(f"{d[k]['mean']:.6f} ± {d[k]['sample_sd_ddof1']:.6f}" for k in ('AUC','FWT','NBT'))+' |')
(W/'THREE_SEED_REPORT.md').write_text('\n'.join(lines)+'\n')
print('THREE_SEED_AGGREGATE_PASS: 1200 formal records')
