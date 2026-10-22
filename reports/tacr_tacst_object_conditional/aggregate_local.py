"""Three-seed reporting; expansion was decided only by each seed41 gate."""
import argparse,json,hashlib,statistics,datetime
from pathlib import Path
from fractions import Fraction as F
ap=argparse.ArgumentParser();ap.add_argument('--suite',choices=['OBJECT'],default='OBJECT');ap.add_argument('--variant',choices=['TACR','TACRST'],required=True)
for s in (41,42,43):ap.add_argument('--seed'+str(s),required=True)
ap.add_argument('--output',required=True);a=ap.parse_args();rows={};proof={}
for seed in (41,42,43):
 p=Path(getattr(a,'seed'+str(seed)));v=json.loads((p/'LOCAL_FINAL_AUDIT.json').read_text());assert v['status']=='COMPLETED_LOCAL_INDEPENDENT_PASS' and v['seed']==seed and v['variant']==a.variant
 rows[seed]=v;proof[str(seed)]=hashlib.sha256((p/'LOCAL_FINAL_AUDIT.json').read_bytes()).hexdigest()
out=dict(status='LOCAL_THREE_SEED_AGGREGATE_PASS',suite='OBJECT',variant=a.variant,seeds=[41,42,43],observed_at=datetime.datetime.now().astimezone().isoformat(),proofs=proof,metrics={})
for k in ('AUC','FWT','NBT','OldFinal','FinalAll'):
 current=[F(rows[s]['metrics'][k]['fraction']) for s in rows];base=[F(rows[s]['baseline'][k]['fraction']) for s in rows];delta=[x-y for x,y in zip(current,base)]
 def pack(x):return dict(per_seed_fraction=[str(z) for z in x],mean_fraction=str(sum(x)/3),mean_percent=float(sum(x)/3*100),sample_sd_percent=statistics.stdev(float(z*100) for z in x))
 out['metrics'][k]=dict(current=pack(current),RA=pack(base),paired_delta=pack(delta))
out['seed41_expansion_gate_passed']=F(rows[41]['metrics']['AUC']['fraction'])>F(701,960)
p=Path(a.output);p.mkdir(exist_ok=False);(p/'LOCAL_AGGREGATE.json').write_text(json.dumps(out,indent=2)+'\n')
lines=['# OBJECT '+a.variant+' three-seed report','','Equal seed weights, sample SD (ddof=1). All negative results retained; no automatic significance claim. Expansion used only seed41 > RA41 701/960.','','|Metric|Current mean +/- SD %|RA mean +/- SD %|Paired delta pp|','|---|---:|---:|---:|']
for k,v in out['metrics'].items():lines.append('|'+k+'|'+f"{v['current']['mean_percent']:.6f} +/- {v['current']['sample_sd_percent']:.6f}|{v['RA']['mean_percent']:.6f} +/- {v['RA']['sample_sd_percent']:.6f}|{v['paired_delta']['mean_percent']:+.6f}|")
(p/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print(json.dumps(out,indent=2))
