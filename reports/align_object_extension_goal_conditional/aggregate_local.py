"""Exact three-seed local metric aggregation and sample SD; no scientific retry."""
import argparse,json,hashlib,statistics,datetime
from pathlib import Path
from fractions import Fraction as F
ap=argparse.ArgumentParser();ap.add_argument('--suite',choices=['OBJECT','GOAL'],required=True);ap.add_argument('--seed41',required=True);ap.add_argument('--seed42',required=True);ap.add_argument('--seed43',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
rows={};proof={}
for seed in [41,42,43]:
 p=Path(getattr(a,'seed'+str(seed)));v=json.loads((p/'LOCAL_FINAL_AUDIT.json').read_text());assert v['status']=='COMPLETED_LOCAL_INDEPENDENT_PASS' and v['seed']==seed
 assert v['suite']==a.suite
 rows[seed]=v;proof[str(seed)]=hashlib.sha256((p/'LOCAL_FINAL_AUDIT.json').read_bytes()).hexdigest()
out=dict(status='LOCAL_THREE_SEED_AGGREGATE_PASS',suite=a.suite,seeds=[41,42,43],observed_at=datetime.datetime.now().astimezone().isoformat(),proofs=proof,metrics={})
for k in ['AUC','FWT','NBT','OldFinal','FinalAll']:
 current=[F(rows[s]['metrics'][k]['fraction']) for s in rows]
 baseline=[F(rows[s].get('baseline',rows[s].get('ra'))[k]['fraction']) for s in rows]
 delta=[x-y for x,y in zip(current,baseline)]
 def pack(x):return dict(per_seed_fraction=[str(z) for z in x],mean_fraction=str(sum(x)/3),mean_percent=float(sum(x)/3*100),sample_sd_percent=statistics.stdev([float(z*100) for z in x]))
 out['metrics'][k]=dict(ALIGN=pack(current),RA=pack(baseline),paired_delta=pack(delta))
auc=out['metrics']['AUC'];above_ra=F(auc['ALIGN']['mean_fraction'])>F(auc['RA']['mean_fraction'])
if a.suite=='OBJECT':out['strict_mean_AUC_gate_passed']=above_ra
else:
 out['mean_AUC_above_RA_descriptive']=above_ra
 out['automatic_branch_authorized_by_this_GOAL_comparison']=False
p=Path(a.output);p.mkdir(exist_ok=False);(p/'LOCAL_AGGREGATE.json').write_text(json.dumps(out,indent=2)+'\n')
lines=['# '+a.suite+' ALIGN 三seed汇总','','三seed等权；SD为样本SD(ddof1)，配对差SD独立计算。保留全部负结果，不作自动显著结论。','','|指标|ALIGN 均值±SD %|RA 均值±SD %|配对差 pp|','|---|---:|---:|---:|']
for k,v in out['metrics'].items():lines.append('|'+k+'|'+f"{v['ALIGN']['mean_percent']:.6f} ± {v['ALIGN']['sample_sd_percent']:.6f}|{v['RA']['mean_percent']:.6f} ± {v['RA']['sample_sd_percent']:.6f}|{v['paired_delta']['mean_percent']:+.6f}| ")
if a.suite=='OBJECT':lines+=['','唯一自动分支门槛：三seed平均AUC严格大于同seed RA。判定：'+str(out['strict_mean_AUC_gate_passed'])]
else:lines+=['','GOAL三seed均值与RA的比较仅作描述，不设后续自动分支或新seed授权；TAC-R/ST各自扩展仍只看自己的OBJECT seed41与同seed RA。']
(p/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print(json.dumps(out,indent=2))
