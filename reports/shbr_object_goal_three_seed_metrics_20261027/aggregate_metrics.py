"""Recompute fixed three-seed OBJECT/GOAL metrics from accepted full matrices."""
import datetime,hashlib,json,statistics
from fractions import Fraction as F
from pathlib import Path
R=Path(__file__).resolve().parent;W=R.parents[1]
sources={};runs=[];summary={}
def read(rel):
 p=W/rel;sources[rel]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text(encoding='utf8'))
def metrics(matrix):
 r={tuple(map(int,k.split('_'))):F(str(x['percent'] if isinstance(x,dict) else x))/100 for k,x in matrix.items()}
 assert set(r)=={(s,t) for s in range(6,10) for t in range(6,s+1)}
 assert all(0<=x<=1 and (x*20).denominator==1 for x in r.values())
 return dict(AUC=sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,
 FWT=sum(r[t,t] for t in range(6,10))/4,
 NBT=sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3)
def add(suite,method,seed,x):
 m=metrics(x['matrix'])
 for k,z in m.items():assert abs(float(z*100)-x['metrics'][k]['percent'])<1e-10
 runs.append(dict(suite=suite,method=method,seed=seed,metrics={k:dict(fraction=str(v),percent=float(v*100)) for k,v in m.items()}))
for suite,relative in [('OBJECT','execution/20261026_ccbp_v4_s_object41_42_43/final_readonly_20261027_120754'),('GOAL','execution/20261026_ccbp_v4_s_goal42_43/final_readonly_20261026_164119')]:
 baseline=read(f'execution/20261025_ccbp_v4_full/{suite}_THREE_SEED_BASELINES.json')
 assert baseline['status']=='PASS_LOCAL_READONLY_RECOMPUTATION'
 c=read(relative+'/COMPARISON.json');a=read(relative+'/LOCAL_FINAL_AUDIT.json')
 assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE' and c['final_audit_sha256']==sources[relative+'/LOCAL_FINAL_AUDIT.json']
 for x in baseline['runs']:add(suite,x['method'],x['seed'],x)
 for seed,x in c['S'].items():add(suite,'RA_ALIGN_SHBR',int(seed),x)
 if suite=='GOAL':
  original=read('execution/20261025_ccbp_v4_full/final_readonly_20261026_014654/LOCAL_FINAL_AUDIT.json')
  assert original['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE'
  assert c['S']['41']['metrics']==original['metrics']['S']
 summary[suite]={}
 for method in ['MLR_IFA','RA','RA_ALIGN','RA_ALIGN_SHBR']:
  rows=[x for x in runs if x['suite']==suite and x['method']==method];assert sorted(x['seed'] for x in rows)==[41,42,43]
  summary[suite][method]={k:dict(mean=statistics.mean(x['metrics'][k]['percent'] for x in rows),sample_sd=statistics.stdev(x['metrics'][k]['percent'] for x in rows)) for k in ['AUC','FWT','NBT']}
out=dict(status='PASS_LOCAL_METRIC_AGGREGATION',seeds=[41,42,43],units='percentage points; mean and sample SD across 3 seeds',
 naming={'MLR_IFA':'原MLR+IFA','RA':'+RA','RA_ALIGN':'+RA+ALIGN','RA_ALIGN_SHBR':'+RA+ALIGN+SHBR (accepted S method; user label)'},
 summary=summary,per_seed=runs,source_sha256=sources,new_training=0,new_forward=0,new_environment=0,
 definitions={'AUC':'Each task averaged over its acquired and subsequent stages, then averaged across tasks.',
 'FWT':'Project definition: mean diagonal acquisition success, not zero-shot transfer.',
 'NBT':'Project definition: acquisition success minus subsequent success, averaged within task and then across 3 old tasks; lower is better.'},
 limits=['Historical baselines do not share verified per-trajectory action noise with S; S uses additional training collection.',
 'OBJECT original seed43 uses frozen RA-protocol repeat1; no posthoc replacement by first run.',
 'Three development seeds and 20 episodes/cell; SD is descriptive, not significance or a confidence interval.'])
(R/'METRICS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
lines=['# OBJECT与GOAL三seed指标汇总','','seed41、42、43，均值±样本标准差，单位%。SHBR按用户命名对应已验收S方法；所有“+”均累积在原MLR+IFA之上。','','|方法|OBJECT AUC ↑|OBJECT FWT ↑|OBJECT NBT ↓|GOAL AUC ↑|GOAL FWT ↑|GOAL NBT ↓|','|---|---:|---:|---:|---:|---:|---:|']
for m,label in [('MLR_IFA','原MLR+IFA'),('RA','+RA'),('RA_ALIGN','+RA+ALIGN'),('RA_ALIGN_SHBR','+RA+ALIGN+SHBR')]:
 lines.append('|'+label+'|'+'|'.join(f"{summary[s][m][k]['mean']:.2f} ± {summary[s][m][k]['sample_sd']:.2f}" for s in ['OBJECT','GOAL'] for k in ['AUC','FWT','NBT'])+'|')
lines+=['','FWT沿用项目定义：任务获得阶段成功率均值；NBT越低越好，负值表示按该定义后续平均表现高于获得阶段。标准差是跨seed样本标准差，不是置信区间。','','历史基线与S不是同动作噪声、等交互对照。OBJECT原MLR+IFA seed43沿用冻结RA协议的repeat1，不依据本次结果替换。数据为项目复现，不是原论文表格。完整逐seed值、精确分数及来源SHA见METRICS.json。']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
(R/'RUN_STATUS.json').write_text(json.dumps(dict(status=out['status'],observation=datetime.datetime.now().astimezone().isoformat(),new_training=0,new_forward=0,new_environment=0),indent=2)+'\n',encoding='utf8')
print('\n'.join(lines))
