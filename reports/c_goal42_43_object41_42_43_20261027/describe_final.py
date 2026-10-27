"""Recompute accepted three-seed C/S/historical comparisons; no execution."""
import json,sys,statistics,hashlib
from pathlib import Path
from fractions import Fraction as F
R=Path(__file__).resolve().parent;W=R.parent;D=Path(sys.argv[1]);sources={}
def read(p):
 sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text(encoding='utf8'))
a=read(D/'LOCAL_FINAL_AUDIT.json');assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE'
oldroot=W/'20261025_ccbp_v3_full/final_readonly_20261025_181226';old=read(oldroot/'LOCAL_FINAL_AUDIT.json');assert old['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE'
cruns={k:dict(matrix=v,metrics=a['metrics'][k]) for k,v in a['matrix'].items()};cruns['GOAL_41']=dict(matrix=old['matrix']['C'],metrics=old['metrics']['C'])
summaries={};sruns={};baselines={};paired={};negative=[]
for suite,folder in [('GOAL','20261026_ccbp_v4_s_goal42_43/final_readonly_20261026_164119'),('OBJECT','20261026_ccbp_v4_s_object41_42_43/final_readonly_20261027_120754')]:
 base=read(W/'20261025_ccbp_v4_full'/(suite+'_THREE_SEED_BASELINES.json'));assert base['status']=='PASS_LOCAL_READONLY_RECOMPUTATION';baselines[suite]=base['summary'] if 'summary' in base else base
 sf=W/folder;s=read(sf/'COMPARISON.json');sa=read(sf/'LOCAL_FINAL_AUDIT.json');assert sa['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE' and s['final_audit_sha256']==sources[str(sf/'LOCAL_FINAL_AUDIT.json')]
 sruns[suite]=s['S'];summaries[suite]={}
 for method,runs in [('C',{str(seed):cruns[suite+'_'+str(seed)] for seed in (41,42,43)}),('S',s['S'])]:
  summaries[suite][method]={k:dict(mean=statistics.mean(x['metrics'][k]['percent'] for x in runs.values()),sample_sd=statistics.stdev(x['metrics'][k]['percent'] for x in runs.values())) for k in ['AUC','FWT','NBT','OldFinal','FinalAll']}
 for seed in (41,42,43):
  case=suite+'_'+str(seed);cp=oldroot/'light/OUTCOMES.json' if case=='GOAL_41' else D/'light'/case/'OUTCOMES.json'
  sp=W/'20261025_ccbp_v4_full/final_readonly_20261026_014654/light/OUTCOMES.json' if case=='GOAL_41' else sf/('seed'+str(seed))/'light/OUTCOMES.json'
  cc=[x for x in read(cp) if x['arm']=='C'];ss=[x for x in read(sp) if x['arm']=='S'];assert len(cc)==len(ss)==200
  cells={}
  for stage in range(6,10):
   for task in range(6,stage+1):
    cr=sorted([x for x in cc if (x['stage'],x['task'])==(stage,task)],key=lambda x:x['initial_index']);sr=sorted([x for x in ss if (x['stage'],x['task'])==(stage,task)],key=lambda x:x['initial_index'])
    assert len(cr)==len(sr)==20
    assert all((x['initial_index'],x['initial_file_sha256'],x['initial_sha256'],x['noise_sha256'],x['env_seed'])==(y['initial_index'],y['initial_file_sha256'],y['initial_sha256'],y['noise_sha256'],y['env_seed']) for x,y in zip(cr,sr))
    sfail=[x['initial_index'] for x,y in zip(cr,sr) if y['success'] and not x['success']];fimprove=[x['initial_index'] for x,y in zip(cr,sr) if not y['success'] and x['success']];delta=5*(len(fimprove)-len(sfail))
    cells['%d_%d'%(stage,task)]=dict(C_minus_S_pp=delta,S_success_C_failure=sfail,S_failure_C_success=fimprove)
    assert sum(x['success'] for x in cr)==cruns[case]['matrix']['%d_%d'%(stage,task)]['successes']
    if delta<0:negative.append(dict(case=case,stage=stage,task=task,C_minus_S_pp=delta))
  paired[case]=cells
out=dict(status='PASS_ACCEPTED_COMPARISON',C=cruns,S=sruns,summary=summaries,paired_C_minus_S=paired,all_negative_C_minus_S=negative,historical_baselines=baselines,source_sha256=sources,limits=['Single common method change is not isolated: S adds fixed successful on-policy retention and training interaction','Historical MLR/RA/ALIGN have different action-noise protocol','Development seeds/old initial states; sample SD is not a confidence interval','GOAL41 C reused accepted v3 current-card control; hardware differs across historical runs'])
(D/'COMPARISON.json').write_text(json.dumps(out,indent=2),encoding='utf8')
lines=['# C完整GOAL/OBJECT三seed扩展最终结果','','真实独立验收完成；所有负单元、分任务矩阵、配对SF/FS见COMPARISON.json。','', '|Suite|Method|AUC|FWT|NBT↓|OldFinal|FinalAll|','|---|---|---:|---:|---:|---:|---:|']
for suite,methods in summaries.items():
 for method,mm in methods.items():lines.append('|'+suite+'|'+method+'|'+'|'.join('%.4f ± %.4f'%(mm[k]['mean'],mm[k]['sample_sd']) for k in ['AUC','FWT','NBT','OldFinal','FinalAll'])+'|')
lines+=['','每seed20回合/cell，均值±样本标准差。C无需训练交互，S每seed额外40次训练采集，不称等交互或唯一机制归因。硬件差异只记录，开发seed重复探索不称显著。','', '完整机制分析仍需结合RECOMPUTED逐窗口和闭环负单元完成；此脚本仅统计汇总，不把离线cost下降等同于闭环收益。']
(D/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('\n'.join(lines))
