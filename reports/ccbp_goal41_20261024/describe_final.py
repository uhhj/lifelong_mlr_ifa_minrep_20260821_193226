"""Local scalar-only paired analysis; no policy forwards or new environments."""
import json,hashlib,statistics,sys
from pathlib import Path
import numpy as np
D=Path(sys.argv[1]);v=json.loads((D/'EVIDENCE.json').read_text());a=json.loads((D/'LOCAL_FINAL_AUDIT.json').read_text())
assert a['status']=='COMPLETE_LOCAL_INDEPENDENT_PASS' and a['final_evidence_sha256']==hashlib.sha256((D/'EVIDENCE.json').read_bytes()).hexdigest()
light={n:json.loads(x['text']) for n,x in v['light'].items() if n.endswith('.json')}
lookup={(r['arm'],r['task'],r['initial_index']):r for r in light['OUTCOMES.json']}
paired={}
for control in 'ABC':
 paired[control]={}
 for t in [8,9]:
  groups={name:[] for name in ['SS','SF','FS','FF']}
  for i in range(20):
   old=bool(lookup[control,t,i]['success']);new=bool(lookup['D',t,i]['success']);groups[('S' if old else 'F')+('S' if new else 'F')].append(i)
  paired[control][str(t)]=groups
diagnostics={}
for arm,rows in v['diagnostics'].items():
 costs=[];tails=[];selected_not_min=0
 for r in rows:
  tc=sum(r['teacher_component_group_costs'][r['teacher_mode']]);sc=[sum(g) for g in r['student_component_group_costs']];c=sc[r['student_mode']];ex=max(0,c-tc-.05);costs.append(c)
  selected_not_min+=r['student_mode']!=int(np.argmin(sc))
  if ex>0:tails.append(dict(sample=r['sample'],sample_sha256=r['sample_sha256'],teacher_cost=tc,student_cost=c,excess=ex,student_group_costs=r['student_component_group_costs'][r['student_mode']],minimum_component_cost=min(sc),selected_mode=r['student_mode']))
 diagnostics[arm]=dict(selected_cost_quantiles={str(q):float(np.quantile(costs,q)) for q in [.5,.9,.95,.99,1]},selected_not_min_count=selected_not_min,budget_exceedances=sorted(tails,key=lambda r:-r['excess']))
epoch_seconds={arm:sum(r['seconds'] for r in light['EPOCHS_'+arm+'.json']) for arm in 'ABCD'}
events=[json.loads(s) for s in v['light']['EVAL_EVENTS.jsonl']['text'].splitlines()]
out=dict(status='COMPLETE_DESCRIPTIVE_ANALYSIS',matrix_percent={arm:{'task8':m['8']*5,'task9':m['9']*5,'mean':sum(m.values())*2.5} for arm,m in a['matrix_success_out_of_20'].items()},paired_initial_indices=paired,offline_tail_diagnostic=diagnostics,epoch_seconds=epoch_seconds,extra_epoch_time_percent_vs_A={arm:100*(epoch_seconds[arm]/epoch_seconds['A']-1) for arm in 'BCD'},evaluation_vector_seconds=sum(r['seconds'] for r in events if r['kind']=='complete'),cpu_diagnostic=dict(windows=v['diagnostic_windows'],batched_calls=v['diagnostic_calls'],elapsed_seconds_including_cpu_audit=v['diagnostic_seconds'],environments=0,optimizer_updates=0),teacher=light['TEACHER.json'],formal_updates=134800,admission_updates=a['admission_updates'],scientific_support=a['scientific_support'],stop_expansion=True,source_final_audit_sha256=hashlib.sha256((D/'LOCAL_FINAL_AUDIT.json').read_bytes()).hexdigest())
(D/'FINAL_ANALYSIS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(out,ensure_ascii=False,indent=2))
