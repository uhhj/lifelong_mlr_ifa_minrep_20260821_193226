"""Additional matched-window and paired-episode analysis of accepted scalar records."""
import json, hashlib, sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent; D=Path(sys.argv[1])
js=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=js(D/'LOCAL_FINAL_AUDIT.json');assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE'
v=js(D/'EVIDENCE.json');out=[];pairs=[];sources={}
for seed in (41,42,43):
 L={k.split('/',1)[1]:json.loads(x['text']) for k,x in v['light'].items() if k.startswith(f'seed{seed}/') and k.endswith('.json')}
 for kind in ('expert','retain'):
  rows={}
  for stage in (8,9):
   p=D/f'diagnostics/s{seed}_{kind}_S_{stage}.json';sources[str(p.relative_to(D))]=sha(p)
   rr=js(p);key=lambda x:(x['task'],x['sample_sha256'] if kind=='expert' else x['feature_sha256'])
   rows[stage]={key(x):x for x in rr};assert len(rows[stage])==len(rr)
  for task in (6,7):
   keys=sorted(k for k in rows[8] if k[0]==task)
   assert set(keys)=={k for k in rows[9] if k[0]==task}
   stats={}
   for stage in (8,9):
    costs=[];regrets=[]
    for k in keys:
     row=rows[stage][k];g=np.asarray(row['component_group_costs' if kind=='expert' else 'component_group_distances'])
     if kind=='expert':
      g=g*np.square(L[f'TRAIN_INPUT_S_{stage}.json']['scale'])/np.square(L['TRAIN_INPUT_S_9.json']['scale'])
     cc=g.sum(-1);mode=int(np.argmax(row['logits']));assert mode==row['student_mode']
     costs.append(float(cc[mode]));regrets.append(float(cc[mode]-cc.min()))
    stats[str(stage)]=dict(mean_execution_cost=float(np.mean(costs)),mean_regret=float(np.mean(regrets)))
   out.append(dict(seed=seed,kind=kind,task=task,windows=len(keys),identity='sample SHA' if kind=='expert' else 'cached feature SHA',
     scale='stage9 common expert scale' if kind=='expert' else 'unchanged task acquisition cache scale',statistics=stats))
 records=L['OUTCOMES.json']
 for task in (6,7,8):
  p={x['initial_index']:x for x in records if x['stage']==8 and x['task']==task}
  q={x['initial_index']:x for x in records if x['stage']==9 and x['task']==task}
  assert set(p)==set(q)==set(range(20))
  assert all(all(p[i][k]==q[i][k] for k in ['initial_file_sha256','initial_sha256','noise_sha256','env_seed']) for i in p)
  pairs.append(dict(seed=seed,task=task,stage8_success=sum(x['success'] for x in p.values()),stage9_success=sum(x['success'] for x in q.values()),
    SF=sum(p[i]['success'] and not q[i]['success'] for i in p),FS=sum(not p[i]['success'] and q[i]['success'] for i in p)))
result=dict(status='PASS_ACCEPTED_SCALAR_RECOMPUTATION',matched_windows=out,stage8_to9_pairs=pairs,source_sha256=sources,
 final_audit_sha256=sha(D/'LOCAL_FINAL_AUDIT.json'),new_forward=0,new_environment=0,new_optimizer=0,
 limits=['Expert costs rescaled to stage9 scale before comparing the exact same sample identities.',
 'Cached-feature comparisons do not establish matching physical states during formal trajectories.',
 'No controlled OBJECT C/O arms or same-noise historical baseline comparison; no single-action causal attribution.'])
(D/'MATCHED_DIAGNOSTIC_ANALYSIS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result,indent=2))
