"""Independent local replay of scalar candidate selection and memory cursors."""
import json,sys,hashlib,math
from pathlib import Path
from collections import Counter
p=Path(sys.argv[1])
def js(p):return json.loads(p.read_text(encoding='utf8'))
d=js(p/'MANIFEST.json');result={}
for name,m in d['manifest'].items():
 raw=(p/name).read_bytes();assert len(raw)==m['bytes'] and hashlib.sha256(raw).hexdigest()==m['sha256']
for t in sorted({h['task'] for h in d['history'] if h['task']>6}):
 rows=js(p/'ra_coreset/dataset_manifests'/('task%d.json'%t))['rows'];n=len(rows);cursor=0;c=Counter();margins=[];selectedR=[];baseR=[];switch=[];gripper=[]
 tasks={j:sorted([r['index'] for r in rows if r['task']==j],key=lambda i:rows[i]['sample']) for j in range(6,t)}
 old=[v[k] for k in range(max(map(len,tasks.values()))) for j,v in sorted(tasks.items()) if k<len(v)]
 for j,ids in tasks.items():
  meta=js(p/'teachers'/('task%d.json'%j));storage=js(p/'ra_coreset/storage'/('task%d.json'%j));assert meta['count']==len(ids)==storage['K']
  assert meta['teacher']=='own_task_ep99' and meta['eval_mode'] and meta['gradient'] is False
  assert {rows[i]['path'].split('/')[-1] for i in ids}=={x['destination'] for x in storage['selected']}
  assert d['teacher_binding'][str(j)]['teacher_sha256']==meta['sha256']
 epoch_checks=[]
 for h in [h for h in d['history'] if h['task']==t]:
  stats=js(p/'ra_coreset/auxiliary'/('task%d_epoch%02d.json'%(t,h['epoch'])));assert len(stats)==h['batches']==math.ceil(n/10)
  assert sorted(i for s in stats for i in s['batch_ids'])==list(range(n))
  assert [s['step'] for s in stats]==list(range(h['steps']-len(stats),h['steps']))
  for s in stats:
   ids=[old[(cursor+k)%len(old)] for k in range(min(32,len(old)))];cursor=(cursor+len(ids))%len(old)
   assert ids==s['old_check_ids'] and cursor==s['cursor'] and len(ids)==s['old_check_count']
   assert s['optimizer_trials']==3 and s['committed_updates']==1
   scores=s['scores'];assert len(scores)==3 and all(v is None or math.isfinite(v) for score in scores for v in score.values())
   assert all(abs(z['R']-(z['KL_per_dim']+.1*z['switch']))<1e-8 for z in scores)
   c0=scores[0]['C'];valid=[i for i,z in enumerate(scores) if c0 is None or z['C']<=c0+.02];assert 0 in valid
   eligible=[i for i in valid if scores[i]['R']<=.1];k=min(eligible or valid,key=lambda i:(scores[i]['R'],scores[i]['C'] or 0.,i))
   assert k==s['selected'] and s['scale']==(0,.5,1)[k] and s['budget_unmet']==(not bool(eligible))
   c['commits']+=1;c['trials']+=3;c['selected_'+str(k)]+=1;c['budget_unmet']+=not bool(eligible);c['old_check_exposures']+=len(ids)
   c['aux_candidate_rejected_by_BC']+=sum(i not in valid for i in (1,2));c['retention_loss_nonzero']+=s['retention_loss']>0;c['mode_loss_nonzero']+=s['mode_loss']>0
   if c0 is not None:margins.append(scores[k]['C']-c0)
   selectedR.append(scores[k]['R']);baseR.append(scores[0]['R']);switch.append(scores[k]['switch']);gripper.append(scores[k]['gripper_mean_abs'])
  z=js(p/'ra_coreset/constraints'/('task%d_epoch%02d.json'%(t,h['epoch'])));assert z['active'] and z['count']==len(old) and z['budget']==.1
  assert abs(z['R']-(z['KL_per_dim']+.1*z['switch']))<1e-8 and z['budget_satisfied']==(z['R']<=.1)
  assert z['teacher_hash']=={str(j):d['teacher_binding'][str(j)]['teacher_sha256'] for j in range(6,t)}
  epoch_checks.append(dict(epoch=h['epoch'],**z))
 result[str(t)]=dict(counts=dict(c),epochs=len(epoch_checks),max_selected_current_NLL_delta=max(margins) if margins else None,mean_selected_R=sum(selectedR)/len(selectedR),mean_native_R=sum(baseR)/len(baseR),mean_selected_switch=sum(switch)/len(switch),mean_selected_gripper_abs=sum(gripper)/len(gripper),budget_unmet_fraction=c['budget_unmet']/c['commits'],full_memory_epochs_satisfied=sum(x['budget_satisfied'] for x in epoch_checks),latest_full_memory=epoch_checks[-1],full_memory_checks=epoch_checks)
out=dict(status='PARTIAL_EFTR_CANDIDATE_LOCAL_PASS',observed_unix=d['observed_unix'],stages=result,manifest_sha256=hashlib.sha256((p/'MANIFEST.json').read_bytes()).hexdigest(),limit='Scalar decisions and teacher identities verified; optimizer state equality uses per-entry isolated GPU proof; no closed-loop claim',final_acceptance=False)
(p/'LOCAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(json.dumps({k:v for k,v in out.items() if k!='stages'}));print(json.dumps({t:{k:v for k,v in z.items() if k!='full_memory_checks'} for t,z in result.items()},indent=2))
