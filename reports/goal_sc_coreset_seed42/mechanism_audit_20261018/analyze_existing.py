"""Existing metadata analysis; no torch dependency, training, model inference or rollouts."""
import json,hashlib,statistics,math,csv
from pathlib import Path
from collections import Counter
R=Path(__file__).resolve().parent;E=R/'evidence'
def js(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=js(R/'INPUT_MANIFEST.json')
for name,v in manifest.items():assert sha(E/name)==v['sha256'],name
contract=js(E/'sc_root/CONTRACT.json');parent=js(E/'sc_root/PARENT_BINDING.json')['parent']
for name,v in manifest.items():
 if name.startswith('coreset/') or name.startswith('parent_root/'):
  remote=parent+'/'+(name if name.startswith('coreset/') else name[len('parent_root/'):])
  assert v['sha256']==contract['files'][remote],remote
 if name.startswith('source/'):assert v['sha256']==contract['source_sha256'][name[7:]]
b={t:js(E/'binding_metadata'/f'task{t}.json')['windows'] for t in range(6,10)}
selection={};selected={}
def profile(t,ids):
 x=[b[t][i] for i in ids];n=len(x);groups=Counter(str(v['group']) for v in x);unique={(v['demo_id'],v['target']) for v in x}
 return dict(n=n,joint=dict(groups),time_bins=dict(Counter(str(v['group']//3) for v in x)),demo=dict(Counter(v['demo_id'] for v in x)),positive_pct=100*sum(v['group']%3==2 for v in x)/n,endpoint_pct=100*sum(v['target']==v['length']-1 for v in x)/n,unique_supervised_times=len(unique),repeated_supervised_time_excess=n-len(unique))
for t in range(6,10):
 sc=js(E/'sc/storage'/f'task{t}.json');old=js(E/'coreset/storage'/f'task{t}.json');rep=sc['selection']
 ids={k:[Path(x['source']).name for x in v['selected']] for k,v in [('sc',sc),('coreset',old)]}
 assert ids['coreset']==rep['original_ids'] and ids['sc']==rep['selected_ids']
 selected[t]=ids
 parts={'full':list(b[t]),**ids,'added':sorted(set(ids['sc'])-set(ids['coreset'])),'removed':sorted(set(ids['coreset'])-set(ids['sc']))}
 profiles={name:profile(t,i) for name,i in parts.items()}
 selection[t]={'profiles':profiles,'swaps':len(parts['added']),'radii':{k:rep[k] for k in ['radius_before','radius_after','mean_distance_before','mean_distance_after']}}
 # Joint total variation and per-demo composition are descriptive, not success predictors.
 for name in ['sc','coreset']:
  counts=profiles[name]['joint'];full=profiles['full']['joint'];q=profiles[name]['n'];n=profiles['full']['n']
  profiles[name]['joint_TV']=.5*sum(abs(counts.get(g,0)/q-c/n) for g,c in full.items())
mix={}
for stage in [7,8,9]:
 entry={}
 for method in ['coreset','sc']:
  cells=[]
  for t in range(6,stage+1):
   ids=list(b[t]) if t==stage else selected[t][method]
   cells += [(t,b[t][i]) for i in ids]
  entry[method]={'n':len(cells),'current_n':len(b[stage]),'old_n':len(cells)-len(b[stage]),'positive_pct':100*sum(v['group']%3==2 for _,v in cells)/len(cells),'endpoint_pct':100*sum(v['target']==v['length']-1 for _,v in cells)/len(cells)}
 entry['old_member_replacements']=sum(selection[t]['swaps'] for t in range(6,stage))
 entry['changed_fraction_total']=entry['old_member_replacements']/entry['sc']['n']
 entry['changed_fraction_old']=entry['old_member_replacements']/entry['sc']['old_n']
 mix[stage]=entry
loss={}
for t in range(6,10):
 loss[t]={}
 for method in ['coreset','sc']:
  ee=[x for x in js(E/method/'epochs.json') if x['task']==t];assert len(ee)==100
  loss[t][method]={'epoch0':ee[0]['loss'],'epoch99':ee[-1]['loss'],'first10':statistics.mean(x['loss'] for x in ee[:10]),'last10':statistics.mean(x['loss'] for x in ee[-10:]),'all100':statistics.mean(x['loss'] for x in ee),'finite':all(math.isfinite(x['loss']) for x in ee)}
 old=[x for x in js(E/'coreset/epochs.json') if x['task']==t];new=[x for x in js(E/'sc/epochs.json') if x['task']==t]
 assert all(all(a[k]==c[k] for k in ['batches','steps','lr_first','lr_after']) for a,c in zip(old,new))
 loss[t]['max_abs_epoch_loss_difference']=max(abs(a['loss']-c['loss']) for a,c in zip(old,new))
paired=[];vectors=[]
allrecords={}
for stage in range(6,10):
 for t in range(6,stage+1):
  records={};attempts={}
  for method in ['coreset','sc']:
   records[method]={x['trial']:x for x in rows(E/method/'evaluation'/f'trials_stage{stage}_task{t}.jsonl')}
   attempts[method]=rows(E/method/'evaluation'/f'attempts_stage{stage}_task{t}.jsonl')
  allrecords[stage,t]=records
  assert set(records['coreset'])==set(records['sc'])==set(range(20))
  ss=sf=fs=ff=0;lost=[];gained=[]
  for i in range(20):
   x,y=records['coreset'][i],records['sc'][i];assert x['initial_state_sha256']==y['initial_state_sha256']
   u,v=x['success'],y['success'];ss+=u and v;sf+=u and not v;fs+=not u and v;ff+=not u and not v
   if u and not v:lost.append(i)
   if not u and v:gained.append(i)
  paired.append({'stage':stage,'task':t,'SS':ss,'SF':sf,'FS':fs,'FF':ff,'lost_initial_indices':lost,'gained_initial_indices':gained,'delta_successes':fs-sf})
  for g in [0,1]:
   row={'stage':stage,'task':t,'group':g}
   for method in ['coreset','sc']:
    d=[x for x in attempts[method] if x['event']=='vector_attempt_complete' and x['group']==g];assert len(d)==1
    row[method+'_steps']=d[0]['steps'];row[method+'_successes']=sum(d[0]['successes'])
   vectors.append(row)
history=[]
for t in range(6,10):
 for i in range(20):
  history.append({'task':t,'initial_index':i,**{m:[int(allrecords[s,t][m][i]['success']) for s in range(t,10)] for m in ['coreset','sc']}})
ifa={str(t):{m:js(E/m/f'ifa_task{t}.json')['info'].get('selected_pairs',[]) for m in ['coreset','sc']} for t in range(6,10)}
def config_diff(a,c,path=''):
 if isinstance(a,dict) and isinstance(c,dict):
  out=[]
  for k in sorted(set(a)|set(c)):out+=config_diff(a.get(k),c.get(k),path+'.'+k)
  return out
 return [] if a==c else [{'field':path,'coreset':a,'sc':c}]
cfgdiff=config_diff(js(E/'coreset/evaluation/resolved_actual.json'),js(E/'sc/evaluation/resolved_actual.json'))
remote=js(R/'REMOTE_TENSOR_RNG_AUDIT.json')
assert remote['status']=='PASS' and remote['files_unchanged'] and remote['optimizer_updates']==remote['model_forwards']==remote['environment_calls']==0
audit=js(E/'sc_root/AUDIT.json');cp={v['task']:v['sha256'] for v in audit['checkpoints']}
for path,h in remote['input_sha256'].items():
 if path in contract['files']:assert h==contract['files'][path],path
 elif '/sc_coreset/models/' in path:assert h==cp[int(Path(path).name.split('_')[1])],path
 else:assert '/sc_coreset/evaluation/rng_before_' in path,path
for row in remote['rng']:
 s,t=row['stage'],row['task'];prefix=[v for v in vectors if v['stage']==s and v['task']<t]
 row['preceding_cell_vector_steps']={m:sum(v[m+'_steps'] for v in prefix) for m in ['coreset','sc']}
out={'status':'PASS_READONLY_ANALYSIS','input_files':len(manifest),'selection':selection,'training_mixture':mix,'loss_curves':loss,'paired_outcomes':paired,'vectors':vectors,'initial_state_history':history,'ifa_pairs':ifa,'resolved_eval_config_differences':cfgdiff,'rng':remote['rng'],'model_tensor_differences':remote['models'],'action_aggregates':remote['action_aggregates'],'limits':['No actions, observation trajectories, per-episode RNG snapshots or per-task training BC loss were logged.','Pairing by initial state is descriptive; random streams are not necessarily matched.','No isolated ablation of individual tasks or replaced samples exists.']}
assert all(sha(E/n)==v['sha256'] for n,v in manifest.items())
out['local_inputs_unchanged']=True
(R/'ANALYSIS.json').write_text(json.dumps(out,indent=2)+'\n')
with (R/'paired_outcomes.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
with (R/'initial_state_history.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
print(json.dumps({k:v for k,v in out.items() if k in ['status','training_mixture','loss_curves','paired_outcomes','resolved_eval_config_differences']},indent=2))
