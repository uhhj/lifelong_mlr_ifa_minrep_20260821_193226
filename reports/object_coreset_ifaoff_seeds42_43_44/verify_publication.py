import json, statistics, hashlib
from types import SimpleNamespace
from pathlib import Path
from fractions import Fraction as F

root=Path(__file__).resolve().parent
class EvidenceFiles:
 def __enter__(self):return self
 def __exit__(self,*args):pass
 def extractfile(self,n):return (root/'evidence'/n).open('rb')
 def getmembers(self):return [SimpleNamespace(name=p.relative_to(root/'evidence').as_posix()) for p in (root/'evidence').rglob('*') if p.is_file()]

with EvidenceFiles() as tar:
 manifest=json.loads((root/'LIGHT_FILE_HASHES.json').read_text())
 assert {m.name for m in tar.getmembers()}==set(manifest)
 for name,h in manifest.items():
  assert hashlib.sha256(tar.extractfile(name).read()).hexdigest()==h,name
 def read(n): return json.load(tar.extractfile(n))
 def rows(n): return [json.loads(x) for x in tar.extractfile(n).read().decode().splitlines()]
 comp=read('queue/COMPARISON.json')
 assert read('queue/FINAL_ACCEPTANCE.json')['status']=='COMPLETED'
 assert tar.extractfile('queue/queue.exit').read().strip()==b'0'
 matrices=[]; values={}; attempts={}; ep_total=step_total=0
 for seed in (42,43,44):
  for cond in ('on','off'):
   prefix=f'{cond}/seed{seed}'
   assert read(prefix+'/INDEPENDENT_ACCEPTANCE.json')['status']=='PASS'
   summary=read(prefix+'/SUMMARY.json')
   if cond=='off':
    assert read(prefix+'/TRAINING_ACCEPTANCE.json')['status']=='PASS'
    assert tar.extractfile(prefix+'/pipeline.exit').read().strip()==b'0'
   for arm in ('random','coreset'):
    rates={}; allrows=[]; events={}
    if cond=='off':
     epochs=read(f'{prefix}/{arm}/epochs.json')
     assert len(epochs)==400 and epochs[-1]['steps']==105500
     assert {(x['task'],x['epoch']) for x in epochs}=={(s,e) for s in range(6,10) for e in range(100)}
     ep_total+=len(epochs);step_total+=epochs[-1]['steps']
    for stage in range(6,10):
     for task in range(6,stage+1):
      rr=rows(f'{prefix}/{arm}/evaluation/trials_stage{stage}_task{task}.jsonl')
      assert len(rr)==20 and {x['trial'] for x in rr}==set(range(20))
      assert all(x['seed']==seed and x['stage']==stage and x['task']==task and x['epoch']==99 and x['initial_state_index']==x['trial'] and type(x['success']) is bool for x in rr)
      if cond=='off':
       old={x['trial']:x for x in rows(f'on/seed{seed}/{arm}/evaluation/trials_stage{stage}_task{task}.jsonl')}
       assert all(x['initial_state_sha256']==old[x['trial']]['initial_state_sha256'] for x in rr)
       ee=rows(f'{prefix}/{arm}/evaluation/attempts_stage{stage}_task{task}.jsonl')
       for e in ee: events[e['event']]=events.get(e['event'],0)+1
       assert sum(x['event']=='vector_attempt_start' for x in ee)==2
       assert sum(x['event']=='vector_attempt_complete' for x in ee)==2
       assert not any(x['event']=='technical_exception' for x in ee)
      rates[stage,task]=F(sum(x['success'] for x in rr),20);allrows.extend(rr)
    assert len(allrows)==200
    m={'AUC':sum(sum(rates[i,j] for i in range(j,10))/(10-j) for j in range(6,10))/4,
       'FWT':sum(rates[j,j] for j in range(6,10))/4,
       'NBT':sum(sum(rates[j,j]-rates[i,j] for i in range(j+1,10))/(9-j) for j in range(6,9))/3}
    for k,v in m.items(): assert str(v)==summary['arms'][arm]['metrics'][k]['fraction']
    values[seed,cond,arm]=m
    matrix=[[float(rates[i,j]*100) if (i,j) in rates else None for j in range(6,10)] for i in range(6,10)]
    found=[x for x in comp['matrices'] if (x['seed'],x['condition'],x['arm'])==(seed,cond,arm)]
    assert len(found)==1 and found[0]['matrix']==matrix
    matrices.append({'seed':seed,'condition':cond,'arm':arm,'matrix':matrix})
    if cond=='off': attempts[f'{seed}_{arm}']=events
 assert (ep_total,step_total)==(2400,633000)
 stats={}
 for k in ('AUC','FWT','NBT'):
  stats[k]={}
  for cond in ('on','off'):
   for arm in ('random','coreset'):
    vv=[float(values[s,cond,arm][k]*100) for s in (42,43,44)]
    stats[k][cond+'_'+arm]={'mean':statistics.mean(vv),'sample_sd':statistics.stdev(vv)}
  deltas={cond:[float((values[s,cond,'coreset'][k]-values[s,cond,'random'][k])*100) for s in (42,43,44)] for cond in ('on','off')}
  for key,vv in [('on_C_minus_R_pp',deltas['on']),('off_C_minus_R_pp',deltas['off']),('interaction_off_minus_on_pp',[float(100*((values[s,'off','coreset'][k]-values[s,'off','random'][k])-(values[s,'on','coreset'][k]-values[s,'on','random'][k]))) for s in (42,43,44)])]:
   stats[k][key]={'mean':statistics.mean(vv),'sample_sd':statistics.stdev(vv)}
  for key,v in stats[k].items():
   for f in ('mean','sample_sd'): assert abs(v[f]-comp['statistics'][k][key][f])<1e-10
 result={'status':'PASS','new_epochs':ep_total,'new_steps':step_total,'new_records':1200,'old_control_records':1200,'matrices':matrices,'statistics':stats,'attempts':attempts,'input_manifest_sha256':hashlib.sha256((root/'LIGHT_FILE_HASHES.json').read_bytes()).hexdigest()}
# Read-only publication check: do not replace original audit evidence.
print(json.dumps({'status':'PASS','statistics':stats,'attempts':attempts},indent=2))

repo=root.parents[1]
exp=repo/'experiments'/'object_coreset_ifaoff_seeds42_43_44'
for seed in (42,43,44):
 c=json.loads((root/f'evidence/off/seed{seed}/CONTRACT.json').read_text())
 for name,h in c['source_sha256'].items():
  assert hashlib.sha256((exp/f'seed{seed}/source'/name).read_bytes()).hexdigest()==h
for name,h in json.loads((root/'PUBLICATION_MANIFEST.json').read_text()).items():
 assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==h,name
print('FROZEN_SOURCE_AND_PUBLICATION_MANIFEST_PASS')
