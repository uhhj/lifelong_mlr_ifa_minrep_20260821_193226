import json,hashlib,sys
from pathlib import Path
R=Path(sys.argv[1]);d=json.loads((R/'EVIDENCE.json').read_text());dl=json.loads((R/'DOWNLOAD.json').read_text())
assert not d['hash_bad']
changes=[]
for n,v in dl.items():
 p=R/'metadata'/n;h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==v['sha256']
 if h!=v['at_snapshot']['sha256']:changes.append(n)
m=R/'metadata';checks={}
if (m/'INPUT_BINDING_AUDIT.json').exists():
 b=json.loads((m/'INPUT_BINDING_AUDIT.json').read_text());assert b['status']=='PASS'
 for t in ('6','7'):
  rows=json.loads((m/('BINDING_TASK%s.json'%t)).read_text());v=b['tasks'][t]
  assert len(rows)==v['population']=={'6':1562,'7':1480}[t]
  assert len({x['sample'] for x in rows})==len(rows)
  for x in rows:
   assert x['current_state_index']==x['target_action_index']==x['last_observation_index']+1 and x['next_state_index']==x['current_state_index']+1
   assert not x['effect_valid'] or x['state_valid'] and x['action_valid']
  for field in ('state_valid','action_valid','effect_valid'):assert sum(r[field] for r in rows)==v[field]
 checks['identity_masks']='PASS'
if (m/'DEMO_SPLITS.json').exists():
 for fs in json.loads((m/'DEMO_SPLITS.json').read_text()).values():
  assert len(fs)==5 and len(set(v for f in fs for v in f['test']))==10
  for f in fs:
   assert [len(f[k]) for k in ('train','validation','test')]==[7,1,2]
   assert len(set(f['train']+f['validation']+f['test']))==10
 checks['whole_demo_disjoint']='PASS'
failures={k:v for k,v in d['phase_exits'].items() if v!='0'}
status='ENGINEERING_FAILURE_PRESERVED' if failures else 'PASS_PROGRESS_NOT_FINAL'
a=dict(status=status,checks=checks,changing_metadata=changes,failures=failures,evidence_sha256=hashlib.sha256((R/'EVIDENCE.json').read_bytes()).hexdigest())
(R/'LOCAL_AUDIT.json').write_text(json.dumps(a,indent=2));print(json.dumps(a,indent=2))
