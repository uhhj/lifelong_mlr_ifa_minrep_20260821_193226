"""Recompute partial progress from immutable episode and attempt metadata."""
import json,hashlib
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone,timedelta
R=Path(__file__).resolve().parent
def js(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
m=js(R/'PACKAGE_MANIFEST.json')
for n,h in m['files'].items():
 b=(R/n).read_bytes();assert len(b)==h['bytes'] and hashlib.sha256(b).hexdigest()==h['sha256'],n
a=js(R/'READONLY_AUDIT.json')
for k in ['frozen_source_config_input_feature_modal','parent_inputs','actual_buffer_payloads']:assert a[k]=='PASS'
e=js(R/'sc_coreset/epochs.json')
assert len(e)==400 and {(r['task'],r['epoch']) for r in e}=={(t,i) for t in range(6,10) for i in range(100)}
assert sum(r['batches'] for r in e)==e[-1]['steps']==76100
assert (R/'train.exit').read_text().strip()=='0'
assert js(R/'TASK6_TENSOR_PARITY.json')['equal']
ev=R/'sc_coreset/evaluation'
tr=[r for p in ev.glob('trials*.jsonl') for r in rows(p)]
assert len(tr)==len({(r['stage'],r['task'],r['trial']) for r in tr})
assert all(type(r['success']) is bool and r['seed']==42 and r['arm']=='sc_coreset' for r in tr)
events=[r for p in ev.glob('attempts*.jsonl') for r in rows(p)]
counts=Counter(r['event'] for r in events)
starts={(r['stage'],r['task'],r['group']):r for r in events if r['event']=='vector_attempt_start'}
done={(r['stage'],r['task'],r['group']):r for r in events if r['event']=='vector_attempt_complete'}
assert len(tr)==len(done)*10
assert not any('exception' in k for k in counts)
dur=[v['unix']-starts[k]['unix'] for k,v in sorted(done.items())]
cells=[]
for s,t in sorted({(r['stage'],r['task']) for r in tr}):
 rr=[r for r in tr if (r['stage'],r['task'])==(s,t)]
 cells.append({'stage':s,'task':t,'n':len(rr),'successes':sum(r['success'] for r in rr),'complete_cell':len(rr)==20})
result={'status':'PARTIAL_TRAINED_EVAL_PENDING','observed_CST':datetime.fromtimestamp(a['time'],timezone(timedelta(hours=8))).isoformat(),'epochs':400,'optimizer_steps':76100,'formal_episodes':len(tr),'completed_vectors':len(done),'remaining_vectors':20-len(done),'active_vectors':[list(k) for k in starts if k not in done],'technical_exceptions':0,'mean_vector_seconds':sum(dur)/len(dur),'recent_four_vector_seconds':dur[-4:],'remaining_minutes_mean':(20-len(done))*sum(dur)/len(dur)/60,'remaining_minutes_recent':(20-len(done))*sum(dur[-4:])/len(dur[-4:])/60,'cells':cells,'metrics_gate':'PENDING_FINAL_ACCEPTANCE'}
(R/'LOCAL_PROGRESS_ACCEPTANCE.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
