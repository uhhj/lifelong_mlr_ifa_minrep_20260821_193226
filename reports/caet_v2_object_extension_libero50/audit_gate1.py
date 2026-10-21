"""Independent metadata recomputation; no training or environment dependencies."""
import json,hashlib,sys
from pathlib import Path
R=Path(__file__).resolve().parent
p=Path(sys.argv[1]); raw=(p/'EVIDENCE.json').read_bytes();d=json.loads(raw)
assert hashlib.sha256(raw).hexdigest()==json.loads((p/'RECEIPT.json').read_text())['sha256']
assert d['protocol_sha256']==hashlib.sha256((R/'PROTOCOL.md').read_bytes()).hexdigest()
assert all(d[k]==0 for k in ('training_calls','model_forward_calls','environment_calls','server_writes'))
assert not d['cuda_initialized']
for t,v in d['tasks'].items():
 assert v['total_windows']==v['valid_next_labels']+sum(v['invalid'].values())
 assert v['ra_quota']=={'6':781,'7':740,'8':780,'9':660}[t]
 assert all(x['post_obs_alignment_exact'] for x in v['physical'].values())
 for x in v['labels_metadata']:
  assert x['next_action_index']==x['from_state']==x['u']+1
  assert x['to_state']==x['u']+2 and x['max_input_obs_index']<x['next_action_index']
  assert x['to_state']<v['physical'][x['demo']]['length']
summary={}
for t,g in d['stages'].items():
 assert g['sidecar_float32_graph_exact']
 assert g['native_batch_audit']['shuffle_seed']==d['seed']
 cur,old=g['current_nodes'],g['old_nodes']; expected=set()
 for row in cur+old:
  assert row['demo']!=d['tasks'][str(row['task'])]['holdout_demo']
 assert len({(r['task'],r['demo'],r['u']) for r in cur})==len(cur)
 assert len({(r['task'],r['demo'],r['u']) for r in old})==len(old)
 for key,v in g['groups'].items():
  pairs={tuple(x) for x in v['initial_mutual_edges']}
  while True:
   drop=set()
   for i,j in pairs:
    if len({old[b]['demo'] for a,b in pairs if a==i})<3 or len({cur[a]['demo'] for a,b in pairs if b==j})<3:drop.add((i,j))
   if not drop:break
   pairs-=drop
  assert pairs=={tuple(x) for x in v['final_pruned_edges']}
  ci={i for i,j in pairs};oi={j for i,j in pairs}
  cd={cur[i]['demo'] for i in ci};od={old[j]['demo'] for j in oi}
  qualified=len(cd)>=3 and len(od)>=3 and len(ci)>=12 and len(oi)>=12
  assert qualified==v['qualified']
  if qualified:expected.update(pairs)
 assert expected=={tuple(x) for x in g['edges']}
 supported={i for i,j in expected};assert len(supported)==g['supported_current']
 assert g['stage_supported']==bool(expected)
 for ep in g['native_batch_audit']['epochs']:
  assert ep['supported_current_exposures']==len(supported)
  assert ep['batches']=={'7':227,'8':309,'9':362}[t]
  assert (len(supported)+9)//10<=ep['batches_with_supported_current']<=len(supported)
 for h in g['heldout_queries']:
  demos={(old[j]['task'],old[j]['demo']) for j in h['old_neighbors']}
  assert len(demos)==h['old_demo_count'] and h['supported']==(len(demos)>=3)
 summary[t]=dict(supported_times=len(supported),qualified_groups=sum(v['qualified'] for v in g['groups'].values()),heldout_supported=g['heldout_supported'],min_active_batches=min(e['batches_with_supported_current'] for e in g['native_batch_audit']['epochs']),max_active_batches=max(e['batches_with_supported_current'] for e in g['native_batch_audit']['epochs']),fallback=not bool(expected))
assert any(g['stage_supported'] for g in d['stages'].values())
out=dict(status='GATE1_PASS',stages=summary,valid_next_labels=sum(v['valid_next_labels'] for v in d['tasks'].values()),evidence_sha256=hashlib.sha256(raw).hexdigest(),zero_training=True,zero_environment=True,interpretation='Scientific feasibility only; predictor reliability and closed-loop gain untested')
(p/'LOCAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
