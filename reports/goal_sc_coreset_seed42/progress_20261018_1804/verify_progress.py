"""Read-only recomputation of this frozen progress snapshot; never loads tensors."""
import json, hashlib, math
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone, timedelta

R = Path(__file__).resolve().parent
def js(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p): return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]

manifest = js(R/'PACKAGE_MANIFEST.json')
for name, row in manifest['files'].items():
    p = R/name
    assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], name
root = R/'sc_coreset'
epochs = js(root/'epochs.json')
assert len(epochs) == 400
assert {(r['task'],r['epoch']) for r in epochs} == {(t,e) for t in range(6,10) for e in range(100)}
assert sum(r['batches'] for r in epochs) == epochs[-1]['steps'] == 76100
assert (R/'train.exit').read_text().strip() == '0'
assert js(R/'TASK6_TENSOR_PARITY.json')['equal']
assert all(v == 0 for v in js(R/'TASK6_TENSOR_PARITY.json')['max_abs_diff'].values())
selections = {}
for t in range(6,10):
    m = js(root/'storage'/f'task{t}.json'); rep = m['selection']
    b = js(R/'binding_metadata'/f'task{t}.json')['windows']
    pre = js(R/'selection_preflight'/f'task{t}.json')
    assert all('last_action' not in v for v in b.values())
    assert rep['selected_ids'] == pre['selected_ids'] and rep['original_ids'] == pre['original_ids']
    q = [499,445,452,985][t-6]; n = len(b)
    ids = [Path(x['source']).name for x in m['selected']]
    assert ids == rep['selected_ids'] and len(set(ids)) == len(ids) == m['K'] == q
    counts = Counter(str(b[i]['group']) for i in ids)
    for g, ng in rep['population'].items():
        assert q*ng//n <= counts[g] <= math.ceil(q*ng/n)
        assert counts[g] == rep['target'][g] == rep['final'][g]
        assert abs(counts[g]*n-q*ng) < n
    swaps = len(set(rep['original_ids'])-set(ids))
    assert swaps == rep['minimum_swaps'] == len(rep['swaps'])
    selections[str(t)] = {k: rep[k] for k in ['minimum_swaps','retained_original','radius_before','radius_after','population','original','final','demo_final']}
    selections[str(t)]['quota'] = q
    selections[str(t)]['positive_percent'] = {key:100*sum(v for g,v in rep[key].items() if int(g)%3==2)/sum(rep[key].values()) for key in ['population','original','final']}
    selections[str(t)]['endpoint_percent'] = {key:100*sum(v for g,v in rep[key].items() if int(g)//3==4)/sum(rep[key].values()) for key in ['population','original','final']}
ev = root/'evaluation'; trials = [r for p in ev.glob('trials*.jsonl') for r in lines(p)]
assert len(trials) == len({(r['stage'],r['task'],r['trial']) for r in trials})
assert all(type(r['success']) is bool and r['seed']==42 and r['arm']=='sc_coreset' for r in trials)
events = [r for p in ev.glob('attempts*.jsonl') for r in lines(p)]
counts = Counter(r['event'] for r in events)
starts = {(r['stage'],r['task'],r['group']):r for r in events if r['event']=='vector_attempt_start'}
finished = [r for r in events if r['event']=='vector_attempt_complete']
assert len(trials)==10*len(finished)
durations = [r['unix']-starts[r['stage'],r['task'],r['group']]['unix'] for r in finished]
assert not any('exception' in k for k in counts)
server = js(R/'REMOTE_PROGRESS_AUDIT.json')
assert all(server[k]=='PASS' for k in ['frozen_source_inputs_features_modal','parent_protected_inputs','actual_buffer_payload_hashes'])
ifa = {}
for t,row in server['ifa'].items():
    a=row['current']['info'].get('selected_pairs',[]); b=row['parent']['info'].get('selected_pairs',[])
    ifa[t] = dict(current=a,parent=b,selected_pairs_equal=a==b)
result = dict(status='PARTIAL_TRAINED_EVAL_PENDING',package_files_verified=len(manifest['files']),snapshot_time_cst=datetime.fromtimestamp(manifest['time'],timezone(timedelta(hours=8))).isoformat(),epochs=400,optimizer_steps=76100,task6_tensor_parity=True,selections=selections,formal_records=len(trials),completed_vectors=len(finished),attempt_events=dict(counts),seconds_per_completed_vector=durations,linear_remaining_minutes=(20-len(finished))*sum(durations)/len(durations)/60,ifa=ifa,server_snapshot=server['snapshot'],final_metrics='PENDING_200_EPISODES_AND_EXIT_ACCEPTANCE')
(R/'LOCAL_PROGRESS_ACCEPTANCE.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['selections','server_snapshot']},indent=2))
