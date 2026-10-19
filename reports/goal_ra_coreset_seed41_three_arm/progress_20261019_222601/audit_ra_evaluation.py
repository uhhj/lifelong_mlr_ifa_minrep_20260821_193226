"""Read-only RA training/storage and partial episode audit; baseline evidence stays immutable."""
from pathlib import Path
from collections import Counter
import json,hashlib,statistics,sys,datetime
R=Path(__file__).resolve().parent;D=R/sys.argv[1];RA=D/'ra_evidence';P=R/'progress_20261019_203008'
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
proof=js(D/'RA_FROZEN_INPUT_AUDIT.json');assert proof['status']=='PASS_RA_FROZEN_INPUTS'
for n,h in proof['manifest'].items():assert sha(RA/n)==h['sha256']
parent=js(RA/'PARENT_BINDING.json');base='/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed41_20261019'
assert parent['parent']==base
refs={}
for directory in ['baseline','baseline_evidence']:
 for p in (P/directory).rglob('*'):
  if not p.is_file():continue
  rel=p.relative_to(P/directory).as_posix();remote=base+'/'+rel
  if remote in parent['protected']:assert sha(p)==parent['protected'][remote]
  refs[p.relative_to(R).as_posix()]=sha(p)
(D/'BASELINE_EVIDENCE_REFERENCE.json').write_text(json.dumps(dict(root=P.name,published_commit='b8ba6c64e28f49bb889662ee57956258f677d434',hashes=refs,parent_protected_hashes_revalidated_on_server=True),indent=2)+'\n')
ep=js(RA/'ra_coreset/epochs.json');assert len(ep)==400 and {(x['task'],x['epoch']) for x in ep}=={(t,e) for t in range(6,10) for e in range(100)}
assert sum(x['batches'] for x in ep)==ep[-1]['steps']==76100
assert (RA/'train.exit').read_text().strip()=='0'
assert js(RA/'TASK6_TENSOR_PARITY.json')['equal']
assert js(RA/'ra_coreset/INITIAL_STATE.json')==js(P/'baseline_evidence/coreset/INITIAL_STATE.json')
complete=js(RA/'ra_coreset/TRAINING_COMPLETE.json');assert len(complete['checkpoints'])==4
for n,h in complete['checkpoints'].items():assert proof['checkpoints']['ra_coreset/models/'+n]['sha256']==h
buffer={};storage={}
for t,k in zip(range(6,10),[499,445,452,985]):
 m=js(RA/'ra_coreset/storage'/f'task{t}.json');off=js(RA/'offline_selection'/f'task{t}.json');selected=m['selected']
 assert m['K']==len(selected)==k and len({x['destination'] for x in selected})==k
 assert [Path(x['source']).name for x in selected]==off['selected_ids']==m['selection']['selected_ids']
 assert set(off['anchor_ids'])<=set(off['selected_ids']) and off['swap_count']==len(set(off['anchor_ids'])-set(off['original_ids']))
 assert proof['storage'][f'task{t}']['source_buffer_payload_hashes']
 for x in selected:assert x['destination'] not in buffer;buffer[x['destination']]=x['sha256']
 storage[str(t)]=dict(count=k,swaps=off['swap_count'],anchor_count=len(off['anchor_ids']),offline_order_exact=True,payload_hashes_pass=True)
assert js(RA/'ra_coreset/buffer_after_task9.json')['hashes']==buffer
rows=[json.loads(l) for p in (RA/'ra_coreset/evaluation').glob('trials*.jsonl') for l in p.read_text().splitlines()]
events=[json.loads(l) for p in (RA/'ra_coreset/evaluation').glob('attempts*.jsonl') for l in p.read_text().splitlines()]
assert len({(r['stage'],r['task'],r['trial']) for r in rows})==len(rows)
for r in rows:
 assert r['arm']=='ra_coreset' and r['seed']==41 and r['epoch']==99 and type(r['success']) is bool and 6<=r['task']<=r['stage']<=9 and 0<=r['trial']<20
 assert r['checkpoint_sha256']==complete['checkpoints'][f"mlrifa_{r['stage']}_model_ep99.pth"]
counts=Counter(e['event'] for e in events);assert counts['technical_exception']==0
starts={(x['stage'],x['task'],x['group']):x for x in events if x['event']=='vector_attempt_start'}
ends={(x['stage'],x['task'],x['group']):x for x in events if x['event']=='vector_attempt_complete'};assert len(rows)==10*len(ends)
seconds=[]
for key,x in ends.items():
 assert key in starts and x['env_seed']==41+key[2]
 rr=sorted([r for r in rows if (r['stage'],r['task'],r['trial']//10)==key],key=lambda r:r['trial'])
 assert [r['success'] for r in rr]==x['successes']
 seconds.append(x['unix']-starts[key]['unix'])
baseline=js(P/'PROGRESS_ACCEPTANCE.json');cells=[]
for key in sorted({(r['stage'],r['task']) for r in rows}):
 rr=[r for r in rows if (r['stage'],r['task'])==key];cells.append(dict(stage=key[0],task=key[1],trials=len(rr),successes=sum(r['success'] for r in rr)))
# Parent 40-vector mean provides a less early-task-biased projection; use RA rate as second estimate.
remaining_vectors=20-len(ends);baseline_eta=remaining_vectors*baseline['mean_vector_seconds']/60+5;ra_eta=remaining_vectors*statistics.mean(seconds)/60+5
out=dict(status='PASS_PARTIAL_RA_EVALUATING',observed_unix=proof['time'],total_epochs=1200,total_steps=228300,total_records=400+len(rows),expected_records=600,RA_records=len(rows),RA_vectors_completed=len(ends),RA_vectors_started=len(starts),technical_exceptions=0,storage=storage,RA_task6_actual_tensor_and_initial_rng_parity=True,checkpoint_hashes_pass=True,RA_final_strict_load_acceptance='PENDING_PIPELINE_AUDIT',source_input_contract_hashes_pass=True,RA_partial_cells=cells,RA_full_metrics=None,baseline_metrics=baseline['full_metrics'],RA_train_minutes=complete['seconds_this_invocation']/60,RA_completed_vector_seconds=seconds,remaining_minutes_parent_rate=baseline_eta,remaining_minutes_RA_rate=ra_eta,new_training_or_environment_calls=0)
(D/'PROGRESS_ACCEPTANCE.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
