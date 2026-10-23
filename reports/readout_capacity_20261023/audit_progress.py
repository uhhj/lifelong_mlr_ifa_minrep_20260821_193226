import json,hashlib,sys,math
from pathlib import Path
R=Path(__file__).resolve().parent;S=Path(sys.argv[1]);ev=json.loads((S/'EVIDENCE.json').read_text());dl=json.loads((S/'DOWNLOAD.json').read_text());assert not ev['hash_bad'];changes=[]
for name,md in dl.items():
 raw=(S/'metadata'/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==md['sha256']
 if md['sha256']!=md['at_snapshot']['sha256']:changes.append(name)
old=json.loads((R.parent/'20261023_physical_effect_probe/snapshot_20261023_141307/metadata/COST_AUDIT.json').read_text());train={x['key'].replace('native_P3_',''):x['train_windows'] for x in old['costs'] if x['key'].startswith('native_P3_')}
epochs={}
if (S/'metadata/EVENTS.jsonl').exists():
 for line in (S/'metadata/EVENTS.jsonl').read_text().splitlines():
  e=json.loads(line)
  if e['event']=='epoch':epochs.setdefault(e['key'],[]).append(e)
for key,rows in epochs.items():
 assert [x['epoch'] for x in rows]==list(range(len(rows))) and len(rows)<=200
 steps=math.ceil(train[key.split('_',1)[1]]/128)
 assert all(x['updates']==steps*(x['epoch']+1) and math.isfinite(x['training_loss']) and math.isfinite(x['validation_position_m']) for x in rows)
fail={k:v for k,v in ev['phase_exits'].items() if v!='0'}
admission=S/'metadata/CAPACITY_ADMISSION.json'
if epochs:
 a=json.loads(admission.read_text());assert a['status']=='PASS_ISOLATED_GPU' and a['optimizer_calls']==9 and a['environment_episodes']==0
 assert a['baseline_untrained_forwards']==6 and a['temporal_ffn']==128 and all(r['state_exact'] and r['output_exact'] and r['rng_exact'] for r in a['baseline_512_equivalence'])
 assert [r['parameters'] for r in a['rows']]==[483335,400775,400775]
 assert [r['kind'] for r in a['rows']]==['P0','G','S'] and all(r['restore_exact'] and r['isolated_optimizer_calls']==3 for r in a['rows'])
out=dict(status='ENGINEERING_FAILURE_PRESERVED' if fail else 'PASS_PROGRESS_NOT_FINAL',complete_probe_epochs=sum(len(v) for v in epochs.values()),complete_probe_updates=sum(v[-1]['updates'] for v in epochs.values()),full_200epoch_fits=sum(len(v)==200 for v in epochs.values()),changing_metadata=changes,failures=fail,evidence_sha256=hashlib.sha256((S/'EVIDENCE.json').read_bytes()).hexdigest(),policy_updates=0,formal_episodes=0)
assert ev['disk']['/data']['root']>=2*1024**3 and ev['disk']['/']['ordinary']>=1024**3
(S/'LOCAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
