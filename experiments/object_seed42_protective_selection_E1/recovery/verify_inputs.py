"""Read-only input verification and failure accounting before recovery."""
import csv,hashlib,json,time
from pathlib import Path
S=Path('/data/mlr_ifa/experiment_workspaces/E1_protective_object_20261008/source')
O=Path('/data/mlr_ifa/experiments/E1_protective_object_seed42_20261008')
R=O/'engineering_recovery/20261008_import_binding'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text())
c=read(O/'CONTRACT.json');assert sha(O/'CONTRACT.json')=='b82fc3d9a58d82302924c0b48eda56e95361227776d59f1f41505dda50b4ee31'
diff={n:dict(expected=h,actual=sha(S/n)) for n,h in c['source_sha256'].items() if sha(S/n)!=h}
assert all(n.startswith('.hydra/') for n in diff),diff
parity=read(O/'FEATURE_PARITY.json');assert sha(O/'FEATURE_PARITY.json')==c['feature_parity']
checked=0
for task,files in parity['features'].items():
    for n,h in files.items():assert sha(O/'features'/task/n)==h;checked+=1
assert checked==5921
assert sha(Path('/data/mlr_ifa/checkpoints/object_multitask_model_ep99.pth'))==c['pretrain_sha256']
t=read(O/'TRAINING_COMPLETE.json')
assert len(t['checkpoints'])==28
assert all(sha(O/'run_001'/n)==h for n,h in t['checkpoints'].items())
rows=list(csv.DictReader((O/'run_001/paper_repro_train_curve.csv').open()))
assert len(rows)==400 and len({(x['stage'],x['epoch']) for x in rows})==400
for task,batches in ((6,157),(7,227),(8,309),(9,362)):
    r=[x for x in rows if int(x['stage'])==task]
    assert sorted(int(x['epoch']) for x in r)==list(range(100))
    assert all(int(x['batches'])==batches and int(x['batch_size'])==10 for x in r)
assert sum(int(x['batches']) for x in rows)==105500
assert not (O/'trials_primary.jsonl').exists() and not (O/'eval_primary.log').exists()
buf=read(O/'buffer_after_task9.json')
assert len(buf)==2961 and all(sha(O/'buffer'/n)==h for n,h in buf.items())
report=dict(status='PASS',timestamp=time.time(),training_epochs=400,optimizer_steps=105500,checkpoints=28,
    feature_hash_matches=checked,buffer_hash_matches=len(buf),pretrain_unchanged=True,
    code_unchanged=True,generated_hydra_metadata_changes=diff,
    failed_attempt=dict(pipeline_exit=1,phase='evaluate_primary',failure='spawn unpickling ModuleNotFoundError libero.libero',
      worker_start_attempts=10,constructed_environments=0,formal_episode_records=0,environment_steps=0,
      evidence='evaluate_primary.log: child fails at reduction.pickle.load before worker function; parent fails at first env.seed before reset or step'),
    recovery_scope='Existing package bind applied in spawned interpreter via external sitecustomize; frozen source/checkpoints/contract untouched.')
(R/'INPUT_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
