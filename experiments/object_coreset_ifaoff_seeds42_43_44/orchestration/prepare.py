import json,hashlib,shutil,datetime
from pathlib import Path
B=Path('/data/mlr_ifa/experiments')
Q=Path('/data/mlr_ifa/analyses/Coreset_IFAoff_original_protocol_20261014')
W=Path('/data/mlr_ifa/experiment_workspaces/Coreset_validation_20261014')
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
Q.mkdir(parents=True,exist_ok=True);assert not (Q/'PREPARED.json').exists()
cancel=read(Path('/data/mlr_ifa/analyses/Coreset_validation_20261014/CANCELLATION.json'))
assert cancel['no_live_workflow_processes'] and cancel['committed_formal_episodes']==11
receipts={}
for s in (42,43,44):
 old=B/f'Coreset_paper_OBJECT_seed{s}_20261013';out=B/f'Coreset_IFAoff_OBJECT_seed{s}_20261014'
 src=W/f'noifa/seed{s}/source';oldsrc=Path(f'/data/mlr_ifa/experiment_workspaces/Coreset_paper_OBJECT_seed{s}_20261013/source')
 c=read(out/'CONTRACT.json');assert c['ifa_weight']==0.0
 assert all(sha(src/n)==h for n,h in c['source_sha256'].items())
 assert all(not (out/a/'epochs.json').exists() for a in ('random','coreset'))
 for n in ('evaluate.py','eval_metric.py','model.py'):
  assert sha(src/'coreset_exp'/n)==sha(oldsrc/'coreset_exp'/n),n
 ec=read(old/'resolved_eval.json');ec.update(ifa_weight=0.0,feature_dir=str(out/'features'),buffer_dir=str(out/'eval/buffer'),experiment_dir=str(out/'eval/models'),run_root=str(out/'eval'),contract=str(out/'CONTRACT.json'))
 assert ec['seed']==s and ec['eval']['n_eval']==20 and ec['eval']['max_steps']==600 and ec['training'] is False
 write(out/'resolved_eval.json',ec)
 # Preserve the parent's complete old-input evidence and additionally protect this parent run.
 protected=read(old/'OLD_INPUTS_READONLY.json')
 for p in sorted(old.rglob('*')):
  if p.is_file() and p.suffix in ('.json','.jsonl','.pth','.md'):
   protected[str(p)]=sha(p)
 write(out/'OLD_INPUTS_READONLY.json',protected)
 contract={'protocol':'IFA_off_original_evaluation_v1','seed':s,'training_contract_sha256':sha(out/'CONTRACT.json'),
  'resolved_eval_sha256':sha(out/'resolved_eval.json'),'old_inputs_sha256':sha(out/'OLD_INPUTS_READONLY.json'),
  'source_evaluation_sha256':{n:sha(src/'coreset_exp'/n) for n in ('evaluate.py','eval_metric.py','model.py')},
  'parent_run':str(old),'new_formal_episodes':400,'initials':list(range(20)),'per_episode_rng':False,
  'new_initial_repeat_evaluation_cancelled':True,'original_input_files':len(protected)}
 write(out/'EVALUATION_CONTRACT.json',contract)
 write(out/'REAUTHORIZATION.json',{'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'user_choice':'Run IFA off for three seeds using original evaluation protocol','previous_status':read(out/'RUN_STATUS.json'),'new_queue':str(Q)})
 (out/'RUN_STATUS.json').write_text(json.dumps({'status':'PREPARED_ORIGINAL_EVAL_IFA_OFF','seed':s,'training_epochs':0,'formal_episodes':0},indent=2)+'\n')
 receipts[s]=contract
 print('PREPARED',s,flush=True)
write(Q/'PREPARED.json',receipts)
