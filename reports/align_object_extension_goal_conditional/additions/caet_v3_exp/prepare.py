import json,os,shutil,time
from pathlib import Path
from coreset_exp.common import O,SEED,write,sha,source_hashes
def main():
 W=Path(os.environ['CORESET_SOURCE']).parent;entry=json.loads((W/'ENTRY.json').read_text());P=Path(entry['parent']);spec=entry['spec']
 assert not O.exists() and entry['seed']==SEED and entry['variant']=='ALIGN'
 c=json.loads((P/'resolved_ra_coreset.json').read_text());assert c['seed']==SEED and c['benchmark_name']==spec['benchmark_name'] and c['gamma']==spec['gamma']
 assert json.loads((P/'AUDIT.json').read_text())['status']=='PASS'
 acceptance=P/entry['parent_acceptance'];assert json.loads(acceptance.read_text())['status']=='PASS'
 O.mkdir(parents=True);files={str(acceptance):sha(acceptance),str(W/'ENTRY.json'):sha(W/'ENTRY.json')}
 for n in ['FEATURES.json','MODAL_CONTRACT.json','PRETRAIN_METADATA.json','TASKS.json']:
  shutil.copy2(P/n,O/n);files[str(P/n)]=sha(P/n)
 for n in ['features','bindings','selection_preflight','offline_selection']:(O/n).symlink_to((P/n).resolve(),target_is_directory=True)
 b=dict(parent=str(P),contract_sha256=sha(P/'CONTRACT.json'),feature_manifest_sha256=sha(P/'FEATURES.json'),modal_sha256=sha(P/'MODAL_CONTRACT.json'),method='RA-TAC-Align',seed=SEED)
 write(O/'PARENT_BINDING.json',b);write(O/'RUN_SPEC.json',spec)
 for n in ['ra_coreset','eval']:
  c=json.loads((P/('resolved_%s.json'%n)).read_text());c.update(feature_dir=str(O/'features'),buffer_dir=str(O/'ra_coreset/buffer'),run_root=str(O/'ra_coreset'),experiment_dir=str(O/'ra_coreset/models'),contract=str(O/'CONTRACT.json'))
  write(O/('resolved_%s.json'%n),c)
 for path,h in json.loads((P/'CONTRACT.json').read_text()).get('files',{}).items():
  if Path(path).is_file():files[path]=h
 for p in (P/'ra_coreset').rglob('*'):
  if p.is_file() and (p.suffix in ('.json','.jsonl') or p.parent.name=='models'):files[str(p)]=sha(p)
 for n in ['PROTOCOL.md','GATE1_EVIDENCE.json','GATE1_LOCAL_AUDIT.json']:
  shutil.copy2(W/n,O/n)
 for p in O.iterdir():
  if p.is_file():files[str(p)]=sha(p)
 write(O/'CONTRACT.json',dict(version='ALIGN_OBJECT_THREE_SEED_GOAL_CONDITIONAL_20261022',variant='ALIGN',method='RA-TAC-Align',seed=SEED,suite=entry['suite'],files=files,source_sha256=source_hashes(),modal_sha256=b['modal_sha256'],epochs=400,optimizer_steps=spec['optimizer_steps'],dataset_sizes=spec['dataset_sizes'],final_checkpoints=4,formal_records=200,vector_attempts=20,bc_target='next_a_u_plus_1_masked',effect_coefficient=0,cross_coefficient=0,preflight_environment_steps=0,created_unix=time.time()))
 write(O/'RUN_STATUS.json',dict(status='PREPARED_AWAITING_PREFLIGHT'))
if __name__=='__main__':main()
