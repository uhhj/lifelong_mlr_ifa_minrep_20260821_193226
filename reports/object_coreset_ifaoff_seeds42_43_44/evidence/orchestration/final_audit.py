"""Run only after queue has exited; no training or environment execution."""
import json,subprocess,hashlib,datetime
from pathlib import Path
Q=Path('/data/mlr_ifa/analyses/Coreset_IFAoff_original_protocol_20261014')
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert not (Q/'FINAL_ACCEPTANCE.json').exists()
assert (Q/'queue.exit').read_text().strip()=='0'
lines=subprocess.check_output(['ps','-eo','args'],text=True).splitlines()
assert not any(' -m coreset_exp.'+m in l for l in lines for m in ('train','evaluate','summarize'))
assert not any(l.startswith('bash ') and ('original_protocol/queue.sh' in l or 'original_protocol/seed_pipeline.sh' in l) for l in lines)
for n,h in read(Q/'orchestration_hashes.json').items():assert sha(n)==h,n
for s in (42,43,44):
 p=Path(f'/data/mlr_ifa/experiments/Coreset_IFAoff_OBJECT_seed{s}_20261014')
 assert (p/'pipeline.exit').read_text().strip()=='0'
 assert read(p/'INDEPENDENT_ACCEPTANCE.json')['status']=='PASS'
 assert read(p/'TRAINING_ACCEPTANCE.json')['status']=='PASS'
 c=read(p/'CONTRACT.json');src=Path(f'/data/mlr_ifa/experiment_workspaces/Coreset_validation_20261014/noifa/seed{s}/source')
 for n,h in c['source_sha256'].items():assert sha(src/n)==h
 for n,h in read(p/'OLD_INPUTS_READONLY.json').items():assert sha(n)==h
 cp=read(p/'EVALUATION_CONTRACT.json');assert sha(p/'resolved_eval.json')==cp['resolved_eval_sha256']
 assert read(p/'SUMMARY.json')['formal_records']==400
x=read(Q/'COMPARISON.json');assert x['new_episodes']==1200 and x['new_epochs']==2400 and x['new_steps']==633000 and len(x['matrices'])==12
result={'status':'COMPLETED','time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'queue_exit':0,'new_epochs':2400,'new_steps':633000,'final_checkpoints':24,'new_formal_episodes':1200,'old_controls_unchanged':True,'no_residual_processes':True}
with (Q/'FINAL_ACCEPTANCE.json').open('x') as f:json.dump(result,f,indent=2)
(Q/'RUN_STATUS.json').write_text(json.dumps(result,indent=2)+'\n')
with (Q/'REPORT.md').open('a') as f:f.write('\n最终队列退出与输入保护验收：COMPLETED。\n')
print(json.dumps(result))
