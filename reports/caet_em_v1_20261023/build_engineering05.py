"""Local derivation only. Preserve engineering04 before narrow continuation changes."""
import ast,json,hashlib,shutil
from pathlib import Path
R=Path(__file__).resolve().parent
archive=R/'engineering_attempt_04';archive.mkdir(exist_ok=False)
for p in [R/'controller.py',R/'deploy.py',*sorted((R/'caet_em_exp').glob('*.py'))]:
 q=archive/p.relative_to(R);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
p=R/'caet_em_exp/common.py'
p.write_text(p.read_text()+'''
def policy_fingerprint(payload):
 h=hashlib.sha256(json.dumps({k:v for k,v in payload.items() if k!='model'},sort_keys=True).encode())
 for k,v in sorted(payload['model'].items()):
  h.update(json.dumps([k,list(v.shape),str(v.dtype)]).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def validate_policy_contract(payload,contract):
 if payload['contract']==contract:return
 manifest=js(ROOT/'RESUME_MANIFEST.json');key='%s_task%d'%(payload['arm'],payload['task'])
 assert key in manifest['accepted_checkpoints'],'Unregistered legacy checkpoint'
 record=manifest['accepted_checkpoints'][key]
 assert payload['contract']==manifest['parent_contract'] and sha(record['path'])==record['sha256']
 assert policy_fingerprint(payload)==record['payload_fingerprint'],'Legacy checkpoint payload changed'
''',encoding='utf8')
p=R/'caet_em_exp/model.py';s=p.read_text();s=s.replace(" assert payload['contract']==contract and payload['codec_sha256']==codec_sha"," validate_policy_contract(payload,contract);assert payload['codec_sha256']==codec_sha");assert 'validate_policy_contract(payload,contract)' in s;p.write_text(s,encoding='utf8')
p=R/'caet_em_exp/train.py';s=p.read_text().replace('from .model import Codec,Policy,nll,masked_mean,huber','from .model import Codec,Policy,nll,masked_mean,huber,strict_policy_restore');s=s.replace("assert prior['contract']==contract and prior['codec_sha256']==sha(frozen_codec());p.load_state_dict(prior['model'],strict=True)","strict_policy_restore(p,prior,contract,sha(frozen_codec()))");assert 'strict_policy_restore(p,prior' in s;p.write_text(s,encoding='utf8')
p=R/'caet_em_exp/dataset.py';p.write_text(p.read_text()+'''
def diagnostic_dataset(arm,task):
 # Training has already retained this task. Reconstruct its exact pre-retention
 # population from the saved training manifest, not the now-expanded buffer.
 root=arm_root(arm);manifest=js(root/'dataset'/('task%d.json'%task));labels=current(task);past={}
 for j in range(6,task):
  side=root/'sidecars'/('task%d.pth'%j);md=js(side.with_suffix('.json'))
  assert sha(side)==md['sha256'] and md['codec_sha256']==sha(frozen_codec())
  past.update(torch.load(side,map_location='cpu'))
 rows=[];paths=[]
 for i,item in enumerate(manifest['rows']):
  tid=item['task'];name=item['sample'];assert item['index']==i and 6<=tid<=task
  path=PARENT/'features'/str(task)/name if tid==task else root/'buffer'/('t%d__'%tid+name)
  r=labels[name] if tid==task else past[path.name]
  assert r['task']==tid and r['sample']==name and r['demo']==item['demo'] and r['u']==item['u']
  assert sha(path)==r['sample_sha256']==item['sha256'] and r['action_mask'].tolist()==item['action_mask']
  assert (r['action_mask']&r['response_mask']&r['visual_mask']).tolist()==item['effect_mask']
  paths.append(str(path));rows.append(r)
 assert len(rows)==POPS[task] and len(set(paths))==len(paths)
 return EvidenceDataset(paths,rows)
''',encoding='utf8')
p=R/'caet_em_exp/diagnose.py';s=p.read_text().replace('from .dataset import make','from .dataset import diagnostic_dataset').replace('ds=make(a.arm,a.task)','ds=diagnostic_dataset(a.arm,a.task)');p.write_text(s,encoding='utf8')
p=R/'controller.py';s=p.read_text();start=s.index("  phase('prepare'");end=s.index("  write(ROOT/'RUN_STATUS.json',dict(status='SERVER_COMPLETE",start)
s=s[:start]+'''  phase('continuation_admission',['continuation_admission'])
  phase('B0_admission',['admission','B0'])
  phase('B0_diagnose7',['diagnose','B0','7'])
  for arm in ARMS:
   if arm!='B0':phase(arm+'_admission',['admission',arm])
   for task in range(8 if arm=='B0' else 6,10):
    if not (ROOT/'evidence'/('task%d'%task)/'TEMPORAL_BINDING_AUDIT.json').exists():phase('evidence%d'%task,['evidence',str(task)])
    phase('%s_train%d'%(arm,task),['train',arm,str(task)])
    phase('%s_diagnose%d'%(arm,task),['diagnose',arm,str(task)])
   phase(arm+'_evaluate',['evaluate',arm]);phase(arm+'_audit',['audit',arm])
'''+s[end:];p.write_text(s,encoding='utf8')
changes={}
for old in archive.rglob('*.py'):
 rel=old.relative_to(archive);new=R/rel
 if new.read_bytes()!=old.read_bytes():changes[rel.as_posix()]=dict(old_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),new_sha256=hashlib.sha256(new.read_bytes()).hexdigest())
for p in [R/'controller.py',*list((R/'caet_em_exp').glob('*.py'))]:ast.parse(p.read_text(),feature_version=(3,8))
(R/'ENGINEERING_REPAIR_05.json').write_text(json.dumps(dict(status='IMPLEMENTED_NOT_ADMITTED',parent_root='/data/mlr_ifa/experiment_workspaces/CAET_EM_OBJECT41_20261023_v1_engineering04',failure_snapshot='snapshot_20261023_221016',preflight='repair05_preflight_readonly_20261023_221421',cause='diagnostics rebuilt native dataset after retain() added current-stage members to buffer; training manifest unchanged',fix='diagnostic-only exact saved-stage population; SHA-pinned legacy B0 checkpoints admitted into new execution contract',completed_main_epochs=200,completed_main_updates=38400,retrain_completed=False,formal_episodes=0,scientific_changes=False,changes=changes),indent=2),encoding='utf8')
print(json.dumps(changes,indent=2))
