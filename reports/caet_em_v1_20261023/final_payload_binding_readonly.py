"""Final CPU payload binding: no model forward, optimizer, environment or remote writes."""
import os,sys,json
from pathlib import Path
ROOT=Path(os.environ['EM_ROOT']);sys.path.insert(0,str(ROOT))
from caet_em_exp.common import *
from caet_em_exp.dataset import diagnostic_dataset
assert (ROOT/'controller.exit').read_text().strip()=='0'
contract=verify();frozen=sha(frozen_codec());proof={}
for pp in Path('/proc').iterdir():
 if pp.name.isdigit() and int(pp.name)!=os.getpid():
  try:env=(pp/'environ').read_bytes().split(b'\0')
  except OSError:continue
  assert ('EM_ROOT='+str(ROOT)).encode() not in env
for arm in ARMS:
 root=arm_root(arm);stages={}
 for t in range(6,10):
  ds=diagnostic_dataset(arm,t);manifest=js(root/'dataset'/('task%d.json'%t))
  # This reconstructs the saved population, never invokes Dataset.__getitem__.
  assert len(ds)==POPS[t] and len(manifest['rows'])==len(ds)
  sm=js(root/'sidecars'/('task%d.json'%t));sp=root/'sidecars'/('task%d.pth'%t)
  assert sha(sp)==sm['sha256'] and sm['codec_sha256']==frozen
  assert sm['projection_sha256']==sha(ROOT/'shared/projection.pth') and sm['normalization_sha256']==sha(ROOT/'shared/normalization.pth')
  storage=js(root/'storage'/('task%d.json'%t));parent=js(PARENT/'ra_coreset/storage'/('task%d.json'%t))
  assert storage['selected']==parent['selected'] and storage['parent_storage_sha256']==sha(PARENT/'ra_coreset/storage'/('task%d.json'%t))
  assert storage['sidecar_sha256']==sha(sp)
  memory=torch.load(sp,map_location='cpu');expected={r['destination']:r['sha256'] for r in parent['selected']}
  assert sm['members']==expected and set(memory)==set(expected)
  for name,h in expected.items():assert sha(root/'buffer'/name)==h==memory[name]['sample_sha256']
  item=dict(population=len(ds),action_valid=sum(int(r['action_mask'].sum()) for r in ds.rows),effect_valid=sum(int((r['action_mask']&r['response_mask']&r['visual_mask']).sum()) for r in ds.rows),RA_members=len(memory),sidecar_sha256=sha(sp),dataset_sha256=sha(root/'dataset'/('task%d.json'%t)))
  if arm=='B2' and t>6:
   tp=root/'teachers'/('task%d.pth'%t);tm=js(tp.with_suffix('.json'));assert sha(tp)==tm['sha256']
   teacher=torch.load(tp,map_location='cpu');binding=dict(contract=contract,codec_sha256=frozen,prior_policy_sha256=sha(root/'models'/('task%d.pth'%(t-1))),dataset_sha256=item['dataset_sha256'])
   assert teacher['binding']==tm['binding']==binding
   assert teacher['z'].shape==(len(ds),4,64)
   assert all(v.shape[0]==len(ds) for q in ('qpred','qgt') for v in teacher[q])
   valid=torch.tensor([bool((r['action_mask']&r['response_mask']&r['visual_mask']).any()) for r in ds.rows]);cur=torch.tensor([r['task']==t for r in ds.rows])
   w=cur&(teacher['support_count']>=3)&(teacher['ra']<=tm['tau_action'])&(teacher['rf']<=tm['tau_evidence'])&valid
   assert torch.equal(w,teacher['w']) and int(w.sum())==tm['accepted']
   item['teacher']=dict(sha256=sha(tp),binding=binding,accepted=int(w.sum()),eligibility_exact=True,policy_payload_rows=len(ds))
  stages[str(t)]=item
 proof[arm]=stages
assert verify()==contract
print(json.dumps(dict(status='PASS_CPU_PAYLOAD_BINDING',contract=contract,arms=proof,codec_sha256=frozen,forward_calls=0,optimizer_calls=0,environment_steps=0,server_writes=0,payload_downloaded=False)))
