import json,hashlib,sys
from pathlib import Path
import torch
seed=int(sys.argv[1]);root=Path(f'/data/mlr_ifa/experiments/Coreset_IFAoff_OBJECT_seed{seed}_20261014');parent=Path(f'/data/mlr_ifa/experiments/Coreset_paper_OBJECT_seed{seed}_20261013')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (root/'TRAINING_ACCEPTANCE.json').exists()
assert read(root/'TASK6_TENSOR_PARITY.json')['equal']
assert (root/'random/INITIAL_STATE.json').read_bytes()==(root/'coreset/INITIAL_STATE.json').read_bytes()
assert (root/'random/INITIAL_STATE.json').read_bytes()==(parent/'random/INITIAL_STATE.json').read_bytes()
loads={}
for arm in ('random','coreset'):
 ep=read(root/arm/'epochs.json');assert [(r['task'],r['epoch']) for r in ep]==[(t,e) for t in range(6,10) for e in range(100)]
 assert ep[-1]['steps']==105500
 complete=read(root/arm/'TRAINING_COMPLETE.json')
 for t in range(6,10):
  mp=root/arm/'storage'/f'task{t}.json';m=read(mp);old=read(parent/arm/'storage'/f'task{t}.json')
  assert [(Path(r['source']).name,r['sha256']) for r in m['selected']]==[(Path(r['source']).name,r['sha256']) for r in old['selected']]
  cp=root/arm/'models'/f'mlrifa_{t}_model_ep99.pth';assert sha(cp)==complete['checkpoints'][cp.name]
  raw=torch.load(cp,map_location='cpu');assert raw['epoch_index']==99 and raw['task']==t and raw['cfg'].ifa_weight==0.0
  assert raw['contract_sha256']==sha(root/'CONTRACT.json') and raw['modal_sha256']==sha(parent/'MODAL_CONTRACT.json')
  assert len(raw['state_dict'])==100;loads[f'{arm}_{t}']=sha(cp)
  if t==6:
   oldcp=torch.load(parent/arm/'models'/cp.name,map_location='cpu')['state_dict'];assert set(oldcp)==set(raw['state_dict'])
   assert all(torch.equal(v,oldcp[k]) for k,v in raw['state_dict'].items())
assert sha(root/'FEATURES.json')==sha(parent/'FEATURES.json')
result=dict(status='PASS',seed=seed,epochs=800,steps=211000,final_checkpoints=8,checkpoint_hashes=loads,
 task6_tensor_equal_to_original_and_across_arms=True,buffer_members_and_order_identical_to_original=True,
 note='CPU checkpoint metadata/tensor audit; downstream complete-policy strict loads additionally required by off-evaluation preparation')
(root/'TRAINING_ACCEPTANCE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
