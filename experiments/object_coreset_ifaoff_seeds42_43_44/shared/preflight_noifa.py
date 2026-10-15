"""Four real GPU optimizer calls per seed, isolated from all formal arms."""
import ast,copy,json,os,time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from coreset_exp.common import O,S,SEED,verify,write,save
from coreset_exp.model import cfg
from coreset_exp.train import build,capture,restore,cpu
def main():
 verify();assert torch.cuda.is_available();assert not (O/'GPU_PREFLIGHT.json').exists()
 source=(S/'coreset_exp/train.py').read_text();tree=ast.parse(source)
 losses=[n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='loss' for t in n.targets)]
 assert any(isinstance(v,ast.Name) and v.id=='bc' for v in losses)
 payload=[]
 for t in (6,7):
  for p in sorted((O/'features'/str(t)).glob('*.pt'))[:5]:payload.append((torch.load(p,map_location='cpu'),str(t)))
 dist.init_process_group('nccl');torch.cuda.set_device(0)
 try:
  c=cfg('random');assert c.ifa_weight==0.0
  c.run_root=str(O/'preflight');c.experiment_dir=str(O/'preflight/models');c.buffer_dir=str(O/'preflight/buffer')
  a,_=build(c);loader=DataLoader(payload,batch_size=10,num_workers=0);a.start_task(-1,loader)
  a.refs={6:payload[0][0]['feature'][0,0,0].cuda(),7:payload[5][0]['feature'][0,0,0].cuda()}
  info={'selected_pairs':[(6,7)]}
  # Same frozen model/RNG and batch, changing the IFA pair list cannot change
  # an IFA-off optimizer update. Compare real tensors, not checkpoint bytes.
  start=capture(a);save(O/'preflight/start.pth',start)
  first=a.observe(next(iter(loader)),7,0,info);expected=cpu(a.policy.state_dict())
  restore(a,start);second=a.observe(next(iter(loader)),7,0,{'selected_pairs':[]})
  assert first==second and all(torch.equal(v,expected[k]) for k,v in cpu(a.policy.state_dict()).items())
  snap=capture(a);save(O/'preflight/recovery.pth',snap)
  first=a.observe(next(iter(loader)),7,0,info);expected=cpu(a.policy.state_dict())
  restore(a,torch.load(O/'preflight/recovery.pth',map_location='cpu'))
  second=a.observe(next(iter(loader)),7,0,info)
  assert first==second and all(torch.equal(v,expected[k]) for k,v in cpu(a.policy.state_dict()).items())
  assert a.step==2
 finally:dist.destroy_process_group()
 verify();write(O/'GPU_PREFLIGHT.json',{'status':'PASS','seed':SEED,'isolated_optimizer_calls':4,'ifa_pair_invariance_actual_tensors':True,'optimizer_scheduler_all_rng_recovery_exact':True,'formal_steps':0,'environment_episodes':0})
 print('NOIFA_GPU_PREFLIGHT_PASS',SEED)
if __name__=='__main__':main()
