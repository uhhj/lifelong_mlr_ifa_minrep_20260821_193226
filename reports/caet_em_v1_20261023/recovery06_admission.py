"""Disposable exact next-batch replay from preserved B2 epoch90; no formal updates."""
import os,sys
from pathlib import Path
Q=Path(os.environ['EM_RECOVERY_ROOT']);ROOT=Path(os.environ['EM_ROOT']);sys.path.insert(0,str(ROOT))
from caet_em_exp.common import *
from caet_em_exp.admission import equal
from caet_em_exp.model import Policy
from caet_em_exp.train import load_codec,optimizer,targets,loss
from caet_em_exp.dataset import diagnostic_dataset
from torch.utils.data import DataLoader,DistributedSampler
proof=js(Q/'RECOVERY_MANIFEST.json');contract=verify();assert contract==proof['science_contract']
path=Q/'provenance/B2_task8_epoch90_recovery.pth';assert sha(path)==proof['recovery']['sha256']
s=torch.load(path,map_location='cpu');assert s['task']==8 and s['next_epoch']==90 and s['steps']==66210 and s['cursor']==0
root=arm_root('B2');assert sha(root/'recovery/latest.pth')==sha(path)
assert sha(root/'teachers/task8.pth')==s['teacher_sha256'] and sha(root/'dataset/task8.json')==s['dataset_sha256'] and sha(frozen_codec())==s['codec_sha256']
seed();pack,codec=load_codec();p=Policy('B2',pack['model'],pack['statistics']).cuda();opt,sched,names=optimizer(p,30900)
ds=diagnostic_dataset('B2',8);targets(ds,codec);cache=torch.load(root/'teachers/task8.pth',map_location='cpu')
manifest=js(root/'dataset/task8.json');info=manifest['ifa'];refs={}
for i,r in enumerate(ds.rows):
 if r['task'] not in refs:refs[r['task']]=torch.load(ds.file_paths[i],map_location='cpu')['feature'][0,0,0].cuda()
assert names==js(root/'optimizer/task8.json')['parameters']
restore(p,opt,sched,s)
actual=capture(p,opt,sched)
for key in ('model','optimizer','scheduler','rng','grads','modes','requires'):assert equal(actual[key],s[key]),key
sampler=DistributedSampler(ds,num_replicas=1,rank=0,shuffle=True,seed=0);loader=DataLoader(ds,batch_size=10,num_workers=0,sampler=sampler)
def step():
 restore(p,opt,sched,s);p.train();sampler.set_epoch(90);b=to_device(next(iter(loader)))
 opt.zero_grad();value,stats=loss(p,b,codec,info,refs,8,cache);assert torch.isfinite(value)
 value.backward();assert all(v.grad is None or torch.isfinite(v.grad).all() for v in p.parameters())
 torch.nn.utils.clip_grad_norm_(p.parameters(),100);opt.step();sched.step()
 return cpu(value),capture(p,opt,sched),b['index'].cpu().tolist(),stats
v,a,ix,stats=step();w,b,iy,again=step();assert torch.equal(v,w) and equal(a,b) and ix==iy and stats==again
assert sha(root/'recovery/latest.pth')==proof['recovery']['sha256']
assert sha(root/'teachers/task8.pth')==s['teacher_sha256'] and verify()==contract
write(Q/'RECOVERY_ADMISSION.json',dict(status='PASS',science_contract=contract,recovery_execution_contract=sha(Q/'EXECUTION_CONTRACT.json'),recovery_sha256=sha(path),exact_GPU_roundtrip_fields=list(actual),exact_next_batch_replay=True,isolated_optimizer_calls=2,formal_updates=0,environment_steps=0,extra_target_tokenizer_windows=len(ds),isolated_policy_windows=20,isolated_retention_decoder_windows=40,first_batch_indices=ix,source_teacher_dataset_codec_unchanged=True,next_epoch=90,steps=66210))
