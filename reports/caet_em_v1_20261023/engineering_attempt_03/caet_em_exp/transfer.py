"""Frozen own-prior-stage teacher; observation support is never action retrieval."""
from torch.utils.data import DataLoader,Subset
from .common import *
from .model import selected,grouped_huber,masked_mean,huber
def support_features(x,mean,std):
 y=(x-mean)/std
 return torch.cat([torch.nn.functional.normalize(y[:,:512],dim=-1)/3**.5,torch.nn.functional.normalize(y[:,512:1024],dim=-1)/3**.5,torch.nn.functional.normalize(y[:,1024:],dim=-1)/3**.5],-1)
def sample_mean(v,m):return (v*m).sum(-1)/m.sum(-1).clamp_min(1)
@torch.no_grad()
def build_cache(arm,task,policy,codec,ds):
 root=arm_root(arm);cp=root/'teachers'/('task%d.pth'%task);mp=cp.with_suffix('.json')
 binding=dict(contract=sha(ROOT/'CONTRACT.json'),codec_sha256=sha(frozen_codec()),prior_policy_sha256=sha(root/'models'/('task%d.pth'%(task-1))),dataset_sha256=sha(root/'dataset'/('task%d.json'%task)))
 if cp.exists():
  m=js(mp);assert m['binding']==binding and sha(cp)==m['sha256'];return torch.load(cp,map_location='cpu')
 policy.eval();codec.eval();old=[i for i,r in enumerate(ds.rows) if r['task']<task and r['identity_valid'] and bool((r['action_mask']&r['response_mask']&r['visual_mask']).any())];assert old
 qpred=[];qgt=[];zpred=[];ra=[];rf=[]
 for b in DataLoader(ds,32,shuffle=False,num_workers=0):
  b=to_device(b);q,z,_=policy(b['feature']);e=z*policy.effect_std+policy.effect_mean;qg=policy.decoder(b['effect'],b['feature']);lr,lv=codec.reconstruction(e,b)
  qpred.append(tuple(cpu(v) for v in q));qgt.append(tuple(cpu(v) for v in qg));zpred.append(cpu(z))
  ra.append(cpu(sample_mean(grouped_huber(selected(q)[0],b['action']),b['action_mask'])))
  rf.append(cpu((sample_mean(lr,b['response_mask'])+sample_mean(lv,b['visual_mask']))/2))
 qp=tuple(torch.cat([q[k] for q in qpred]) for k in range(3));qg=tuple(torch.cat([q[k] for q in qgt]) for k in range(3));z=torch.cat(zpred);ra=torch.cat(ra);rf=torch.cat(rf)
 raw=torch.stack([r['support'] for r in ds.rows]);mean=raw[old].mean(0);std=raw[old].std(0,unbiased=False).clamp_min(1e-4);obs=support_features(raw,mean,std)
 ids=[(r['task'],r['demo']) for r in ds.rows];oldids=[ids[i] for i in old]
 # At most ~3k legal members: chunk the distance matrix, never scan discarded demos.
 cross=[];support_count=[];all_d=[]
 for start in range(0,len(ds),128):all_d.append(torch.cdist(obs[start:start+128],obs[old]))
 distances=torch.cat(all_d)
 for i in old:
  mask=torch.tensor([d!=ids[i] for d in oldids]);assert mask.any();cross.append(distances[i,mask].min())
 radius=float(torch.quantile(torch.stack(cross),.95));valid=torch.tensor([bool((r['action_mask']&r['response_mask']&r['visual_mask']).any()) for r in ds.rows])
 eligible=torch.tensor(old)[valid[old]];assert len(eligible)>0
 tau_a=max(float(torch.quantile(ra[eligible],.9)),.02);tau_f=max(float(torch.quantile(rf[eligible],.9)),.1)
 for i in range(len(ds)):support_count.append(len({oldids[j] for j in torch.where(distances[i]<=radius)[0].tolist()}))
 support=torch.tensor(support_count)>=3;cur=torch.tensor([r['task']==task for r in ds.rows]);w=cur&support&(ra<=tau_a)&(rf<=tau_f)&valid
 cache=dict(z=z,qpred=qp,qgt=qg,w=w,ra=ra,rf=rf,support_count=torch.tensor(support_count),binding=binding)
 save(cp,cache);report=dict(status='PASS',binding=binding,sha256=sha(cp),radius_q95=radius,tau_action=tau_a,tau_evidence=tau_f,old_members=len(old),current_members=int(cur.sum()),accepted=int(w.sum()),failures=dict(support=int((cur&~support).sum()),action=int((cur&(ra>tau_a)).sum()),evidence=int((cur&(rf>tau_f)).sum()),invalid=int((cur&~valid).sum())),teacher_policy_windows=len(ds),teacher_decoder_GT_windows=len(ds),reconstruction_windows=len(ds),threshold_source='old_RA_only',optimizer_steps=0,environment_steps=0)
 write(mp,report);write(root/'TRANSFER_ADMISSION_BY_STAGE'/('task%d.json'%task),report);return cache
def transfer_loss(z,b,cache,current_task):
 ix=b['index'].detach().cpu();w=cache['w'][ix].to(z.device);target=cache['z'][ix].to(z.device);mask=b['effect_mask']&w[:,None]
 return masked_mean(huber(z,target),mask),int(mask.sum()),int(w.sum())
