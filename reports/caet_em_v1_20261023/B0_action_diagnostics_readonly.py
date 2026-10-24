"""Standalone CPU scalar diagnostics; streamed stdout, no server filesystem writes."""
import os,sys,json,time,math,hashlib,subprocess
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['LIBERO_CONFIG_PATH']='/data/mlr_ifa/config'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.nice(10)
root=Path(os.environ['EM_ROOT']);sys.path.insert(0,str(root))
from caet_em_exp.common import verify,sha,js,arm_root,frozen_codec,torch
from caet_em_exp.model import Policy,strict_policy_restore,selected,grouped_huber
from caet_em_exp.dataset import diagnostic_dataset
from torch.utils.data import DataLoader
torch.set_num_threads(1);torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)
def emit(value):print(json.dumps(value,allow_nan=False,separators=(',',':')),flush=True)
contract=verify()
assert contract=='2f6af444986a5710cbc84572f25ec15f9b3d44ea72aacbd01fa3ad4e624681b0'
br=arm_root('B0');codec_sha=sha(frozen_codec());complete=js(br/'TRAINING_COMPLETE.json')
assert (root/'phases/B0_evaluate.exit').read_text().strip()=='0'
assert (root/'phases/B0_audit.exit').read_text().strip()=='0'
start=time.time();cpu_start=time.process_time()
emit(dict(kind='start',contract=contract,unix=start,device='cpu',threads=1,nice=os.nice(0),cuda_initialized=torch.cuda.is_initialized(),optimizer_steps=0,environment_steps=0))
checkpoints={};costs=[]
for stage in range(6,10):
 tic=time.time();tc=time.process_time();cp=br/'models'/('task%d.pth'%stage);cp_sha=sha(cp)
 assert cp_sha==complete['checkpoints'][str(stage)]
 payload=torch.load(cp,map_location='cpu');model=Policy('B0')
 nkeys=strict_policy_restore(model,payload,contract,codec_sha)
 model.eval().requires_grad_(False)
 assert all(not p.requires_grad and p.device.type=='cpu' for p in model.parameters())
 assert all(not m.training for m in model.modules())
 ds=diagnostic_dataset('B0',stage);dm=br/'dataset'/('task%d.json'%stage);dm_sha=sha(dm)
 old=js(br/'diagnostics'/('task%d.json'%stage));assert old['checkpoint_sha256']==cp_sha
 records={};calls=0;parity=[]
 for batch in DataLoader(ds,batch_size=32,shuffle=False,num_workers=0):
  q,_,_=model(batch['feature']);calls+=1
  mu,sd,logits=[x.detach() for x in q];chosen_mu,chosen_sd,k=selected(q)
  target=batch['action'];mask=batch['action_mask'].bool()
  assert torch.isfinite(mu).all() and torch.isfinite(sd).all() and torch.isfinite(logits).all() and (sd>0).all()
  probs=logits.softmax(-1);lp=logits.log_softmax(-1)
  top=probs.topk(2,-1);entropy=-(probs*lp).sum(-1)/math.log(probs.shape[-1])
  comp_lp=torch.distributions.Normal(mu,sd).log_prob(target[:,:,None,:]).sum(-1)
  joint=lp+comp_lp;mix_lp=joint.logsumexp(-1);posterior=joint.softmax(-1)
  select_lp=comp_lp.gather(-1,k[...,None]).squeeze(-1)
  comp_err=grouped_huber(mu,target[:,:,None,:].expand_as(mu))
  oracle_err,oracle_mode=comp_err.min(-1)
  err=chosen_mu-target;gh=grouped_huber(chosen_mu,target)
  second_idx=top.indices[...,1,None,None].expand(*k.shape,1,7)
  second_mu=mu.gather(-2,second_idx).squeeze(-2)
  separation=grouped_huber(chosen_mu,second_mu)
  chosen_post=posterior.gather(-1,k[...,None]).squeeze(-1)
  post_mode=posterior.argmax(-1)
  tail=.5*(1+torch.erf((-1-chosen_mu)/(chosen_sd*math.sqrt(2))))+.5*(1-torch.erf((1-chosen_mu)/(chosen_sd*math.sqrt(2))))
  assert torch.isfinite(comp_lp).all() and torch.isfinite(gh).all()
  for j,index in enumerate(batch['index'].tolist()):
   r=ds.rows[index];previous=old['rows'][index]
   assert (r['sample'],r['task'],r['sample_sha256'])==(previous['sample'],previous['task'],previous['sample_sha256'])
   v=previous['metrics']['predicted_action'];s=float(gh[j][mask[j]].sum());c=int(mask[j].sum())
   assert c==v['count'];parity.append(dict(index=index,task=r['task'],old_sum=v['sum'],new_sum=s,count=c))
   for h in range(4):
    if not mask[j,h]:continue
    row=dict(index=index,sample_sha256=r['sample_sha256'],task=r['task'],current=r['task']==stage,horizon=h+1,
     huber=float(gh[j,h]),oracle_huber=float(oracle_err[j,h]),selected_mode=int(k[j,h]),oracle_mode=int(oracle_mode[j,h]),posterior_mode=int(post_mode[j,h]),
     pmax=float(top.values[j,h,0]),weight_margin=float(top.values[j,h,0]-top.values[j,h,1]),entropy=float(entropy[j,h]),selected_posterior=float(chosen_post[j,h]),
     mixture_nll=float(-mix_lp[j,h]),selected_nll=float(-select_lp[j,h]),top2_huber=float(separation[j,h]),box_tail=float(tail[j,h].mean()),sigma_min=float(chosen_sd[j,h].min()),sigma_max=float(chosen_sd[j,h].max()))
    for label,sl in [('translation',slice(0,3)),('rotation',slice(3,6)),('gripper',slice(6,7))]:
     er=err[j,h,sl];sig=chosen_sd[j,h,sl];z=er.abs()/sig
     row.update({label+'_mae':float(er.abs().mean()),label+'_mse':float(er.square().mean()),label+'_sigma_rms':float(sig.square().mean().sqrt()),label+'_coverage1':float((z<=1).float().mean()),label+'_coverage2':float((z<=2).float().mean())})
    records.setdefault('stage%d_task%d_h%d'%(stage,r['task'],h+1),[]).append(row)
 # Read-only identities and all checkpoint tensors must still be unchanged.
 assert sha(cp)==cp_sha and sha(dm)==dm_sha and sha(frozen_codec())==codec_sha
 assert all(torch.equal(v,payload['model'][key]) for key,v in model.state_dict().items())
 assert all(sha(path)==r['sample_sha256'] for path,r in zip(ds.file_paths,ds.rows))
 assert not torch.cuda.is_initialized()
 checks=dict(stage=stage,checkpoint_sha256=cp_sha,strict_keys=nkeys,dataset_sha256=dm_sha,old_diagnostic_sha256=sha(br/'diagnostics'/('task%d.json'%stage)),codec_sha256=codec_sha,windows=len(ds),forward_calls=calls,valid_positions=sum(len(v) for v in records.values()),wall_seconds=time.time()-tic,cpu_seconds=time.process_time()-tc,model_tensors_unchanged=True,input_sha_before_after=True,parity=parity)
 for name,rows in sorted(records.items()):emit(dict(kind='case',name=name,rows=rows))
 emit(dict(kind='stage',evidence=checks));costs.append({k:v for k,v in checks.items() if k!='parity'});checkpoints[str(stage)]=cp_sha
 del model,payload,ds,records
assert verify()==contract
assert all(sha(br/'models'/('task%s.pth'%t))==h for t,h in checkpoints.items())
emit(dict(kind='complete',status='PASS_CPU_READONLY_DIAGNOSTIC_EXTRA_FORWARDS_NOT_NEW_CLOSED_LOOP',contract=contract,codec_sha256=codec_sha,checkpoints=checkpoints,costs=costs,wall_seconds=time.time()-start,cpu_seconds=time.process_time()-cpu_start,policy_windows=sum(x['windows'] for x in costs),forward_calls=sum(x['forward_calls'] for x in costs),optimizer_steps=0,environment_steps=0,cuda_initialized=torch.cuda.is_initialized(),server_artifact_writes=0))
