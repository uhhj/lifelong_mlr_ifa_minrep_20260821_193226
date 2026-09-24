import copy,time,gc
import numpy as np,torch
from scipy.sparse.linalg import LinearOperator,eigsh
from gradient_relation_experiment import kernel as g
from actual_update_fidelity_v1 import engine as native
from gn_functional_geometry_v1.engine import Engine
from gn_h3_subspace_closure_v1.project import project,orthonormal_random
from .common import *

def head_flat(model):
 return torch.cat([p.detach().reshape(-1) for p in model.policy_head.parameters()])
def head_copy(model,v):
 offset=0
 with torch.no_grad():
  for p in model.policy_head.parameters():
   p.copy_(v[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()
 assert offset==len(v)
def model_copy(algo,state):
 algo.policy.load_state_dict(state,strict=True);algo.policy.eval()
def flat_state(algo,state,head=False):
 return torch.cat([state[n].reshape(-1) for n,p in algo.policy.named_parameters() if p.requires_grad and (not head or 'policy_head.' in n)])
def pair_proposal(algo,ds,start,base):
 t=time.monotonic();g.restore(algo,base);algo.optimizer.zero_grad();data=g.batch(ds,start['record_indices']);data[0]=algo.map_tensor_to_device(data[0])
 ps=native.ps(algo);ids=torch.tensor([int(x) for x in data[1]],device='cuda');model=g.module(algo)
 assert algo.loss_scale==1. and model.policy_head.loss_coef==1.
 losses,_=model.compute_loss(data[0],None,None,reduction='none')
 assert losses.shape==ids.shape and torch.isfinite(losses).all()
 full=losses.mean();old=(losses*(ids!=start['task_id'])).sum()/len(ids);new=(losses*(ids==start['task_id'])).sum()/len(ids)
 gr=torch.autograd.grad(old,ps,allow_unused=True,retain_graph=True)
 gn=torch.autograd.grad(new,ps,allow_unused=True,retain_graph=True)
 full.backward();gm=tuple(None if p.grad is None else p.grad.detach().clone() for p in ps)
 err=g.norm(tuple(None if a is None else a-(torch.zeros_like(a) if b is None else b)-(torch.zeros_like(a) if c is None else c) for a,b,c in zip(gm,gr,gn)))
 assert err<=max(1e-6,2e-5*g.norm(gm)),err
 gr=tuple(None if a is None else torch.zeros_like(a) if b is None else b.detach().clone() for a,b in zip(gm,gr))
 forward=dict(after_rng=g.cpu_state(g.rng_state()),buffers={n:g.cpu_state(b) for n,b in algo.policy.named_buffers()})
 values=dict(full=float(full.detach()),replay=float(old.detach()),new=float(new.detach()),decomposition_error=err,decomposition_relative=err/max(g.norm(gm),1e-30),batch_size=len(ids),new_count=int((ids==start['task_id']).sum()),original_denominator=len(ids))
 del losses,full,old,new,gn
 states={};counts={}
 for kind,grad in [('M',gm),('R',gr)]:
  g.restore(algo,base)
  before_lr=[x['lr'] for x in algo.optimizer.param_groups]
  norms=g.apply_first(algo,grad,forward);s=g.capture(algo)
  assert s['optimizer_step']==base['optimizer_step']+1 and s['scheduler_step']==base['scheduler_step']+1
  states[kind]=s['model']
  counts[kind]=dict(**norms,clip_coefficient=min(1.,algo.cfg.train.grad_clip/(norms['preclip_norm']+1e-6)),lr=before_lr,optimizer_step=s['optimizer_step'],scheduler_step=s['scheduler_step'],first_moment_l2=sum(float(v['exp_avg'].double().square().sum()) for v in s['optimizer']['state'].values() if 'exp_avg' in v)**.5,second_moment_l2=sum(float(v['exp_avg_sq'].double().square().sum()) for v in s['optimizer']['state'].values() if 'exp_avg_sq' in v)**.5)
  if kind=='M':expected=s
  else:del s
 # Native independent actual step, same process/state/RNG. Always transactional.
 g.restore(algo,base);g.native(algo,g.batch(ds,start['record_indices']),start['task_id'],start['epoch'])
 parity=g.compare_states(expected,g.capture(algo));assert all(parity.values()),parity
 g.restore(algo,base);rollback=g.compare_states(base,g.capture(algo));assert all(rollback.values())
 values.update(branches=counts,parity=parity,rollback=rollback,seconds=time.monotonic()-t,common_loss_terms='none; official compute_loss only terminal GMM NLL',counterfactual_zero_semantics='explicit zero for native participating coordinates, retain native None')
 return states,values

def latents(algo,rows):
 zs=[];actions=[];model=g.module(algo)
 with torch.no_grad():
  for i in range(0,len(rows),16):
   data=[torch.load(ROOT/r['origin'],map_location='cpu') for r in rows[i:i+16]]
   h=torch.cat([r['feature'] for r in data]).cuda();a=torch.cat([r['actions'] for r in data]).cuda()
   zs.append(model.temporal_encode(h)[:,-1,:].detach());actions.append(a[:,-1,:])
 return torch.cat(zs),torch.cat(actions)
def random_controls(x,v,r,seeds):
 result={'GN':v};P=len(x);norm=x.norm()
 if norm==0:
  for k in METHODS[1:]:result[k]=torch.zeros_like(x)
  return result
 a=float(v.square().sum()/x.square().sum());assert -1e-10<=a<=1+1e-10
 a=min(1.,max(0.,a))
 for i,seed in enumerate(seeds):
  R=orthonormal_random(P,r,seed,x.device) if r else x.new_zeros(P,0)
  result['C1_'+str(i)]=R@(R.T@x);del R
  gen=torch.Generator(device=x.device);gen.manual_seed(seed+100)
  z=torch.randn(P,device=x.device,dtype=x.dtype,generator=gen)
  for _ in range(2):z-=x*torch.dot(x,z)/norm.square()
  z/=z.norm();t=a*x+(a*(1-a))**.5*norm*z
  assert abs(float(t.norm()-v.norm()))<=1e-9*max(1.,float(norm))
  assert abs(float(x@t-t@t))<=1e-10*max(1.,float(norm.square()))
  result['C2_'+str(i)]=t
 result['C3']=v.norm()/norm*x
 return result

def geometry(algo,base,states,rows,protocol):
 t=time.monotonic();model_copy(algo,base['model']);model=g.module(algo)
 z,a=latents(algo,rows);weights=[];groups=[]
 for lo,hi in [(0,3),(3,6),(6,7)]:
  raw=float(a[:,lo:hi].double().var(0,unbiased=False).mean().sqrt());scale=max(raw,.05)
  weights.extend([1/((hi-lo)*scale**2)]*(hi-lo));groups.append(dict(lo=lo,hi=hi,raw_scale=raw,scale=scale))
 e=Engine(copy.deepcopy(model.policy_head).double(),{'old_build':z.double()},{'old_build':a.double()},torch.tensor(weights,device='cuda',dtype=torch.float64))
 P=e.theta.numel();op=LinearOperator((P,P),matvec=lambda q:e.cv(q).cpu().numpy(),dtype=np.float64)
 opt=protocol['eigensolver'];vals,U=eigsh(op,k=32,ncv=opt['ncv'],tol=opt['tol'],maxiter=opt['maxiter'],v0=np.random.RandomState(opt['seeds'][0]).normal(size=P))
 order=np.argsort(vals)[::-1];vals=vals[order];Q=torch.as_tensor(U[:,order].copy(),device='cuda');del U
 checks=protocol['numerics'];gramerr=float((Q.T@Q-torch.eye(32,device='cuda')).abs().max())
 residuals=[float((e.cv(Q[:,k])-vals[k]*Q[:,k]).norm())/max(abs(vals[k]),1e-30) for k in range(32)]
 assert gramerr<=checks['orthogonal_max'] and max(residuals)<=checks['eigen_relative_residual_max']
 assert np.all(np.diff(vals)<=0) and vals.min()>=-1e-10*vals.max()
 trace=e.trace();assert vals.sum()<=trace*(1+checks['trace_excess_relative'])
 rank=min(16,int(np.sum(vals>max(1e-12,vals[0]*1e-10))))
 basis=Q[:,:rank]
 x=(flat_state(algo,states['M'],True).double()-flat_state(algo,states['R'],True).double()).cuda()
 v=project(basis,x) if rank else torch.zeros_like(x);tail=x-v
 full=float(x@e.cv(x));inside=float(v@e.cv(v));outside=float(tail@e.cv(tail));cross=2*float(v@e.cv(tail))
 assert abs(full-inside-outside-cross)<=max(1e-16,1e-7*abs(full))
 allnorm2=sum(float((states['M'][n].double()-states['R'][n].double()).square().sum()) for n,p in algo.policy.named_parameters() if p.requires_grad)
 engagement=dict(rho_head=float(v.square().sum()/x.square().sum()) if x.norm()>0 else None,rho_all=float(v.square().sum())/allnorm2 if allnorm2 else None,conditional_increment_norm=allnorm2**.5,conditional_head_norm=float(x.norm()),projection_norm=float(v.norm()),E_full=full,E_r=inside,E_tail=outside,E_cross=cross,head_parameter_count=P,r_eff=rank,truncated_energy=float((basis.T@x).square()@torch.as_tensor(vals[:rank],device='cuda')) if rank else 0.)
 check=dict(orthogonal_error=gramerr,max_relative_eigen_residual=max(residuals),eigenvalues=vals.tolist(),trace=trace,trace_coverage={str(k):float(vals[:k].sum()/trace) for k in [8,16,32]},eigengap16=float(vals[15]-vals[16]),action_metric=dict(diagonal=weights,groups=groups),rank=rank,jvp=e.jvps,vjp=e.vjps,seconds=time.monotonic()-t)
 directions=random_controls(x,v,rank,protocol['random_seeds'])
 result={k:w.cpu() for k,w in directions.items()}
 check['basis_storage']='No persistent large basis; deterministic source/data/operator/eigensolver seeds and coefficients retained'
 check['coefficients']=(basis.T@x).cpu().tolist()
 del e,Q,basis,z,a;gc.collect();torch.cuda.empty_cache()
 return result,engagement,check

def candidates(algo,base,states,directions):
 heads={};meta={}
 b=flat_state(algo,base['model'],True);m=flat_state(algo,states['M'],True);r=flat_state(algo,states['R'],True)
 heads['BASE']=b;heads['X00']=r;heads['X10']=(r.double()+directions['GN']).float();heads['KEEP']=m
 meta.update(BASE='BASE',X00='R',X10='R',KEEP='M')
 for method in METHODS:
  for lam in [0.,.5]:
   name=method+('_REMOVE' if lam==0 else '_HALF')
   heads[name]=(m.double()-(1-lam)*directions[method]).float();meta[name]='M'
 return heads,meta
def candidate_metric(head,z,actions):
 dist=head(z);return head.loss_fn(dist,actions[:,None,:],reduction='none'),dist
def eval_roles(algo,bundle,roles,prefix,numerical=False,distribution=False):
 heads,meta=bundle['heads'],bundle['meta'];model=g.module(algo);results=[];effects={};cost=time.monotonic()
 for role,rows in roles.items():
  if not role.startswith(prefix+'_') or not rows:continue
  cache={}
  for source,state in [('BASE',bundle['base']['model']),('R',bundle['states']['R']),('M',bundle['states']['M'])]:
   model_copy(algo,state);cache[source]=latents(algo,rows)
  # Fix KEEP distribution and model samples before evaluating alternatives, without global RNG mutation.
  model_copy(algo,bundle['states']['M']);z,act=cache['M'];head_copy(model,heads['KEEP'].cuda())
  with torch.no_grad():
   keep_loss,keep_dist=candidate_metric(model.policy_head,z,act);keep_mean=keep_dist.mean.detach()
   with torch.random.fork_rng(devices=[0]):
    torch.manual_seed(read(OUT/'PROTOCOL.json')['kl_seed']);draws=keep_dist.sample((64,)) if distribution else None
   keep_log=keep_dist.log_prob(draws) if distribution else None
  values={};values64={};extra={}
  for name,h in heads.items():
   head_copy(model,h.cuda());assert torch.equal(head_flat(model).cpu(),h);zs,actions=cache[meta[name]]
   with torch.no_grad():
    loss,dist=candidate_metric(model.policy_head,zs,actions)
    values[name]=loss.double().cpu().numpy()
    if numerical:
     hd=copy.deepcopy(model.policy_head).double()
     values64[name]=candidate_metric(hd,zs.double(),actions.double())[0].cpu().numpy();del hd
    if distribution:
     dif=keep_log-dist.log_prob(draws);kl=dif.mean(0);se=dif.std(0,unbiased=True)/8
     extra[name]=dict(kl=kl.cpu().numpy(),kl_se=se.cpu().numpy(),mean_change=(dist.mean-keep_mean).square().sum(-1).cpu().numpy())
  if numerical:
   delta={name:float(np.max(abs((v-values['KEEP'])-(values64[name]-values64['KEEP'])))) for name,v in values.items()}
   effects[role]=dict(task_id=rows[0]['task_id'],max_paired_precision_difference=max(delta.values()),per_candidate=delta)
   # Repeated deterministic forward explicitly checked, beyond cross-precision comparison.
   head_copy(model,heads['KEEP'].cuda())
   with torch.no_grad():repeat=candidate_metric(model.policy_head,*cache['M'])[0].double().cpu().numpy()
   assert np.array_equal(repeat,values['KEEP'])
  for name,losses in values.items():
   for i,(row,loss) in enumerate(zip(rows,losses)):
    rr={k:row[k] for k in ['task_id','trajectory_id','window_index','trajectory_length','terminal_index','time_stratum','gripper_change','origin']}
    rr.update(start_id=bundle['start']['start_id'],seed=bundle['start']['seed'],transition=bundle['start']['transition'],role=role,candidate=name,nll=float(loss),delta_base=float(loss-values['BASE'][i]),difference_keep=float(loss-values['KEEP'][i]))
    if distribution:rr.update({k:float(v[i]) for k,v in extra[name].items()})
    results.append(rr)
 del cache;return results,effects,time.monotonic()-cost
def physical_checks(bundle):
 heads=bundle['heads'];gn=bundle['directions']['GN'];actual=heads['KEEP'].double()-heads['GN_REMOVE'].double()
 return dict(realized_delete_norm=float(actual.norm()),ideal_delete_norm=float(gn.norm()),materialization_error_norm=float((actual-gn).norm()),status='BELOW_PHYSICAL_RESOLUTION' if torch.equal(heads['KEEP'],heads['GN_REMOVE']) else 'RESOLVED',nonhead_same=True,explanation='Each M branch shares unchanged native decoder cache/parameters; only head copied')
def prepare(algo,ds,start,roles,base):
 protocol=read(OUT/'PROTOCOL.json');t=time.monotonic();torch.cuda.reset_peak_memory_stats()
 states,parity=pair_proposal(algo,ds,start,base)
 directions,engagement,checks=geometry(algo,base,states,roles['G_old'],protocol)
 heads,meta=candidates(algo,base,states,directions)
 bundle=dict(start=start,base=base,states=states,directions=directions,heads=heads,meta=meta,parity=parity,engagement=engagement,geometry=checks)
 bundle['physical']=physical_checks(bundle)
 bundle['D'],bundle['noise'],bundle['D_seconds']=eval_roles(algo,bundle,roles,'D',numerical=start['smoke'])
 g.restore(algo,base);assert all(g.compare_states(base,g.capture(algo)).values())
 bundle['prepare_seconds']=time.monotonic()-t;bundle['peak_gpu']=torch.cuda.max_memory_allocated()
 return bundle
def trajectory_mean(rows,candidate,role):
 group={}
 for r in rows:
  if r['candidate']==candidate and r['role']==role:group.setdefault(r['trajectory_id'],[]).append(r['nll'])
 return float(np.mean([np.mean(v) for v in group.values()]))
def select_D(rows,taus):
 choices={}
 for method in METHODS:
  names=['KEEP',method+'_HALF',method+'_REMOVE']
  values={n:.5*(trajectory_mean(rows,n,'D_old')+trajectory_mean(rows,n,'D_new')) for n in names}
  tasks={r['role']:str(r['task_id']) for r in rows}
  tie=.5*(taus[tasks['D_old']]['tau_num']+taus[tasks['D_new']]['tau_num'])
  minimum=min(values.values());name=next(n for n in names if values[n]<=minimum+tie)
  choices[method]=dict(candidate=name,lambda_value=1. if name=='KEEP' else .5 if name.endswith('HALF') else 0.,D_objectives=values,tie_tolerance=tie)
 return choices
def finish(algo,bundle,roles,taus):
 sid=bundle['start']['start_id'];dest=OUT/'starts'/sid;dest.mkdir(parents=True,exist_ok=True)
 # This function's selection sees only D rows; atomically freeze before T forward/evaluation.
 choices=select_D(bundle['D'],taus);write(dest/'SELECTIONS_D_ONLY.json',choices)
 table(dest/'D_WINDOWS.csv',bundle['D'])
 bundle['T'],_,ts=eval_roles(algo,bundle,roles,'T',distribution=True)
 table(dest/'T_WINDOWS.csv',bundle['T'])
 for name in ['parity','engagement','geometry','physical','noise']:write(dest/(name.upper()+'.json'),bundle[name])
 if bundle['start']['smoke']:
  SCRATCH.mkdir(parents=True,exist_ok=True)
  # Native trainable model only and REMOVE head delta, frozen for later rollout; no optimizer continuation.
  state={n.replace('module.','',1):v for n,v in bundle['states']['M'].items() if n.replace('module.','',1).startswith(('temporal_transformer.','policy_head.'))}
  g.atomic_save(SCRATCH/(sid+'_behavior.pt'),dict(native_model=state,remove_head=bundle['heads']['GN_REMOVE'],start_id=sid,formal_training_resumed=False))
 g.restore(algo,bundle['base']);rollback=g.compare_states(bundle['base'],g.capture(algo));assert all(rollback.values())
 write(dest/'COMPLETE.json',dict(start_id=sid,status='PASS',prepare_seconds=bundle['prepare_seconds'],T_seconds=ts,D_seconds=bundle['D_seconds'],peak_gpu_bytes=bundle['peak_gpu'],rollback=rollback,formal_training_resumed=False,selection_sha256=sha(dest/'SELECTIONS_D_ONLY.json')))
 status('POINT_COMPLETE',start_id=sid,completed=len(list((OUT/'starts').glob('*/COMPLETE.json'))))
