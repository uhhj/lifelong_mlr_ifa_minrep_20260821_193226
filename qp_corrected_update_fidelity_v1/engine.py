import gc,time,traceback,os
import numpy as np,pandas as pd,torch
from actual_update_fidelity_v1 import engine as native
from gradient_relation_experiment import kernel as g
from functional_geometry_diagnostic_v1.geometry import gmm
from sample_qp_optimized_v1_continuation_v4.core import Workspace,inner,project,Timings
from .common import *
def npz(path,**arrays):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp.npz');np.savez(tmp,**arrays);os.replace(tmp,path)
def refs_batches(algo,ds,refs):
 return [algo.map_tensor_to_device(g.batch(ds,[r['record_id'] for r in refs[j:j+8]])[0]) for j in range(0,len(refs),8)]
def reference_matrix(algo,ds,refs,w):
 before_rng=g.digest(g.rng_state());modes={n:m.training for n,m in algo.policy.named_modules()};algo.policy.eval()
 def loss(data):return g.module(algo).policy_head.loss_fn(gmm(algo,data),data['actions'].squeeze(1),'none')
 try:H=w.sample_gradients(refs_batches(algo,ds,refs),loss,len(refs))
 finally:
  for n,m in algo.policy.named_modules():m.training=modes[n]
 assert before_rng==g.digest(g.rng_state())
 K,_=inner(H,gram=True);return H,K

def evaluate(algo,ds,base,start,rows,d0,dfull,role):
 data=native.probe_data(algo,ds,rows);g.restore(algo,base);algo.policy.eval();ps=native.ps(algo);output=[]
 oldsource=pd.read_csv(SOURCE/'starts'/start['start_id']/'measurements.csv')
 oldsource=oldsource[(oldsource.arm=='adamw')&(oldsource.alpha==1)].set_index('record_id')
 for row,datum in zip(rows,data):
  loss=native.single_loss(algo,datum);value=float(loss.detach())
  with torch.no_grad():repeat=float(native.single_loss(algo,datum))
  if role=='old' and row.get('prior_tau_reused',False):tau=float(oldsource.loc[row['record_id'],'tau_zero']);assert np.isclose(value,float(oldsource.loc[row['record_id'],'base_loss']),rtol=1e-12,atol=1e-12)
  else:tau=max(2*abs(value-repeat),32*np.finfo(np.float32).eps*max(1,abs(value)))
  gradients=torch.autograd.grad(loss,ps,allow_unused=True)
  h=native.flat([torch.zeros_like(p) if v is None else v for p,v in zip(ps,gradients)]).detach().double()
  output.append(dict(start_id=start['start_id'],seed=start['seed'],transition=start['transition'],stage=start['stage'],task_id=row['task_id'],trajectory_id=row['trajectory_id'],window_index=row['window_index'],record_id=row['record_id'],role=role,base_loss=value,tau_zero=tau,h_dot_d0=float(torch.dot(h,d0)),h_dot_dfull=float(torch.dot(h,dfull)) if dfull is not None else None,reference_trajectory_overlap=row.get('reference_trajectory_overlap',False),replacement_for_origin=row.get('replacement_for_origin')))
  del h,gradients,loss
 for name,d in [('d0',d0),('dfull',dfull)]:
  if d is None:
   for row in output:row['measured_delta_'+name]=None
   continue
  native.apply(algo,base,d,1.);algo.policy.eval()
  with torch.no_grad():values=[float(native.single_loss(algo,dat)) for dat in data]
  assert np.isfinite(values).all()
  for row,value in zip(output,values):row['measured_delta_'+name]=value-row['base_loss']
 for row in output:row['protection_gain']=None if dfull is None else row['measured_delta_dfull']-row['measured_delta_d0']
 g.restore(algo,base);return output

def corrected(algo,ds,start,refs,base,destination):
 begin=time.monotonic();proposal=native.proposal(algo,g.batch(ds,start['record_indices']),base);d0=proposal['d0'];w=Workspace(native.ps(algo));before=w.snapshot_parameters()
 t=time.monotonic();H,K=reference_matrix(algo,ds,refs,w);reference_seconds=time.monotonic()-t
 g.restore(algo,proposal['candidate']);u=w.displacement(before);_,f=inner(H,u)
 un=float(u.double().norm());valid=np.flatnonzero(np.diag(K)>1e-12);scale=np.sqrt(np.diag(K)[valid]);A=K[np.ix_(valid,valid)]/scale[:,None]/scale[None,:];b=f[valid]/scale/max(un,1e-30)
 npz(destination/'QP_INPUT.npz',K=K,f=f,u_norm=np.asarray(un),valid=valid,scale=scale,A=A,b=b)
 context=dict(start_id=start['start_id'],seed=start['seed'],task=start['task_id'],epoch=start['epoch'],batch=start['batch_index'],global_step=start['global_step'],optimizer_step_before=base['optimizer_step'],scheduler_step_before=base['scheduler_step'])
 rec=dict(**context,constraints=len(refs),reference_seconds=reference_seconds,solver_version='v4 unchanged',formal_training_resumed=False)
 dfull=None;solution=None
 try:
  projection_start=time.perf_counter();timer=Timings(True);correction,solution=project(H,K,u,timer,diag_dir=destination/'solver',context=context)
  w.subtract_correction(correction);written=w.displacement(before);_,rates=inner(H,written);hn=np.sqrt(np.maximum(np.diag(K),0));wn=float(written.double().norm())
  positive=int((rates>5e-6*hn*wn+1e-15).sum());rec.update(actual_positive=positive,physical_rate_max=float(np.maximum(rates/np.maximum(hn*max(wn,1e-30),1e-30),0).max()))
  assert positive==0,('DIAGNOSTIC_PHYSICAL_WRITEBACK_FAILED',positive)
  dfull=native.flat([p.detach().double()-base['model'][n].to(p.device).double() for n,p in algo.policy.named_parameters() if p.requires_grad])
  dn=float(d0.norm());fn=float(dfull.norm());cn=float((dfull-d0).norm());fb=solution.get('fallback',{});tc=solution.get('second_fallback',{})
  rec.update(status='PASS',projection_total_seconds=time.perf_counter()-projection_start,native_norm=dn,corrected_norm=fn,correction_norm=cn,correction_ratio=cn/max(dn,1e-30),norm_ratio=fn/max(dn,1e-30),cosine=float(torch.dot(dfull,d0)/(dfull.norm()*d0.norm())) if dn*fn else None,kkt_before=float(np.maximum(b,0).max(initial=0)),standard_kkt=solution['standard_kkt'],kkt_after=solution['kkt_error'],iterations=solution['iterations'],base_sweeps=solution['base_sweeps'],continuation_sweeps=solution['continuation_sweeps'],slsqp_iterations=solution['slsqp_iterations'],solver_branch='trust-constr' if tc.get('triggered') else ('L-BFGS-B' if fb.get('triggered') else 'standard'),solve_seconds=timer.seconds.get('qp_solver'),projection_residual=solution['written_normalized_rate_max'],gram_condition=float(np.linalg.cond(A)) if np.isfinite(np.linalg.cond(A)) else None,formal_d0_rounding_difference=float((u.double()-d0).norm()),optimizer_candidate=algo._optimizer_step_count,scheduler_candidate=algo._scheduler_step_count)
  assert algo._optimizer_step_count==base['optimizer_step']+1 and algo._scheduler_step_count==base['scheduler_step']+1
  write(destination/'SOLUTION.json',solution)
 except (AssertionError,RuntimeError,ValueError) as error:
  dfull=None
  rec.update(status='FAILED',error=str(error),traceback=traceback.format_exc(),native_norm=float(d0.norm()));write(destination/'FAILURE_CONTEXT.json',rec)
 finally:g.restore(algo,base)
 rec['seconds']=time.monotonic()-begin
 return d0,dfull,rec,H,K,w,proposal

def measure(algo,ds,start,refs,old,new):
 destination=OUT/'starts'/start['start_id'];destination.mkdir(parents=True,exist_ok=True);base=g.capture(algo)
 try:
  d0,dfull,record,H,K,w,proposal=corrected(algo,ds,start,refs,base,destination)
  rows=evaluate(algo,ds,base,start,old,d0,dfull,'old')
  newrows=evaluate(algo,ds,base,start,new,d0,dfull,'new')
  refrows=evaluate(algo,ds,base,start,refs,d0,dfull,'reference')
  _,refp0=inner(H,d0);refpb=inner(H,dfull)[1] if dfull is not None else None
  for j,row in enumerate(refrows):
   row['single_forward_h_dot_d0']=row['h_dot_d0'];row['h_dot_d0']=float(refp0[j]);row['h_dot_dfull']=float(refpb[j]) if refpb is not None else None
   row['prediction_source']='exact formal reference gradients batched8'
  table(destination/'OLD.csv',rows);table(destination/'NEW.csv',newrows);table(destination/'REFERENCE.csv',refrows)
  g.restore(algo,base);assert all(g.compare_states(base,g.capture(algo)).values());record['source_restored']=True
  write(destination/'COMPLETE.json',record);return record
 finally:g.restore(algo,base)
