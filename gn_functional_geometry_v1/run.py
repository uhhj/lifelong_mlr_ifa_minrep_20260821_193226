import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
import time,sys,traceback,numpy as np,torch
from scipy.sparse.linalg import eigsh,LinearOperator
from .common import *
from .engine import load

def status(stage,**extra):write(OUT/'STATUS.json',dict(stage=stage,training_active=False,formal_training_resumed=False,**extra));print(stage,extra,flush=True)
def main():
 if (OUT/'SUMMARY.json').exists():
  print('Existing completed run preserved; use report-only for summaries.');return
 proto=read(OUT/'PROTOCOL.json');t0=time.time();status('INITIALIZING');e=load();P=e.theta.numel();device=e.theta.device
 torch.cuda.reset_peak_memory_stats();epsilon=proto['epsilon'];ncheck=proto['numerics'];timings={};status('GN_EIGENSOLVER',P=P)
 def matvec(v):return e.cv(v).cpu().numpy()
 op=LinearOperator((P,P),matvec=matvec,dtype=np.float64);t=time.time()
 if not (OUT/'OLD_TASK_GN_BASIS.npz').exists():
  vals,U=eigsh(op,k=32,ncv=72,tol=1e-6,maxiter=500,v0=np.random.RandomState(2026092303).normal(size=P))
  ix=np.argsort(vals)[::-1];vals=vals[ix];U=U[:,ix].astype(np.float32)
  save(OUT/'OLD_TASK_GN_BASIS.npz',U=U,eigenvalues=vals,parameter_names=np.array(e.names),parameter_sizes=np.array(e.sizes))
 else:
  a=np.load(OUT/'OLD_TASK_GN_BASIS.npz');U=a['U'];vals=a['eigenvalues']
 timings['basis_seconds']=time.time()-t;status('GN_REPEAT_CHECK',basis_seconds=timings['basis_seconds'])
 t=time.time();vals2,V=eigsh(op,k=32,ncv=72,tol=1e-6,maxiter=500,v0=np.random.RandomState(2026092304).normal(size=P))
 ix=np.argsort(vals2)[::-1];vals2=vals2[ix];V=V[:,ix];cosines=np.linalg.svd(U.T@V,compute_uv=False);del V
 Q=torch.from_numpy(U.astype(np.float64)).to(device);del U
 gram=(Q.T@Q).cpu().numpy();res=[]
 for k in range(32):res.append(float((e.cv(Q[:,k])-vals[k]*Q[:,k]).norm())/max(abs(vals[k]),1e-30))
 timings['basis_check_seconds']=time.time()-t;status('GN_TRACE',repeat_min_cosine=float(min(cosines)))
 t=time.time();trace=e.trace();timings['trace_seconds']=time.time()-t
 checks=dict(unit_norm_max_error=float(np.max(abs(np.diag(gram)-1))),orthogonal_max_error=float(np.max(abs(gram-np.eye(32)))),eigenvalues=vals.tolist(),eigen_relative_residuals=res,
  repeated_eigen_max_relative_difference=float(np.max(abs(vals-vals2)/np.maximum(abs(vals),1e-30))),repeat_principal_cosines=cosines.tolist(),full_trace=trace,top32_trace=float(vals.sum()),
  explicit_fixture_unit_tests='tests.log',basis_storage_dtype='float32, reloaded into float64 for all checks and outcomes',variance_branch_has_zero_mean_action_jacobian=True)
 passed=checks['orthogonal_max_error']<=ncheck['orthogonal_max'] and max(res)<=ncheck['eigen_relative_residual_max'] and min(cosines)>=ncheck['repeat_min_principal_cosine'] and min(vals)>=-1e-10*max(vals) and vals.sum()<=trace*(1+ncheck['trace_excess_relative'])
 checks['status']='PASS' if passed else 'FAIL';write(OUT/'GN_NUMERICAL_CHECK.json',checks)
 if not passed:status('STOPPED_NUMERICAL_BASIS');return
 # Private streams, no training RNG dependency. Random subspace does not use task losses.
 gen=torch.Generator(device='cuda');gen.manual_seed(2026092305)
 R=torch.randn(P,32,device=device,dtype=torch.float64,generator=gen);R=torch.linalg.qr(R,mode='reduced')[0]
 def rand_orth(basis):
  v=torch.randn(P,device=device,dtype=torch.float64,generator=gen)
  for _ in range(2):v=v-basis@(basis.T@v)
  return v/v.norm()
 random=torch.stack([rand_orth(Q[:,:8]) for _ in range(8)],1)
 complement=torch.stack([rand_orth(Q) for _ in range(8)],1)
 with torch.no_grad():base_out=e.output(e.theta,'old_probe');base_losses={r:e.losses(e.theta,r).detach() for r in ['old_build','old_gradient','old_probe','new_gradient','new_probe']}
 save(OUT/'BASELINE_LOSSES.npz',**{r:v.cpu().numpy() for r,v in base_losses.items()})
 status('H1_PERTURBATIONS');rows=[]
 for kind,dirs in [('GN_TOP',Q[:,:8]),('RANDOM_ORTH_TOP8',random),('COMPLEMENT_TOP32',complement)]:
  for k in range(8):
   v=dirs[:,k];energy=float(v@e.cv(v))
   for scale in proto['scales']:
    for sign in [-1,1]:
     t=time.time();eps=epsilon*scale;delta=eps*sign*v
     with torch.no_grad():diff=e.output(e.theta+delta,'old_probe')-base_out;per=(diff.square()*e.weight).sum(-1);measured=float(per.mean())
     rows.append(dict(direction_type=kind,direction=k,sign=sign,scale=scale,epsilon=eps,perturbation_norm=float(delta.norm()),eigenvalue=float(vals[k]) if kind=='GN_TOP' else None,
      predicted_change=eps*eps*energy,measured_change=measured,ratio=measured/max(eps*eps*energy,1e-300),seconds=time.time()-t))
 table('H1_SENSITIVITY.csv',rows)
 status('H2_FINGERPRINTS');t=time.time();q=[]
 for k in range(32):q.append((e.jv(Q[:,k]).square()*e.weight).sum(-1).cpu().numpy())
 q=np.stack(q,1);meta=read(OUT/'DATA_SPLIT.json')['roles']['old_build']
 save(OUT/'WINDOW_FUNCTIONAL_FINGERPRINTS.npz',q=q,trajectory_id=np.array([r['trajectory_id'] for r in meta]),window_index=np.array([r['window_index'] for r in meta]),task_id=np.full(len(meta),8))
 qn=q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-300);pairs=[]
 for i in range(len(meta)):
  for j in range(i+1,len(meta)):
   a,b=meta[i],meta[j];distance=abs(a['window_index']-b['window_index']);same=a['trajectory_id']==b['trajectory_id']
   kind='CROSS_TRAJECTORY' if not same else ('ADJACENT' if distance==1 else ('SAME_FAR' if distance>=.2*a['trajectory_length'] else None))
   if kind is None:continue
   pairs.append(dict(kind=kind,i=i,j=j,trajectory_i=a['trajectory_id'],trajectory_j=b['trajectory_id'],distance=distance,overlap_fraction=max(0,8-distance)/8 if same else 0,cosine=float(qn[i]@qn[j])))
 table('H2_WINDOW_REDUNDANCY.csv',pairs);timings['fingerprints_seconds']=time.time()-t
 status('H3_GRADIENTS');go=e.grad('old_gradient');gn=e.grad('new_gradient');gop=e.grad('old_probe');gnp=e.grad('new_probe')
 norms={k:float(v.norm()) for k,v in [('old_gradient',go),('new_gradient',gn),('old_probe',gop),('new_probe',gnp)]}
 write(OUT/'GRADIENT_NORMS.json',norms);h3=[];window=[]
 for kind,basis in [('GN',Q),('RANDOM',R)]:
  zo=basis.T@go;zn=basis.T@gn;zop=basis.T@gop;znp=basis.T@gnp
  for k in range(16):
   weak=abs(float(zo[k]))<=1e-6*norms['old_gradient'] or abs(float(zn[k]))<=1e-6*norms['new_gradient']
   label='WEAK' if weak else ('SHARED' if float(zo[k]*zn[k])>0 else 'CONFLICT')
   for scale in proto['scales']:
    t=time.time();eps=epsilon*scale
    # Weak records retained, but no artificial amplification of numerical noise.
    delta=torch.zeros_like(e.theta) if weak else -eps*torch.sign(zn[k])*basis[:,k]
    with torch.no_grad():
     do=e.losses(e.theta+delta,'old_probe')-base_losses['old_probe'];dn=e.losses(e.theta+delta,'new_probe')-base_losses['new_probe']
    row=dict(subspace=kind,direction=k,scale=scale,label=label,weak=weak,z_old=float(zo[k]),z_new=float(zn[k]),eta=None if weak else eps/abs(float(zn[k])),perturbation_norm=float(delta.norm()),
     pred_old_change=float(go@delta),pred_new_change=float(gn@delta),matched_probe_pred_old=float(gop@delta),matched_probe_pred_new=float(gnp@delta),measured_old_change=float(do.mean()),measured_new_change=float(dn.mean()),seconds=time.time()-t)
    h3.append(row)
    if scale==1.:
     for role,diffs in [('old_probe',do),('new_probe',dn)]:
      for i,val in enumerate(diffs.cpu().numpy()):window.append(dict(subspace=kind,direction=k,role=role,index=i,loss_change=float(val)))
 table('H3_DIRECTION_PERTURBATION.csv',h3);table('H3_PER_WINDOW_EFFECTS.csv',window)
 raw=[]
 for scale in proto['scales']:
  d=-epsilon*scale*gn/gn.norm()
  with torch.no_grad():do=float((e.losses(e.theta+d,'old_probe')-base_losses['old_probe']).mean());dn=float((e.losses(e.theta+d,'new_probe')-base_losses['new_probe']).mean())
  raw.append(dict(scale=scale,cosine=float(go@gn/(go.norm()*gn.norm())),pred_old_change=float(go@d),measured_old_change=do,measured_new_change=dn,perturbation_norm=float(d.norm())))
 write(OUT/'RAW_GRADIENT_BASELINE.json',dict(results=raw,limitation='One checkpoint gives one cosine. Three scales are not independent prediction samples; no statistical baseline-superiority claim.'))
 assert torch.equal(e.theta,torch.cat([p.detach().flatten() for p in e.module.parameters()]))
 write(OUT/'EFFICIENCY.json',dict(**timings,total_seconds=time.time()-t0,jvp_count=e.jvps,vjp_count=e.vjps,cv_count=e.cvs,GPU_peak_bytes=torch.cuda.max_memory_allocated(),basis_bytes=(OUT/'OLD_TASK_GN_BASIS.npz').stat().st_size,rank=32,P=P,formal_parameters_unchanged=True,optimizer_steps_executed=0))
 status('MEASUREMENTS_COMPLETE');from .report import main as report;report()
 from .render import main as render;render()
if __name__=='__main__':
 try:main()
 except Exception as exc:
  write(OUT/'STATUS.json',dict(stage='FAILED_STOP',error=repr(exc),traceback=traceback.format_exc(),training_active=False,formal_training_resumed=False));raise
