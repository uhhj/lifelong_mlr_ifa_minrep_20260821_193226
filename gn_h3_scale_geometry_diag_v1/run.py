import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import time,traceback,numpy as np,torch
from scipy.sparse.linalg import LinearOperator,eigsh
from .common import *
from .fisher import ModelFisher,sample

def status(stage,**kw):
 write(OUT/'STATUS.json',dict(stage=stage,training_active=False,formal_training_resumed=False,**kw));print(stage,kw,flush=True)
def main():
 if (OUT/'SUMMARY.json').exists():print('Completed experiment preserved');return
 p=read(OUT/'PROTOCOL.json');t0=time.time();status('INITIALIZING')
 import gn_functional_geometry_v1.engine as previous
 previous.OUT=OUT # Explicit reuse of the diagnostic loader, writing only to this new report directory.
 e=previous.load();theta=e.theta;device=theta.device;P=len(theta);torch.cuda.reset_peak_memory_stats()
 assert sha(p['mean_basis_path'])==p['mean_basis_sha256'];assert sha(OUT/'DATA_SPLIT.json')==p['split_sha256']
 Umean=torch.tensor(np.load(p['mean_basis_path'])['U'][:,:16],device=device,dtype=torch.float64)
 if not (OUT/'FISHER_MODEL_DRAWS.npz').exists():
  before=torch.cuda.get_rng_state().clone();draws,modes=sample(e);assert torch.equal(before,torch.cuda.get_rng_state())
  save(OUT/'FISHER_MODEL_DRAWS.npz',actions=draws.cpu().numpy(),mixture_indices=modes.cpu().numpy())
 else:draws=torch.tensor(np.load(OUT/'FISHER_MODEL_DRAWS.npz')['actions'],device=device,dtype=torch.float64)
 f=ModelFisher(e,draws);op=LinearOperator((P,P),matvec=lambda v:f.mv(v).cpu().numpy(),dtype=np.float64)
 status('FISHER_BASIS',samples=len(f.a));t=time.time()
 if not (OUT/'NLL_MODEL_FISHER_BASIS.npz').exists():
  vals,U=eigsh(op,k=16,ncv=40,tol=1e-6,maxiter=500,v0=np.random.RandomState(2026092307).normal(size=P));ix=np.argsort(vals)[::-1];vals=vals[ix];U=U[:,ix].astype(np.float32)
  save(OUT/'NLL_MODEL_FISHER_BASIS.npz',U=U,eigenvalues=vals,parameter_names=np.array(e.names),parameter_sizes=np.array(e.sizes))
 else:
  z=np.load(OUT/'NLL_MODEL_FISHER_BASIS.npz');U=z['U'];vals=z['eigenvalues']
 basis_seconds=time.time()-t;status('FISHER_NUMERICAL_REPEAT',basis_seconds=basis_seconds);t=time.time()
 vals2,V=eigsh(op,k=16,ncv=40,tol=1e-6,maxiter=500,v0=np.random.RandomState(2026092308).normal(size=P));ix=np.argsort(vals2)[::-1];vals2=vals2[ix];V=V[:,ix]
 cosines=np.linalg.svd(U.T@V,compute_uv=False);del V
 UF=torch.tensor(U,device=device,dtype=torch.float64);del U
 gram=(UF.T@UF).cpu().numpy();res=[float((f.mv(UF[:,k])-vals[k]*UF[:,k]).norm())/abs(vals[k]) for k in range(16)]
 numerical=dict(orthogonal_max_error=float(abs(gram-np.eye(16)).max()),relative_eigen_residuals=res,eigenvalues=vals.tolist(),repeat_principal_cosines=cosines.tolist(),repeat_eigen_relative_error=float(np.max(abs(vals-vals2)/abs(vals))),
  all_finite=bool(np.isfinite(vals).all() and np.isfinite(res).all()),eigenvalues_sorted=bool(np.all(np.diff(vals)<=0)),all_positive=bool(min(vals)>0),sample_set_sha256=sha(OUT/'FISHER_MODEL_DRAWS.npz'),
  source='Model-generated actions detached at theta0; uncentered score second moment, full official mixture log density; not expert-NLL Hessian')
 ok=numerical['all_finite'] and numerical['all_positive'] and numerical['eigenvalues_sorted'] and numerical['orthogonal_max_error']<2e-5 and max(res)<.002 and min(cosines)>.99
 numerical['status']='PASS' if ok else 'FAIL';write(OUT/'FISHER_NUMERICAL_CHECK.json',numerical)
 if not ok:status('STOPPED_NUMERICAL_CHECK');return
 check_seconds=time.time()-t;status('H3_TWO_BY_FOUR');t=time.time()
 grad={role:e.grad(role) for role in ['old_gradient','new_gradient','old_probe','new_probe']}
 with torch.no_grad():base={role:e.losses(theta,role).detach() for role in ['old_probe','new_probe']}
 rows=[];windows=[]
 for kind,basis in [('MEAN_GN',Umean),('MODEL_FISHER_MC8',UF)]:
  projections={key:basis.T@v for key,v in grad.items()}
  for k in range(16):
   zn=projections['new_gradient'][k];weak=abs(float(zn))<=1e-6*float(grad['new_gradient'].norm());sign=torch.sign(zn)
   for scale in p['scales']:
    start=time.time();eps=p['epsilon0']*scale;delta=torch.zeros_like(theta) if weak else -eps*sign*basis[:,k]
    with torch.no_grad():
     losses={role:e.losses(theta+delta,role)-base[role] for role in base}
    row=dict(geometry=kind,direction=k,scale=scale,epsilon=eps,norm=float(delta.norm()),weak_new_projection=weak,z_new=float(zn),
     z_old_relation=float(projections['old_gradient'][k]),z_old_probe=float(projections['old_probe'][k]),
     pred_old_same_probe=float(grad['old_probe']@delta),pred_old_cross_trajectory=float(grad['old_gradient']@delta),
     pred_new_same_probe=float(grad['new_probe']@delta),pred_new_cross_trajectory=float(grad['new_gradient']@delta),
     measured_old=float(losses['old_probe'].mean()),measured_new=float(losses['new_probe'].mean()),seconds=time.time()-start)
    rows.append(row)
    for role,diffs in losses.items():
     for i,value in enumerate(diffs.cpu().numpy()):windows.append(dict(geometry=kind,direction=k,scale=scale,role=role,index=i,delta_nll=float(value)))
 table('DIRECTION_EFFECTS.csv',rows);table('PER_WINDOW_EFFECTS.csv',windows)
 # Match previous primary-scale measurements, not merely the saved basis.
 import csv
 old=list(csv.DictReader((OLD/'H3_DIRECTION_PERTURBATION.csv').open()));parity=[]
 for row in rows:
  if row['geometry']!='MEAN_GN' or row['scale'] not in (1.,.25):continue
  prior=next(r for r in old if r['subspace']=='GN' and int(r['direction'])==row['direction'] and float(r['scale'])==row['scale'])
  parity.append(dict(direction=row['direction'],scale=row['scale'],old_nll_abs_diff=abs(row['measured_old']-float(prior['measured_old_change'])),new_nll_abs_diff=abs(row['measured_new']-float(prior['measured_new_change'])),matched_prediction_abs_diff=abs(row['pred_old_same_probe']-float(prior['matched_probe_pred_old']))))
 write(OUT/'PARENT_PARITY.json',dict(status='PASS' if max(max(z[k] for k in ['old_nll_abs_diff','new_nll_abs_diff','matched_prediction_abs_diff']) for z in parity)<1e-9 else 'FAIL',rows=parity))
 assert read(OUT/'PARENT_PARITY.json')['status']=='PASS'
 assert torch.equal(theta,torch.cat([v.detach().flatten() for v in e.module.parameters()]))
 write(OUT/'EFFICIENCY.json',dict(basis_seconds=basis_seconds,basis_repeat_and_validation_seconds=check_seconds,perturbations_seconds=time.time()-t,total_seconds=time.time()-t0,fisher_operator_calls=f.calls,
  peak_GPU_bytes=torch.cuda.max_memory_allocated(),basis_bytes=(OUT/'NLL_MODEL_FISHER_BASIS.npz').stat().st_size,training_active=False,optimizer_steps_executed=0))
 status('MEASUREMENTS_COMPLETE')
if __name__=='__main__':
 try:main()
 except Exception as exc:status('FAILED_STOP',error=repr(exc),traceback=traceback.format_exc());raise
