import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import time,traceback,numpy as np,torch
from scipy.sparse.linalg import LinearOperator,eigsh
from .common import *
from .project import project,orthonormal_random

def status(stage,**kw):
 write(OUT/'STATUS.json',dict(stage=stage,training_active=False,formal_training_resumed=False,**kw));print(stage,kw,flush=True)
def main():
 if (OUT/'SUMMARY.json').exists():print('Completed results preserved');return
 t0=time.time();p=read(OUT/'PROTOCOL.json');status('INITIALIZING')
 for asset in p['assets'].values():assert sha(asset['path'])==asset['sha256']
 assert sha(OUT/'DATA_SPLIT.json')==p['split_sha256']
 import gn_functional_geometry_v1.engine as loader
 loader.OUT=OUT;e=loader.load();theta=e.theta;device=theta.device;P=len(theta);torch.cuda.reset_peak_memory_stats()
 Umean=torch.tensor(np.load(p['assets']['mean_basis']['path'])['U'][:,:21],device=device,dtype=torch.float64)
 UF16=torch.tensor(np.load(p['assets']['fisher_basis16']['path'])['U'],device=device,dtype=torch.float64)
 R=orthonormal_random(P,21,2026092310,device)
 grads={role:e.grad(role) for role in ['old_gradient','new_gradient','old_probe','new_probe']}
 with torch.no_grad():base={role:e.losses(theta,role).detach() for role in ['old_probe','new_probe']}
 status('A_DIRECTIONAL_CLOSURE');ta=time.time();directions=[]
 for kind,U in [('MEAN_GN',Umean[:,:16]),('MODEL_FISHER',UF16)]:
  for k in range(16):directions.append((kind,k,U[:,k]/U[:,k].norm()))
 for name,v in [('OLD_PROBE_GRADIENT',grads['old_probe']),('NEW_GRADIENT',grads['new_gradient']),('RANDOM',R[:,0])]:directions.append((name,0,v/v.norm()))
 rows=[];groups=[]
 for kind,k,v in directions:
  for role in ['old_probe','new_probe']:
   analytic=float(grads[role]@v);gnorm=float(grads[role].norm());denom=max(abs(analytic),1e-8*gnorm);local=[]
   for factor in p['A']['h_factors']:
    h=p['epsilon0']*factor
    with torch.no_grad():plus=e.losses(theta+h*v,role);minus=e.losses(theta-h*v,role);fd=float((plus-minus).mean()/(2*h))
    rel=abs(fd-analytic)/max(denom,1e-300)
    row=dict(direction_type=kind,direction=k,task=role,h_factor=factor,h=h,analytic=analytic,finite_difference=fd,absolute_error=abs(fd-analytic),relative_error=rel,
     denominator=denom,gradient_norm=gnorm,weak_derivative=abs(analytic)<1e-8*gnorm,closed=bool(rel<=1e-4),v_norm=float(v.norm()))
    rows.append(row);local.append(row)
   pairs=[(a['h_factor'],b['h_factor']) for a,b in zip(local,local[1:]) if a['h_factor']<=.001 and a['closed'] and b['closed']]
   groups.append(dict(direction_type=kind,direction=k,task=role,status='PASS' if pairs else 'FAIL',closed_adjacent_scales=pairs,min_relative_error=min(r['relative_error'] for r in local),analytic=analytic,weak_derivative=local[0]['weak_derivative']))
  table('A_DIRECTIONAL_CLOSURE.csv',rows)
 closure=all(g['status']=='PASS' for g in groups);write(OUT/'A_REPORT.json',dict(status='PASS' if closure else 'FAIL',groups=groups,passed_groups=sum(g['status']=='PASS' for g in groups),total_groups=len(groups),seconds=time.time()-ta,criterion=p['A']['pass_rule']))
 if not closure:
  status('STOPPED_A_CLOSURE_FAILED');return
 status('A_PASS',groups=len(groups));del directions
 # Only now extend the existing Fisher span from 16 to 21 using identical draws.
 from gn_h3_scale_geometry_diag_v1.fisher import ModelFisher
 draws=torch.tensor(np.load(p['assets']['fisher_draws']['path'])['actions'],device=device,dtype=torch.float64);f=ModelFisher(e,draws)
 op=LinearOperator((P,P),matvec=lambda v:f.mv(v).cpu().numpy(),dtype=np.float64);te=time.time();status('B_FISHER_EXTENSION')
 vals,V=eigsh(op,k=21,ncv=48,tol=1e-6,maxiter=500,v0=np.random.RandomState(2026092309).normal(size=P));ix=np.argsort(vals)[::-1];vals=vals[ix];V=V[:,ix]
 cos=np.linalg.svd(UF16.cpu().numpy().T@V[:,:16],compute_uv=False)
 extra=torch.tensor(V[:,16:21],device=device,dtype=torch.float64);del V
 extra=extra-project(UF16,extra);extra=torch.linalg.qr(extra,mode='reduced')[0]
 extra=extra.float().double();UF=torch.cat([UF16,extra],1)
 residuals=[float((f.mv(UF[:,k])-vals[k]*UF[:,k]).norm())/abs(vals[k]) for k in range(21)]
 numerical=dict(retained_top16_min_principal_cosine=float(min(cos)),relative_eigen_residuals=residuals,orthogonal_max_error=float((UF.T@UF-torch.eye(21,device=device)).abs().max()),eigenvalues=vals.tolist(),samples_unchanged=True,old16_columns_preserved=True,seconds=time.time()-te)
 numerical['status']='PASS' if min(cos)>.99 and max(residuals)<.002 and numerical['orthogonal_max_error']<2e-5 else 'FAIL'
 write(OUT/'FISHER_EXTENSION_CHECK.json',numerical)
 if numerical['status']!='PASS':status('STOPPED_INVALID_EXTENSION');return
 save(OUT/'FISHER_EXTRA_FIVE.npz',U_extra=extra.float().cpu().numpy(),eigenvalues=vals,original16_path=np.array(p['assets']['fisher_basis16']['path']),original16_sha256=np.array(p['assets']['fisher_basis16']['sha256']))
 status('B_SUBSPACE_MEASUREMENTS');tb=time.time();effects=[];perwindow=[];rotations=[]
 for kind,U in [('MEAN_GN',Umean),('MODEL_FISHER_MC8',UF),('RANDOM',R)]:
  for rank in p['B']['ranks']:
   basis=U[:,:rank];proj=project(basis,grads['new_gradient']);pnorm=float(proj.norm());gnorm=float(grads['new_gradient'].norm());score=float(grads['old_gradient']@proj)
   O=orthonormal_random(rank,rank,2026092311+rank,device);rot_proj=project(basis@O,grads['new_gradient']);rot_score=float(grads['old_gradient']@rot_proj)
   vector_error=float((rot_proj-proj).norm())/max(pnorm,1e-300);score_error=abs(rot_score-score)/max(abs(score),1e-8*float(grads['old_gradient'].norm())*pnorm,1e-300)
   rotation=dict(geometry=kind,rank=rank,relative_projected_vector_difference=vector_error,relative_score_difference=score_error,passed=bool(vector_error<=1e-10 and score_error<=1e-10))
   rotations.append(rotation);assert rotation['passed'],'BASIS_INVARIANCE_FAILED'
   weak=pnorm<=1e-8*gnorm
   for scale in p['B']['scales']:
    eps=p['epsilon0']*scale;delta=torch.zeros_like(theta) if weak else -eps*proj/pnorm
    with torch.no_grad():diffs={role:e.losses(theta+delta,role)-base[role] for role in base}
    row=dict(geometry=kind,rank=rank,scale=scale,score_old_relation=score,score_old_probe=float(grads['old_probe']@proj),projected_new_gradient_norm=pnorm,projection_energy_fraction=pnorm**2/gnorm**2,
     eta=None if weak else eps/pnorm,perturbation_norm=float(delta.norm()),weak_projection=weak,
     pred_old_cross=float(grads['old_gradient']@delta),pred_old_same_probe=float(grads['old_probe']@delta),pred_new_cross=float(grads['new_gradient']@delta),pred_new_same_probe=float(grads['new_probe']@delta),
     measured_old=float(diffs['old_probe'].mean()),measured_new=float(diffs['new_probe'].mean()))
    effects.append(row)
    for role,values in diffs.items():
     for i,v in enumerate(values.cpu().numpy()):perwindow.append(dict(geometry=kind,rank=rank,scale=scale,role=role,index=i,delta_nll=float(v)))
 table('B_SUBSPACE_EFFECTS.csv',effects);table('B_PER_WINDOW_EFFECTS.csv',perwindow)
 write(OUT/'BASIS_INVARIANCE.json',dict(status='PASS' if all(r['passed'] for r in rotations) else 'FAIL',rows=rotations))
 assert torch.equal(theta,torch.cat([v.detach().flatten() for v in e.module.parameters()]))
 write(OUT/'EFFICIENCY.json',dict(total_seconds=time.time()-t0,A_seconds=read(OUT/'A_REPORT.json')['seconds'],Fisher_extension_seconds=numerical['seconds'],B_seconds=time.time()-tb,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_basis_bytes=(OUT/'FISHER_EXTRA_FIVE.npz').stat().st_size,optimizer_steps_executed=0))
 status('MEASUREMENTS_COMPLETE')
if __name__=='__main__':
 try:main()
 except Exception as exc:status('FAILED_STOP',error=repr(exc),traceback=traceback.format_exc());raise
