import json,hashlib,time,os,ast
from pathlib import Path
torch.set_num_threads(1);torch.manual_seed(730041)
S=Path('/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022/OBJECT_42/source')
import sys
sys.path.insert(0,str(S))
from libero.lifelong.models.policy_head import GMMHead
T=nn.Linear(32,512);G=GMMHead(input_size=512,output_size=7,num_modes=5,min_std=1e-4);student=Student()
x=torch.randn(10,32);a=torch.randn(10,7)*.1
h=T(x);hp,c,g,delta=student.deploy(h)
assert torch.equal(h,hp) and float(delta.abs().max())==0 and torch.allclose(g,torch.full_like(g,.1),atol=1e-7)
assert all(torch.equal(u,v) for u,v in zip(G.forward_fn(h),G.forward_fn(hp)))
saved=torch.get_rng_state();aa=G(h,train=False).sample();torch.set_rng_state(saved);bb=G(hp,train=False).sample();assert torch.equal(aa,bb)
bc=-G(hp).log_prob(a).mean();u_grad=torch.autograd.grad(bc,list(student.deploy.U.parameters()),retain_graph=True);assert sum(float(v.square().sum()) for v in u_grad)>0
initial_R_grad=torch.autograd.grad(bc,list(student.deploy.R.parameters()),allow_unused=True,retain_graph=True);assert all(v is None or float(v.abs().max())==0 for v in initial_R_grad)
with torch.no_grad():student.deploy.U.weight.normal_(0,.01)
h=T(x);hp,c,g,delta=student.deploy(h)
groups={'Temporal':list(T.parameters()),'GMM':list(G.parameters()),'R':list(student.deploy.R.parameters()),'U':list(student.deploy.U.parameters()),'gate':list(student.deploy.gate.parameters()),'E':list(student.E.parameters()),'D':list(student.D.parameters())}
flat=[p for ps in groups.values() for p in ps]
expected={'BC':{'Temporal','GMM','R','U','gate'},'IFA':{'Temporal'},'effect':{'R','E','D'},'ST':{'R','E'}}
losses={'BC':-G(hp).log_prob(a).mean(),'IFA':h.square().mean(),'effect':F.smooth_l1_loss(student.D(student.condition(c,a)),torch.randn(10,7))}
keys=F.normalize(torch.randn(5,128),dim=-1).detach();z=F.normalize(student.condition(c,a),dim=-1);scores=z@keys.T/.1;losses['ST']=(torch.logsumexp(scores,1)-scores[:,0]).mean()
proof={}
for name,loss in losses.items():
 grads=torch.autograd.grad(loss,flat,allow_unused=True,retain_graph=True);offset=0;norms={}
 for group,ps in groups.items():
  gs=grads[offset:offset+len(ps)];offset+=len(ps);norms[group]=sum(float(v.double().square().sum()) for v in gs if v is not None)
 assert {k for k,v in norms.items() if v>0}==expected[name],(name,norms)
 proof[name]=norms
assert not keys.requires_grad
ratio=delta.square().mean(-1).sqrt()/h.square().mean(-1).sqrt();assert float(ratio.max())<=.200001
assert set(map(id,student.control_parameters())).isdisjoint(map(id,student.representation_parameters()))
parents={s:Path('/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed%d_20261019'%s) for s in (41,42,43)}
cfgs={s:json.loads((p/'resolved_ra_coreset.json').read_text()) for s,p in parents.items()}
sizes={}
for label,path in [('ALIGN41',Path('/data/mlr_ifa/experiments/CAETv3_ALIGN_OBJECT_seed41_20261022_collate01')),('ACTIVE_ALIGN42',Path('/data/mlr_ifa/experiments/ALIGN_OBJECT_seed42_20261022'))]:
 sizes[label]=sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink() and 'features' not in p.relative_to(path).parts and 'bindings' not in p.relative_to(path).parts and 'selection_preflight' not in p.relative_to(path).parts and 'offline_selection' not in p.relative_to(path).parts)
disk={m:dict(ordinary=(v:=os.statvfs(m)).f_bavail*v.f_frsize,root=v.f_bfree*v.f_frsize) for m in ('/data','/')}
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='CPU_AUTOGRAD_AND_ACTUAL_GMM_IDENTITY_PASS',observed_unix=time.time(),gradient_norm2=proof,initial_delta_zero=True,initial_actual_GMM_parameters_and_sample_exact=True,initial_U_BC_gradient_nonzero=True,initial_R_BC_gradient_zero_expected=True,teacher_key_requires_grad=False,residual_RMS_ratio_max=float(ratio.max()),deploy_keys=len(student.deploy.state_dict()),student_keys=len(student.state_dict()),original_complete_run_bytes=sizes,disk=disk,config_optimizer=cfgs[41]['train'],base_schedule_source=(S/'libero/lifelong/algos/base.py').read_text(),cuda_initialized=False,optimizer_calls=0,environment_steps=0,server_writes=0)))
