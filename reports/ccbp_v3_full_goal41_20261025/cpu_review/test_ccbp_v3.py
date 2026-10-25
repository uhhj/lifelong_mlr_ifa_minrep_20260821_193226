"""Local synthetic invariants, not model or robot validation."""
from __future__ import annotations
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from losses_ccbp_v3 import (
    auxiliary_v3, expert_cost, action_normalizer, directional_anchor,
    segment_distance_squared,
)

torch.set_num_threads(1)
torch.manual_seed(20261025)
DTYPE = torch.float64

def scalar(x):
    return float(x.detach()) if isinstance(x, torch.Tensor) else float(x)
checks = {}

def make(b=8, k=5, grad=True):
    return (torch.randn(b, k, 7, dtype=DTYPE).tanh().requires_grad_(grad),
            (0.01+0.03*torch.rand(b, k, 7, dtype=DTYPE)).requires_grad_(grad),
            torch.randn(b, k, dtype=DTYPE).requires_grad_(grad))

q, teacher = make(), make()
y = torch.randn(8,7,dtype=DTYPE).tanh()
scales = torch.tensor([0.46725568,0.1,1.0],dtype=DTYPE)
w = torch.rand(8,dtype=DTYPE,requires_grad=True)

# Cost is exactly distance to the expert point in the stated coordinates.
z = torch.cat((q[0]*action_normalizer(scales),q[1]*action_normalizer(scales)), -1)
e = torch.cat((y*action_normalizer(scales),torch.zeros_like(y)), -1)[:,None]
err = scalar(((z-e).square().sum(-1)-expert_cost(q[0],q[1],y,scales)).abs().max())
assert err < 1e-12
checks['cost_coordinate_identity_max_abs'] = err

# All progress along the correction segment is unpenalized.
t0 = torch.randn(32,14,dtype=DTYPE)
e0 = torch.randn(32,14,dtype=DTYPE)
fractions = torch.linspace(0,1,11,dtype=DTYPE)
points = t0[:,None]+fractions[None,:,None]*(e0-t0)[:,None]
r,_ = segment_distance_squared(points,t0[:,None],e0[:,None])
assert scalar(r.max()) < 1e-24
checks['correction_path_max_anchor'] = scalar(r.max())

# General projection geometry: 0 <= anchor <= expert_cost and nonopposing
# OUTPUT-COORDINATE gradients (NOT a neural-parameter or cross-task guarantee).
start = torch.randn(4096,14,dtype=DTYPE)
end = torch.randn(4096,14,dtype=DTYPE)
point = (3*torch.randn(4096,14,dtype=DTYPE)).requires_grad_(True)
r,_ = segment_distance_squared(point,start,end)
c = (point-end).square().sum(-1)
gr = torch.autograd.grad(r.sum(),point,retain_graph=True)[0]
gc = torch.autograd.grad(c.sum(),point)[0]
dot = (gr*gc).sum(-1)
assert scalar(r.min()) >= 0 and scalar((r-c).max()) < 1e-10
assert scalar(dot.min()) >= -1e-10
checks['projection_test_vectors'] = len(point)
checks['anchor_minus_expert_cost_max'] = scalar((r-c).max())
checks['output_gradient_dot_min'] = scalar(dot.min())

# Degenerate segment remains finite and equals distance to its single point.
a=torch.randn(6,14,dtype=DTYPE)
x=torch.randn(6,14,dtype=DTYPE,requires_grad=True)
r,t=segment_distance_squared(x,a,a)
assert torch.equal(r,(x-a).square().sum(-1)) and bool((t==0).all())
checks['coincident_endpoint_handling'] = True

# Above/below segment in a simple 2D slice.
pts=torch.tensor([[.5,0.],[.5,1.],[-1.,0.],[2.,0.]],dtype=DTYPE)
r,_=segment_distance_squared(pts,torch.zeros(2,dtype=DTYPE),torch.tensor([1.,0.],dtype=DTYPE))
assert torch.equal(r,torch.tensor([0.,1.,1.,1.],dtype=DTYPE))
checks['backward_sideways_overshoot_examples'] = r.tolist()

# Real-shaped [B,5,7] parameter gradients; no teacher or weight gradient.
loss,st,terms = auxiliary_v3(q,y,teacher,scales,w)
loss.backward()
assert all(v.grad is not None and bool(torch.isfinite(v.grad).all()) and scalar(v.grad.abs().sum())>0 for v in q)
assert all(v.grad is None for v in teacher) and w.grad is None
checks['student_gradient_l1'] = [scalar(v.grad.abs().sum()) for v in q]
checks['teacher_and_weight_frozen'] = True

# Exact C parity when directional retention is switched off.
q1=tuple(v.detach().clone().requires_grad_(True) for v in q)
q2=tuple(v.detach().clone().requires_grad_(True) for v in q)
l1,_,_=auxiliary_v3(q1,y,teacher,scales,w.detach(),anchor_strength=0.)
mu,sd,z=q2
cc=expert_cost(mu,sd,y,scales);ix=torch.arange(len(y));win=z.detach().argmax(-1)
good=F.one_hot(cc.detach().argmin(-1),5).bool()
rr=F.relu(.1+z.masked_fill(good,-1e9).max(-1).values-z.masked_fill(~good,-1e9).max(-1).values)
l2=(cc[ix,win]+.1*rr).mean()
g1=torch.autograd.grad(l1,q1);g2=torch.autograd.grad(l2,q2)
assert torch.equal(l1,l2) and all(torch.equal(a,b) for a,b in zip(g1,g2))
checks['anchor_off_exact_C_loss_and_gradients'] = True

# Component permutation for unique expert costs and logits.
perm=torch.tensor([3,0,4,1,2])
qp=tuple(v.detach()[:,perm].requires_grad_(True) for v in q)
lp,_,_=auxiliary_v3(qp,y,teacher,scales,w.detach())
lo,_,_=auxiliary_v3(tuple(v.detach() for v in q),y,teacher,scales,w.detach())
assert torch.allclose(lp,lo,atol=1e-12,rtol=1e-12)
checks['unique_component_permutation_loss_invariant'] = True

# Tiny sigmas have no inverse-variance denominator.
qs=list(make()); qs[1]=torch.full((8,5,7),1e-4,dtype=DTYPE,requires_grad=True)
ls,_,_=auxiliary_v3(tuple(qs),y,teacher,scales,w.detach())
ls.backward()
assert bool(torch.isfinite(ls)) and all(bool(torch.isfinite(v.grad).all()) for v in qs)
checks['tiny_sigma_loss_finite'] = scalar(ls)

# FP32 pass and empty-batch differentiable zero.
q32=tuple(v.detach().float().requires_grad_(True) for v in q)
t32=tuple(v.detach().float() for v in teacher)
l32,_,_=auxiliary_v3(q32,y.float(),t32,scales.float(),w.detach().float())
l32.backward();assert all(bool(torch.isfinite(v.grad).all()) for v in q32)
qe,te=make(0),make(0)
le,_,_=auxiliary_v3(qe,y[:0],te,scales,w.detach()[:0]);le.backward()
assert scalar(le)==0. and all(v.grad is not None for v in qe)
checks['fp32_and_empty_batch'] = True

# Reject invalid scales rather than silently altering the experimental metric.
try:
    auxiliary_v3(q,y,teacher,torch.zeros(3,dtype=DTYPE),w.detach())
    raise AssertionError('Invalid scale was not rejected')
except ValueError:
    checks['invalid_scale_rejected'] = True

out={'status':'PASS_LOCAL_SYNTHETIC_ONLY','torch_version':torch.__version__,
     'real_checkpoints_loaded':0,'optimizer_updates':0,'environment_steps':0,
     'notes':['Output-coordinate gradient property does not imply parameter-gradient compatibility.',
              'No claim of closed-loop benefit, novelty, convergence or physical feasibility.',
              'C parity preserves first-index ties, not arbitrary-index symmetry at exact cost ties.'],
     'checks':checks}
Path(__file__).with_name('synthetic_test_results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
