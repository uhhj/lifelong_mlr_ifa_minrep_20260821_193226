import copy,unittest,torch,numpy as np
from .engine import random_controls,select_D
from gn_h3_subspace_closure_v1.project import project
class Tests(unittest.TestCase):
 def setUp(self):torch.manual_seed(724)
 def test_original_denominator(self):
  x=torch.tensor([2.,3.,7.],requires_grad=True);mask=torch.tensor([1.,0.,1.])
  total=x.square().mean();r=(x.square()*mask).sum()/3;n=(x.square()*(1-mask)).sum()/3
  self.assertTrue(torch.equal(torch.autograd.grad(total,x,retain_graph=True)[0],torch.autograd.grad(r+n,x)[0]))
  self.assertNotEqual(float(r),float(x[[0,2]].square().mean()))
 def test_adam_same_state_zero_none_rollback(self):
  p=torch.nn.Parameter(torch.tensor([2.]));opt=torch.optim.AdamW([p],lr=.01)
  p.grad=torch.ones_like(p);opt.step();state=copy.deepcopy(opt.state_dict());base=p.detach().clone()
  candidates=[]
  for _ in range(2):
   p.data.copy_(base);opt.load_state_dict(copy.deepcopy(state));p.grad=torch.zeros_like(p);opt.step();candidates.append(p.detach().clone())
  self.assertTrue(torch.equal(*candidates));self.assertFalse(torch.equal(candidates[0],base))
  p.data.copy_(base);opt.load_state_dict(copy.deepcopy(state));p.grad=None;opt.step();self.assertTrue(torch.equal(p,base))
 def test_projector_and_rotation(self):
  U=torch.linalg.qr(torch.randn(14,4,dtype=torch.float64))[0];x=torch.randn(14,dtype=torch.float64);y=torch.randn_like(x)
  Q=torch.linalg.qr(torch.randn(4,4,dtype=torch.float64))[0];v=project(U,x)
  self.assertTrue(torch.allclose(project(U,v),v));self.assertTrue(torch.allclose(project(U@Q,x),v))
  self.assertAlmostEqual(float(y@v),float(x@project(U,y)))
 def test_energy_decomposition(self):
  J=torch.randn(8,14,dtype=torch.float64);A=J.T@J;U=torch.linalg.qr(torch.randn(14,4,dtype=torch.float64))[0]
  x=torch.randn(14,dtype=torch.float64);v=project(U,x);w=x-v
  self.assertTrue(torch.allclose(x,v+w));self.assertAlmostEqual(float(x@A@x),float(v@A@v+w@A@w+2*v@A@w),places=10)
 def test_factorial_identity(self):
  r=torch.randn(7);v=torch.randn(7);w=torch.randn(7)
  loss=lambda x:float(x.square().sum())
  interaction=loss(r+v+w)-loss(r+w)-loss(r+v)+loss(r)
  self.assertAlmostEqual(interaction,2*float(v@w),places=5)
 def test_matched_random(self):
  x=torch.randn(40,dtype=torch.float64);U=torch.linalg.qr(torch.randn(40,4,dtype=torch.float64))[0];v=project(U,x)
  ds=random_controls(x,v,4,[1,2,3])
  for k in ['C2_0','C2_1','C2_2']:
   t=ds[k];self.assertAlmostEqual(float(t.norm()),float(v.norm()),places=12);self.assertAlmostEqual(float(x@t),float(t@t),places=12)
   self.assertTrue(torch.allclose(project((t/t.norm())[:,None],x),t))
  self.assertAlmostEqual(float(ds['C3'].norm()),float(v.norm()),places=12)
 def test_keep_remove_head_only(self):
  source={'decoder':torch.randn(3),'head':torch.randn(4)};copy1=copy.deepcopy(source);copy1['head']=(source['head'].double()-.01).float()
  self.assertTrue(torch.equal(source['decoder'],copy1['decoder']));self.assertFalse(torch.equal(source['head'],copy1['head']))
 def test_reuse_without_update(self):
  weight=torch.tensor(2.);x=torch.tensor(3.);self.assertEqual(float(weight*x),6.);self.assertEqual(float(weight*x),6.)
 def test_same_sensitivity_opposite_residual(self):
  theta=torch.tensor(0.,requires_grad=True)
  a=torch.autograd.grad((theta-1)**2,theta,retain_graph=True)[0];b=torch.autograd.grad((theta+1)**2,theta)[0]
  self.assertLess(float(a*b),0);self.assertEqual(float(torch.autograd.functional.jacobian(lambda t:t,theta)),1.)
 def test_D_only_and_tie(self):
  rows=[]
  from .common import METHODS
  names=['KEEP']+[m+s for m in METHODS for s in ['_HALF','_REMOVE']]
  for name in names:
   for role,task in [('D_old',6),('D_new',7)]:
    for q in ['a','b']:rows.append(dict(candidate=name,role=role,task_id=task,trajectory_id=q,nll=1.))
  before=select_D(rows,{'6':{'tau_num':1e-7},'7':{'tau_num':1e-7}})
  self.assertTrue(all(v['candidate']=='KEEP' for v in before.values()))
  self.assertEqual(before,select_D(rows,{'6':{'tau_num':1e-7},'7':{'tau_num':1e-7}}))
 def test_explicit_matrix_free_fixture(self):
  J=torch.randn(3,7,dtype=torch.float64);theta=torch.randn(7,dtype=torch.float64);w=torch.tensor([1.,2.,3.],dtype=torch.float64);v=torch.randn(7,dtype=torch.float64)
  jv=torch.autograd.functional.jvp(lambda t:J@t,theta,v)[1]
  out=torch.autograd.functional.vjp(lambda t:J@t,theta,w*jv)[1]
  self.assertTrue(torch.allclose(out,(J.T@torch.diag(w)@J)@v))
 def test_rng_state_restore(self):
  before=torch.get_rng_state();torch.randn(7);torch.set_rng_state(before);self.assertTrue(torch.equal(before,torch.get_rng_state()))
 def test_paired_block_bootstrap_constant_gains(self):
  from .report import bootstrap
  from .common import METHODS
  starts=[];allmeans={};choices={}
  for seed in [45,46,47]:
   for task in [7,8]:
    for j in range(6):
     sid=str((seed,task,j));starts.append(dict(start_id=sid,seed=seed,task_id=task,cell_order=j))
     a={};d={}
     for role in ['old','new']:
      for demo in ['a','b']:
       for name in ['KEEP']+[m+'_REMOVE' for m in METHODS]:
        benefit=.1 if name=='GN_REMOVE' else .04 if name!='KEEP' else 0.
        a['T_'+role,name,demo]=2.-benefit;d['D_'+role,name,demo]=2.-benefit
     allmeans[sid]={'D':d,'T':a};choices[sid]={m:dict(candidate=m+'_REMOVE') for m in METHODS}
  result=bootstrap(starts,allmeans,choices)
  self.assertAlmostEqual(result['GN'][0],.1,places=12)
  self.assertAlmostEqual(result['GN_minus_C2_simultaneous97_5'][0],.06,places=12)
  self.assertAlmostEqual(result['GN_minus_C3_simultaneous97_5'][1],.06,places=12)
 def test_control_materialization_metadata(self):
  from .control_metadata import records
  h=torch.tensor([1.,-2.,.3],dtype=torch.float32);x=torch.tensor([.01,-.02,.03],dtype=torch.float64)
  dirs={'GN':x*.2,'C3':x*.3};eng={'conditional_increment_norm':float(x.norm())*2}
  out=records(h,x,dirs,eng,2)
  self.assertAlmostEqual(out[0]['rho_head'],.04,places=12)
  self.assertAlmostEqual(out[0]['rho_all'],.01,places=12)
  self.assertAlmostEqual(out[1]['realized_delete_norm'],float((h.double()-(h.double()-dirs['C3']).float().double()).norm()),places=12)
if __name__=='__main__':unittest.main()
