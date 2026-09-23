import unittest,copy,tempfile
import numpy as np,torch
from .analysis import metrics,clean
from . import engine
from .common import *
class Policy(torch.nn.Module):
 def __init__(self):
  super().__init__();self.layer=torch.nn.Linear(3,2)
 def compute_loss(self,data,*args):return (self.layer(data['x'])-data['y']).square().mean(),None
class Algo:
 def __init__(self):
  self.policy=Policy();self.optimizer=torch.optim.AdamW(self.policy.parameters(),lr=.003,weight_decay=.1)
  self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,lambda step:1-step/100)
  self._optimizer_step_count=0;self._scheduler_step_count=0;self.loss_scale=1.
  from types import SimpleNamespace
  self.cfg=SimpleNamespace(train=SimpleNamespace(grad_clip=.2))
 def map_tensor_to_device(self,x):return x
 def optimizer_step_and_maybe_schedule(self):
  self.optimizer.step();self.scheduler.step();self._optimizer_step_count+=1;self._scheduler_step_count+=1
class Tests(unittest.TestCase):
 def test_candidate_matches_native_with_history(self):
  from gradient_relation_experiment import kernel as g
  a=Algo();data={'x':torch.randn(7,3),'y':torch.randn(7,2)}
  for _ in range(4):
   a.optimizer.zero_grad();a.policy.compute_loss(data)[0].backward();torch.nn.utils.clip_grad_norm_(a.policy.parameters(),.2);a.optimizer_step_and_maybe_schedule()
  base=g.capture(a);p=engine.proposal(a,[data,[]],base)
  self.assertTrue(all(g.compare_states(base,g.capture(a)).values()))
  a.optimizer.zero_grad();a.policy.compute_loss(data)[0].backward();torch.nn.utils.clip_grad_norm_(a.policy.parameters(),.2);a.optimizer_step_and_maybe_schedule()
  self.assertTrue(all(g.compare_states(p['candidate'],g.capture(a)).values()))
 def test_nonaccumulating_displacements(self):
  from gradient_relation_experiment import kernel as g
  a=Algo();base=g.capture(a);d=torch.ones(sum(p.numel() for p in a.policy.parameters()))*.01
  engine.apply(a,base,d,1.);engine.apply(a,base,d,.25)
  got=engine.flat([p.detach()-base['model'][n] for n,p in a.policy.named_parameters()])
  self.assertTrue(torch.allclose(got,d*.25,atol=1e-7));engine.apply(a,base,d,0.)
  self.assertTrue(all(g.compare_states(base,g.capture(a)).values()))
 def test_neutral_denominators(self):
  r=metrics([1,0,1,-1],[1,1,0,-1],[.1]*4)
  self.assertEqual(r['harm_recall'],.5);self.assertEqual(r['harm_precision'],.5);self.assertEqual(r['sign_accuracy'],1.);self.assertEqual(r['decisive_fraction'],.5)
 def test_undefined_harm_not_pass(self):
  r=metrics([-1,-2],[-1,-2],[.1,.1]);self.assertTrue(np.isnan(r['harm_recall']));self.assertIsNone(clean(r)['harm_precision'])
 def test_linear_calibration(self):
  r=metrics([-2,-1,1,2],[-2,-1,1,2],[1e-6]*4)
  self.assertAlmostEqual(r['slope'],1);self.assertEqual(r['mae'],0);self.assertEqual(r['spearman'],1)
 def test_atomic_json_rejects_nan(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'result.json';write(p,{'x':1});self.assertEqual(read(p),{'x':1})
   with self.assertRaises(ValueError):write(p,{'x':float('nan')})
   self.assertEqual(read(p),{'x':1})
 def test_probe_isolation_and_manifest(self):
  starts=read(OUT/'START_MANIFEST.json')['starts'];probes=read(OUT/'PROBE_MANIFEST.json')['windows'];self.assertEqual(len(starts),120)
  for s in starts:
   rows=[r for r in probes if r['start_id']==s['start_id']];self.assertEqual(len(rows),24)
   self.assertFalse(set(r['record_id'] for r in rows)&set(s['record_indices']))
   for p in s['probe_isolation']:self.assertFalse(set(p['excluded_batch_trajectories'])&set(p['probe_trajectories']))
 def test_hvp_selection_before_outcome(self):
  from .run import hvp_ids
  ids=hvp_ids(read(OUT/'START_MANIFEST.json')['starts']);self.assertEqual(len(set(ids)),20)
if __name__=='__main__':unittest.main()
