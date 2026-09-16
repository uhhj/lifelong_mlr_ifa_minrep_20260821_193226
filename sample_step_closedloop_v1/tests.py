import unittest
import numpy as np
from .common import *
from .analyze import summarize,decision
from sample_task_qp_v1.solver import solve

class Tests(unittest.TestCase):
 def test_sample_constraints_not_task_average(self):
  H=np.array([[1.,-1.],[0.,0.]]);u=np.array([2.,3.]);s=solve(H.T@H,H.T@u,np.linalg.norm(u));v=u-H@np.array(s['lam'])
  np.testing.assert_allclose(v,[0,3],atol=1e-8);avg=H.mean(1,keepdims=True);q=solve(avg.T@avg,avg.T@u,np.linalg.norm(u));np.testing.assert_array_equal(u-avg@np.array(q['lam']),u)
 def test_trial_success_ignores_warmup(self):
  d=np.zeros((8,10),bool);d[0,0]=True;d[7,1]=True;r=trial_summary(d,[-4,-3,-2,-1,0,1,2,3]);self.assertFalse(r[0]['success']);self.assertEqual(r[1]['first_success_step'],3)
 def test_observation_hash_is_content(self):
  a=np.empty(2,dtype=object);a[0]={'x':np.arange(4)};a[1]={'x':np.arange(2)}
  import copy
  self.assertEqual(observation_digest(a),observation_digest(copy.deepcopy(a)));a[0]['x'][0]=9;self.assertNotEqual(observation_digest(a),observation_digest([{'x':np.arange(4)},{'x':np.arange(2)}]))
 def test_weighting_and_decision(self):
  trials=[dict(seed=s,task=t,arm=a,trial=i,success=int(i<(30 if a=='MLR' else 32 if a=='TASK_STEP_QP' else 35))) for s in [45,46,47] for t in [6,7,8] for a in ARMS for i in range(50)]
  c,s,p,pa,d=summarize(trials);self.assertEqual(next(r for r in p if r['arm']=='SAMPLE_STEP_QP')['seen'],70)
  self.assertEqual(decision(d,c)['status'],'CANDIDATE_FOR_LARGER_STUDY')
  for r in trials:
   if r['arm']=='SAMPLE_STEP_QP':r['success']=int(r['trial']<20)
  c,s,p,pa,d=summarize(trials);self.assertEqual(decision(d,c)['status'],'NOT_SUPPORTED')
 def test_dtype_restoration(self):
  import hashlib
  actions=[np.zeros((10,7),np.float64),np.ones((10,7),np.float32)/3];stack=np.stack(actions)
  self.assertEqual(b''.join(a.tobytes() for a in actions),b''.join(np.asarray(a,dtype=b.dtype).tobytes() for a,b in zip(stack,actions)))
if __name__=='__main__':
 r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));write(OUT/'CPU_TESTS.json',dict(status='PASS' if r.wasSuccessful() else 'FAIL',tests=r.testsRun));raise SystemExit(0 if r.wasSuccessful() else 1)
