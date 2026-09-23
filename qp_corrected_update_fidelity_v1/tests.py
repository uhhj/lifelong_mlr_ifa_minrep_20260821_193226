import tempfile,unittest
import numpy as np,pandas as pd
from pathlib import Path
from .common import *
from .analysis import metrics,bootstrap,fidelity_gate
from .worksets import fps,stride,generate
class Tests(unittest.TestCase):
 def test_exact_membership(self):
  refs=read(OUT/'REFERENCE_MANIFEST.json')['cells'];old=read(OUT/'OLD_PROBE_MANIFEST.json')['windows'];new=read(OUT/'NEW_PROBE_MANIFEST.json')['windows']
  for s in read(OUT/'START_MANIFEST.json')['starts']:
   a=[r for r in old if r['start_id']==s['start_id']];b=[r for r in new if r['start_id']==s['start_id']];rr=refs[f'{s["seed"]}/{s["task_id"]}']
   self.assertEqual(len(a),24);self.assertEqual(len(b),24);self.assertFalse({r['origin'] for r in a}&{r['origin'] for r in rr});self.assertFalse({r['record_id'] for r in a+b}&set(s['record_indices']))
 def test_neutral_denominators(self):
  v=metrics([1,0,-1],[1,1,-1],[.01]*3);self.assertEqual(v['harm_recall'],.5);self.assertEqual(v['harm_precision'],1);self.assertEqual(v['decisive_fraction'],2/3)
 def test_bootstrap_cluster(self):
  rows=[dict(seed=s,transition=t,damage_difference=-1) for s in [45,46,47] for t in ['6->7','7->8'] for _ in range(20)]
  self.assertEqual(bootstrap(pd.DataFrame(rows),100),[-1,-1])
  rows[0]['damage_difference']=np.nan;self.assertEqual(bootstrap(pd.DataFrame(rows),100),[None,None])
 def test_fidelity_missing_not_pass(self):
  frame=pd.DataFrame({k:[np.nan]*20 for k in ['spearman','sign_accuracy','harm_recall','harm_precision','decisive_fraction']});self.assertFalse(fidelity_gate(frame)['passed'])
 def test_constraint_generation_full_scan(self):
  with tempfile.TemporaryDirectory() as tmp:
   lam,rec,tr=generate(np.eye(3),np.array([-.1,.4,.2]),1.,[0],Path(tmp),{})
   self.assertEqual(rec['status'],'PASS');self.assertEqual(len(tr),3);self.assertEqual(tr[0]['omitted_violations'],[1,2]);np.testing.assert_allclose(lam,[0,.4,.2],atol=1e-10)
 def test_zero_update(self):
  with tempfile.TemporaryDirectory() as tmp:
   lam,rec,tr=generate(np.eye(3),-np.ones(3),1.,[0],Path(tmp),{})
   self.assertEqual(rec['status'],'PASS');np.testing.assert_array_equal(lam,np.zeros(3));self.assertEqual(len(tr),1)
 def test_fixed_fps(self):
  z=np.array([[1.,0.],[1.,0.],[0.,1.],[-1.,0.]])
  self.assertEqual(fps(z,2,[0,1,2,3]),[0,3]);self.assertEqual(fps(z,4,[0,1,2,3]),[0,1,2,3])
 def test_heterogeneous_failure_csv(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'a.csv';table(p,[{'status':'PASS','kkt':1e-12},{'status':'FAIL','error':'captured'}]);d=pd.read_csv(p);self.assertEqual(d.error.iloc[1],'captured');self.assertTrue(pd.isna(d.kkt.iloc[1]))
if __name__=='__main__':unittest.main()
