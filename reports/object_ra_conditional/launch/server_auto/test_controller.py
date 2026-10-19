"""Synthetic/controller tests, zero training and environment calls."""
import unittest,tempfile,json,ast
from pathlib import Path
from fractions import Fraction as F
from flow_metrics import gate,metrics
from prepare_zr import parent_spec
class Tests(unittest.TestCase):
 def test_gate_strict_mean_and_two(self):
  self.assertTrue(gate([F(1,10),F(1,20),F(-1,100)],True)['passed'])
  self.assertFalse(gate([F(1,10),F(-1,100),F(-1,100)],True)['passed'])
  self.assertFalse(gate([F(1,10),F(1,10),F(-1)],True)['passed'])
  self.assertFalse(gate([F(1),F(1),F(-2)],True)['passed'])
 def test_engineering_not_metric_failure(self):
  with self.assertRaises(RuntimeError):gate([F(-1)]*3,False)
 def test_seed_count(self):
  with self.assertRaises(AssertionError):gate([F(1)]*2,True)
 def test_parent_registry(self):
  self.assertIn('RA_Coreset_OBJECT_seed43',parent_spec('OBJECT_RA',43)['parent'])
  self.assertIn('Coreset_paper_GOAL_seed41_20261019',parent_spec('GOAL_Coreset',41)['parent'])
  self.assertIn('RA_Coreset_GOAL_seed42_20261018',parent_spec('GOAL_RA',42)['parent'])
  with self.assertRaises(AssertionError):parent_spec('OBJECT_RA',44)
 def test_fraction_records(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)
   for s in range(6,10):
    for t in range(6,s+1):
     rows=[dict(seed=41,arm='ra_coreset',stage=s,task=t,trial=i,initial_state_index=i,success=i<10) for i in range(20)]
     (p/('trials_stage%d_task%d.jsonl'%(s,t))).write_text(''.join(json.dumps(r)+'\n' for r in rows))
   m,*_=metrics(p,41,'ra_coreset');self.assertEqual(m['AUC'],F(1,2));self.assertEqual(m['NBT'],0)
   f=p/'trials_stage6_task6.jsonl';f.write_text(f.read_text()+f.read_text().splitlines()[0]+'\n')
   with self.assertRaises(AssertionError):metrics(p,41,'ra_coreset')
 def test_suite_budget(self):
  for counts,steps,ex in [([1562,1480,1560,1319],10550,105240),([997,889,904,1969],7610,75980)]:
   q=[(n+1)//2 for n in counts];lens=[n+sum(q[:i]) for i,n in enumerate(counts)]
   self.assertEqual(10*sum((n+9)//10 for n in lens),steps);self.assertEqual(10*sum(lens),ex)
 def test_template_unmodified_core(self):
  p=Path(__file__).parent/'zr_template/source/zr2_exp/model.py'
  for name in ['model.py','diagnostics.py']:
   h=__import__('hashlib').sha256(p.with_name(name).read_bytes()).hexdigest()
   original=json.loads((Path(__file__).parent/'ZR_ADAPTATION_HASHES.json').read_text())[name]['original_sha256'];self.assertEqual(h,original)
if __name__=='__main__':unittest.main(verbosity=2)
