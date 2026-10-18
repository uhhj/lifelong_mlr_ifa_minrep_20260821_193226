"""Synthetic formula checks; none are experiment results."""
import importlib.util,unittest
from pathlib import Path
from fractions import Fraction as F
spec=importlib.util.spec_from_file_location('m',Path(__file__).parent/'source/zrg_exp/metrics.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Tests(unittest.TestCase):
    def test_parent_exact(self):self.assertEqual(m.calculate(m.EXPECTED),dict(AUC=F(109,160),FWT=F(13,20),NBT=F(-7,120),OldFinal=F(11,15),FinalAll=F(53,80)))
    def test_perfect(self):self.assertEqual(m.calculate({k:20 for k in m.EXPECTED}),dict(AUC=F(1),FWT=F(1),NBT=F(0),OldFinal=F(1),FinalAll=F(1)))
    def test_zero(self):self.assertTrue(all(v==0 for v in m.calculate({k:0 for k in m.EXPECTED}).values()))
    def test_forgetting_sign(self):self.assertEqual(m.calculate({k:20 if k[0]==k[1] else 0 for k in m.EXPECTED})['NBT'],1)
    def test_task_balancing(self):
        v={k:0 for k in m.EXPECTED};v[9,9]=20
        self.assertEqual(m.calculate(v)['AUC'],F(1,4))
if __name__=='__main__':unittest.main()
