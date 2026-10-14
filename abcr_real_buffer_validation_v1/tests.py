import tempfile,unittest
from pathlib import Path
import numpy as np
from .common import write,read,save,PROTOCOL
from .analyze import phys,dist,select

class Tests(unittest.TestCase):
    def test_nonoverlap(self):
        x=np.zeros((2,8,7));x[1,4:]=2
        self.assertEqual(dist(x,np.array([0]),np.array([1]),0,4,np.ones(7))[0],0)
        self.assertEqual(dist(x,np.array([0]),np.array([1]),4,8,np.ones(7))[0],2)
    def test_rms(self):
        x=np.zeros((2,8,7));x[1]=3
        self.assertEqual(dist(x,np.array([0]),np.array([1]),0,1,np.ones(7))[0],3)
    def test_physical(self):
        d=dict(eef=np.zeros((2,3)),quat=np.array([[0,0,0,1],[0,0,0,-1.]]),state=np.zeros((2,9)),target_relative=np.zeros((2,3)),target_valid=np.ones(2,bool))
        self.assertTrue(phys(d,np.array([0]),np.array([1]))['physical_pass'][0])
        d['eef'][1,0]=.031;self.assertFalse(phys(d,np.array([0]),np.array([1]))['physical_pass'][0])
    def test_gripper_not_clipped(self):
        d=dict(eef=np.zeros((2,3)),quat=np.array([[0,0,0,1],[0,0,0,1.]]),state=np.zeros((2,9)),target_relative=np.zeros((2,3)),target_valid=np.ones(2,bool));d['state'][:,7]=.09
        self.assertFalse(phys(d,np.array([0]),np.array([1]))['physical_pass'][0])
    def test_cache_refuses(self):
        with tempfile.TemporaryDirectory() as p:
            f=Path(p)/'a.json';write(f,{'x':1},True)
            with self.assertRaises(AssertionError):write(f,{'x':2},True)
    def test_atomic_npz(self):
        with tempfile.TemporaryDirectory() as p:
            f=Path(p)/'a.npz';save(f,x=np.arange(3));self.assertTrue(np.array_equal(np.load(f)['x'],np.arange(3)))
    def test_protocol(self):
        self.assertEqual(PROTOCOL['stages'],[7,8,9]);self.assertFalse(PROTOCOL['formal_training_resumed'])

if __name__=='__main__':unittest.main()
