import tempfile
import unittest
from pathlib import Path
import torch
from . import original_probe as p
from .batches import labels, compare
from .io import sha, write


class ScreenTests(unittest.TestCase):
    def test_original_script_unchanged(self):
        self.assertEqual(sha(Path(p.__file__)), 'e03d3075d5925781fde81ebb1474f8806a18a746379dffe67b327e3b0a4be0c8')

    def test_modality_and_last_action(self):
        with tempfile.TemporaryDirectory() as d:
            h = torch.zeros(1,8,4,512)
            h[0,-1,1,2] = 2
            h[0,-1,2,3] = 3
            a = torch.arange(56,dtype=torch.float32).reshape(1,8,7)
            f=Path(d)/'toy.pt';torch.save(dict(feature=h,actions=a),f)
            v,act=p._feature_and_actions(f,1)
            self.assertTrue(torch.equal(act,a[0,-1:]))
            self.assertAlmostEqual(float(v.norm()),1,places=6)
            self.assertEqual(set(v.nonzero().flatten().tolist()),{2,515})

    def test_visual_is_two_cosines(self):
        a=torch.tensor([[1.,0.,0.,1.]])/2**.5
        b=torch.tensor([[1.,0.,0.,-1.]])/2**.5
        z=torch.zeros(1,1,7)
        v,*_=p._similarities(a,b,z,z,torch.ones(6))
        self.assertAlmostEqual(float(v),0,places=6)

    def test_labels_disjoint_and_gripper_conflict(self):
        v=torch.ones(1,5)
        d=torch.tensor([[.5,.6,1.5,.1,.1]])
        g=torch.tensor([[False,False,False,True,False]])
        match,share,bif=labels(v,d,g,.9)
        self.assertEqual(share.tolist(),[[True,False,False,False,True]])
        self.assertEqual(bif.tolist(),[[False,False,True,True,False]])
        self.assertFalse(bool((share&bif).any()))

    def test_idle_is_raw_rms_not_standardized(self):
        a=torch.full((1,1,7),.049);b=a.clone()
        _,_,_,idle=p._similarities(torch.ones(1,2),torch.ones(1,2),a,b,torch.full((6,),.1))
        self.assertTrue(bool(idle))
        b[:,:,:6]=.051
        self.assertFalse(bool(p._similarities(torch.ones(1,2),torch.ones(1,2),a,b,torch.ones(6))[3]))

    def test_iqr_floor(self):
        a=torch.zeros(5,1,7)
        self.assertTrue(torch.equal(p._action_stats(a,a),torch.full((6,),.1)))

    def test_paired_batches_and_reproducibility(self):
        g=torch.Generator().manual_seed(19)
        nv=torch.nn.functional.normalize(torch.randn(6,8,generator=g),dim=1)
        ov=torch.nn.functional.normalize(torch.randn(8,8,generator=g),dim=1)
        na=torch.randn(6,1,7,generator=g);oa=torch.randn(8,1,7,generator=g)
        args=(nv,ov,na,oa,torch.ones(6),torch.tensor([6]*4+[7]*4))
        a,s=compare(*args,seed=42,trials=3);b,t=compare(*args,seed=42,trials=3)
        self.assertEqual(a,b);self.assertEqual(s,t)
        for trial in range(3):
            rows=[x for x in a if x['trial']==trial]
            self.assertEqual(len(set(x['current_indices'] for x in rows)),1)
            self.assertTrue(all(x['evaluated_pairs']==25 for x in rows))
            self.assertTrue(all(x['unique_trajectory_pairs'] is None for x in rows))

    def test_json_rejects_nonfinite(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                write(Path(d)/'x.json',dict(x=float('nan')))


if __name__=='__main__':
    unittest.main()
