"""Small contract tests, no checkpoint / CUDA / network."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from abcr_pair_validation_b_v1 import original_experiment_b as b
from .evaluate import metrics, trajectory_bootstrap


class Contracts(unittest.TestCase):
    def test_metrics_separate_unaccepted_positive_and_false_match(self):
        rows = [dict(label=1,predicted_match=True),dict(label=1,predicted_match=False),
                dict(label=0,predicted_match=True),dict(label=0,predicted_match=False)]
        m = metrics(rows)
        self.assertEqual((m['tp'],m['fp'],m['tn'],m['fn']), (1,1,1,1))
        self.assertEqual(m['precision'],.5)
        self.assertEqual(m['recall'],.5)

    def test_no_accepted_pairs_has_undefined_precision(self):
        m = metrics([dict(label=1,predicted_match=False),dict(label=0,predicted_match=False)])
        self.assertIsNone(m['precision'])
        self.assertEqual(m['recall'],0.)

    def test_bootstrap_reproducible_and_preserves_shared_endpoint(self):
        rows = [dict(label=i%2,predicted_match=bool(i%3), task_a=0,task_b=1,
                     traj_a='t0/a'+str(i%3),traj_b='t1/b'+str(i%4)) for i in range(12)]
        self.assertEqual(trajectory_bootstrap(rows,30),trajectory_bootstrap(rows,30))

    def test_native_script_hash_unchanged(self):
        self.assertEqual(b.sha256(b.__file__),'e5905b95b34f8d27041a8181313f9fc3cd6cf7ca6df814d2433437220869b853')

    def test_frozen_threshold_higher_quantile_and_immutable_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scores = np.linspace(.1,.99,100)
            n = 200
            features = np.zeros((n,2))
            features[::2,0] = 1
            features[1::2,0] = scores
            features[1::2,1] = np.sqrt(1-scores**2)
            np.savez(root/'records.npz',ids=np.array([str(i) for i in range(n)]),task_ids=np.arange(n)%2,
                     traj_ids=np.array(['traj'+str(i) for i in range(n)]),alignment_verified=np.ones(n,bool),
                     eef_pos_m=np.zeros((n,3)),eef_quat_xyzw=np.tile([0,0,0,1],(n,1)),
                     gripper_open_fraction=np.zeros(n),agent_unmod=features,eih_unmod=features,
                     future_actions=np.zeros((n,3,7)),future_valid=np.ones((n,3),bool))
            b.save_csv(root/'pairs.csv',[dict(id_a=2*i,id_b=2*i+1,label=int(i>=70)) for i in range(100)],['id_a','id_b','label'])
            args=SimpleNamespace(records=str(root/'records.npz'),pairs=str(root/'pairs.csv'),max_base_task=5,
                pos_m=.03,rot_deg=15.,grip_frac=.10,velocity_mps=.05,angular_velocity_radps=.35,q=3,
                min_pos=20,min_neg=40,negative_quantile=.95,encoder_sha256='a'*64,motion_share=.5,motion_bif=1.5,out=str(root/'frozen.json'))
            with contextlib.redirect_stdout(io.StringIO()):
                b.cmd_calibrate(args)
            self.assertAlmostEqual(json.loads((root/'frozen.json').read_text())['visual']['tau_v'],np.quantile(scores[:70],.95,method='higher')+1e-7)
            with self.assertRaises(FileExistsError):
                b.cmd_calibrate(args)

    def test_unverified_alignment_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'pairs.csv'
            b.save_csv(p,[dict(id_a='a',id_b='b')],['id_a','id_b'])
            with self.assertRaisesRegex(ValueError,'NOT independently verified'):
                b.get_pairs(p,dict(index={'a':0,'b':1},alignment_verified=[True,False]))


if __name__ == '__main__':
    unittest.main()
