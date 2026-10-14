"""Contract tests for the unchanged supplied B script and pack sampling."""
import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from . import original_experiment_b as b
from .prepare import physical_distances,pick_pairs,orientation_crosscheck


def records(n=4):
    return dict(ids=['r%d'%i for i in range(n)],index={'r%d'%i:i for i in range(n)},
        task_ids=np.array([0,1,2,3]),traj_ids=np.array(['t%d'%i for i in range(n)]),
        alignment_verified=np.ones(n,dtype=bool),eef_pos_m=np.zeros((n,3)),
        eef_quat_xyzw=np.tile([0.,0.,0.,1.],(n,1)),gripper_open_fraction=np.zeros(n),
        agent_unmod=np.tile([1.,0.],(n,1)),eih_unmod=np.tile([1.,0.],(n,1)),
        future_actions=np.zeros((n,3,7)),future_valid=np.ones((n,3),dtype=bool))


class Contracts(unittest.TestCase):
    def test_rotation_vector_branch_cut_is_not_physical_mismatch(self):
        theta=3.3
        q=np.array([[np.sin(theta/2),0,0,np.cos(theta/2)]])
        self.assertLess(orientation_crosscheck(q,np.array([[theta-2*np.pi,0,0]])),1e-14)

    def test_quaternion_double_cover(self):
        d=records();d['eef_quat_xyzw'][1]*=-1
        q=b.pair_descriptors(d,[(0,1,None)],.03,15,.1)
        self.assertEqual(float(q['ds'][0]),0)

    def test_max_not_average_gate(self):
        d=records();d['eef_pos_m'][1,0]=.031
        self.assertGreater(float(b.pair_descriptors(d,[(0,1,None)],.03,15,.1)['ds'][0]),1)

    def test_visual_uses_worst_view(self):
        d=records();d['eih_unmod'][1]=[0,1]
        self.assertEqual(float(b.pair_descriptors(d,[(0,1,None)],.03,15,.1)['sv'][0]),0)

    def test_future_not_history_or_padding(self):
        d=records();d['future_valid'][1,-1]=False
        ok,_,_=b.motion_pair(d,[(0,1,None)],3,np.ones(6));self.assertFalse(ok[0])

    def test_gripper_conflict_any_future_step(self):
        d=records();d['future_actions'][1,2,6]=1
        _,_,conflict=b.motion_pair(d,[(0,1,None)],3,np.ones(6));self.assertTrue(conflict[0])

    def test_unverified_alignment_rejected(self):
        d=records();d['alignment_verified'][1]=False
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'pairs.csv';p.write_text('id_a,id_b,label\nr0,r1,1\n')
            with self.assertRaisesRegex(ValueError,'alignment'):b.get_pairs(p,d,True)

    def test_blank_human_labels_rejected(self):
        d=records()
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'pairs.csv';p.write_text('id_a,id_b,label\nr0,r1,\n')
            with self.assertRaisesRegex(ValueError,'label'):b.get_pairs(p,d,True)

    def test_conflicting_duplicate_labels_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'pairs.csv';p.write_text('id_a,id_b,label\nr0,r1,1\nr1,r0,0\n')
            with self.assertRaisesRegex(ValueError,'contradictory'):b.get_pairs(p,records(),True)

    def test_insufficient_human_labels_never_make_threshold(self):
        d=records()
        with tempfile.TemporaryDirectory() as td:
            td=Path(td);np.savez(td/'r.npz',**{k:v for k,v in d.items() if k!='index'})
            (td/'p.csv').write_text('id_a,id_b,label\nr0,r1,1\nr2,r3,0\n')
            args=SimpleNamespace(records=td/'r.npz',pairs=td/'p.csv',max_base_task=5,q=3,
                pos_m=.03,rot_deg=15,grip_frac=.1,velocity_mps=.05,angular_velocity_radps=.35,
                min_pos=20,min_neg=40,out=td/'frozen.json')
            with self.assertRaisesRegex(ValueError,'Not enough'):b.cmd_calibrate(args)
            self.assertFalse(args.out.exists())

    def test_pack_metric_matches_supplied_metric(self):
        d=records();d['eef_pos_m'][1]=[.01,.01,0];d['gripper_open_fraction'][1]=.03
        a=dict(rows=[dict(traj_id='a')],p=d['eef_pos_m'][:1],q=d['eef_quat_xyzw'][:1],g=d['gripper_open_fraction'][:1])
        c=dict(rows=[dict(traj_id='b')],p=d['eef_pos_m'][1:2],q=d['eef_quat_xyzw'][1:2],g=d['gripper_open_fraction'][1:2])
        ds,*_=physical_distances(a,c)
        np.testing.assert_array_equal(ds.ravel(),b.pair_descriptors(d,[(0,1,None)],.03,15,.1)['ds'])
        self.assertEqual(pick_pairs(a,c,20,42),pick_pairs(a,c,20,42))


if __name__=='__main__':unittest.main()
