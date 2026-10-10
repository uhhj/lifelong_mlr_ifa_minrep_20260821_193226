"""Synthetic software checks, not formal experiment results."""
import sys, copy, random, unittest, types
from pathlib import Path
from unittest.mock import patch
import torch
import test_replay_selection as fixture
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from protective_e1.identity import IdentityDataset, identity_rows
s=fixture.s

class TestUniform(unittest.TestCase):
    def setUp(self):
        self.f=fixture.TestSelection();self.f.setUp()
    def make(self,**kwargs):
        return self.f.make(reference_per_old_task=2,candidates_per_old_task=4,pool_weighting='uniform',**kwargs)
    def test_config_validity_and_default(self):
        self.assertEqual(s.SelectionConfig().pool_weighting,'score')
        with self.assertRaises(AssertionError):s.SelectionConfig(pool_weighting='invalid')
    def test_exact_uniform_sidepass(self):
        sel=self.make();r=sel.refresh(0)
        for task,ps in r['probabilities'].items():
            self.assertEqual(ps,[1./len(ps)]*len(ps));self.assertAlmostEqual(sum(ps),1.)
            self.assertEqual(sel.distributions[int(task)]['probabilities'],ps)
            self.assertAlmostEqual(sum(r['score_probabilities'][task]),1.)
    def test_finite_scores_cannot_control_uniform_draw(self):
        a=self.make();b=self.make()
        with patch.object(s,'protective_score',side_effect=[-.9,-.3,.4,.9]*2):ra=a.refresh(0)
        with patch.object(s,'protective_score',side_effect=[.9,.4,-.3,-.9]*2):rb=b.refresh(0)
        self.assertNotEqual(ra['score_probabilities'],rb['score_probabilities'])
        self.assertEqual(ra['candidates'],rb['candidates'])
        for _ in range(100):self.assertEqual(a.replace_old_slots(self.f.batch)[1],b.replace_old_slots(self.f.batch)[1])
        self.assertEqual(a.rng.getstate(),b.rng.getstate())
    def test_paired_random_consumption_and_future_pools(self):
        a=self.make();b=self.f.make(reference_per_old_task=2,candidates_per_old_task=4,pool_weighting='score')
        a.refresh(0);b.refresh(0)
        for _ in range(100):
            ea=a.replace_old_slots(self.f.batch)[1];eb=b.replace_old_slots(self.f.batch)[1]
            self.assertEqual([(e['slot'],e['task']) for e in ea],[(e['slot'],e['task']) for e in eb])
            self.assertEqual(a.rng.getstate(),b.rng.getstate())
        ra=a.refresh(500);rb=b.refresh(500)
        for k in ('references','candidates','current_reference','step'):self.assertEqual(ra[k],rb[k])
    def test_uniform_restore(self):
        a=self.make();a.refresh(0);b=self.make();b.load_state_dict(a.state_dict())
        self.assertEqual(a.replace_old_slots(self.f.batch)[1],b.replace_old_slots(self.f.batch)[1])
        wrong=self.f.make()
        with self.assertRaises(AssertionError):wrong.load_state_dict(a.state_dict())
    def test_actual_indices_single_sampler_iteration(self):
        class Base(torch.utils.data.Dataset):
            file_paths=['/features/8/x.pt','/buffer/t6__y.pt','/features/8/z.pt']
            def __getitem__(self,i):return {'feature':torch.tensor([i]),'actions':torch.tensor([i])},('6' if i==1 else '8')
            def __len__(self):return 3
        class Once(torch.utils.data.Sampler):
            def __init__(self):self.calls=0
            def __iter__(self):
                self.calls+=1;assert self.calls==1
                yield from [2,1,0]
            def __len__(self):return 3
        sampler=Once();loader=torch.utils.data.DataLoader(IdentityDataset(Base()),batch_size=3,sampler=sampler,num_workers=0)
        payload,tasks,ids=next(iter(loader))
        self.assertEqual(sampler.calls,1);self.assertEqual(ids,('current/8/z.pt','buffer/t6__y.pt','current/8/x.pt'))
        self.assertEqual(payload['feature'].flatten().tolist(),[2,1,0])
        rows=identity_rows(ids,tasks,[dict(slot=1,task=6,candidate='buffer/t6__y.pt',applied=True)])
        self.assertTrue(rows[1]['attempted']);self.assertFalse(rows[1]['id_changed'])
        self.assertFalse(rows[0]['attempted'])
    def test_identity_hook_payload_isolation(self):
        e=fixture.HookedFixtureUpdate(self.f.policy)
        e.configure_e1_stage(8,self.f.old,self.f.new,self.f.load,'fixture',s.SelectionConfig(pool_weighting='uniform'))
        # Stub only the native optimizer update: this test adds no optimizer call.
        def native(obj,data,*args,**kwargs):
            self.assertEqual(len(data),2);self.assertEqual(set(data[0]),{'feature','actions'})
            obj._optimizer_step_count+=1;return 0.
        with patch.object(fixture.NativeFixtureUpdate,'observe',native):
            e.observe([*self.f.batch,self.f.ids],8,0)
        rows=e.pop_e1_events()[0]['actual_samples']
        self.assertEqual(len(rows),len(self.f.ids))
        for row in rows:
            if row['task']==8:self.assertEqual(row['original_id'],row['final_id']);self.assertFalse(row['attempted'])
    def test_paths_and_actual_configuration(self):
        source=Path(__file__).resolve().parents[1]/'source'
        env=(source/'protective_e1/env.sh').read_text()
        self.assertIn('E1O_coverage_order_20261009/source',env)
        self.assertIn('E1O_coverage_order_seed42_20261009',env)
        prep=(source/'protective_e1/prepare.py').read_text()
        self.assertEqual(prep.count("asdict(OrderConfig())"),2)
        self.assertIn('hydra.run.dir="$E1_OUT/',(source/'protective_e1/evaluate.sh').read_text())

if __name__=='__main__':unittest.main(verbosity=2)
