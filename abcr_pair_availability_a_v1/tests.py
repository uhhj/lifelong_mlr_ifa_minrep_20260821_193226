import tempfile
import unittest
import json
import os
from unittest.mock import patch
from pathlib import Path
import torch
from . import original_probe as p
from .batches import labels, compare
from .io import sha, write
from .settings import FROZEN_PARAMETERS, assert_consistent, native_args
from .cache import CacheMismatch, freeze_many, make_binding, verify_complete


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
        parameters=dict(FROZEN_PARAMETERS,batch_trials=3)
        a,s=compare(*args,seed=42,parameters=parameters);b,t=compare(*args,seed=42,parameters=parameters)
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

    def test_custom_parameters_flow_to_batches(self):
        v=torch.ones(6,2);a=torch.zeros(6,1,7)
        params=dict(FROZEN_PARAMETERS,topk=2,batch_new=3,batch_old=4,batch_trials=2,visual_thresholds=[.8])
        rows,summaries=compare(v,v,a,a,torch.ones(6),torch.tensor([6]*6),seed=0,parameters=params)
        self.assertEqual(len(rows),4)
        self.assertEqual({r['evaluated_pairs'] for r in rows},{6,12})
        self.assertEqual({r['strategy'] for r in rows},{'RANDOM_3_PLUS_4','RETRIEVE_TOP2_PER_ANCHOR'})
        with self.assertRaises(AssertionError):
            compare(v,v,a,a,torch.ones(6),torch.tensor([6]*6),seed=0,trials=9,parameters=params)

    def test_native_parameter_drift_rejected(self):
        args=native_args(dict(seed=42,task=7,feature_dir='/a',buffer_dir='/b'),'/c',FROZEN_PARAMETERS)
        args.action_share=.6
        with self.assertRaises(AssertionError):assert_consistent(args,FROZEN_PARAMETERS)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        new=self.root/'features/7';old=self.root/'old';new.mkdir(parents=True);old.mkdir()
        (new/'0.pt').write_bytes(b'aaaa');(old/'t6__0.pt').write_bytes(b'bbbb')
        self.stage=dict(seed=42,task=7,feature_dir=str(new.parent),buffer_dir=str(old),n_current=1,n_old=1,
            old_files=['t6__0.pt'],buffer_evidence=dict(kind='TEST'))
        self.code={'source':'hash1'};self.runtime={'torch':'test'};self.parameters=dict(FROZEN_PARAMETERS)
        self.binding=self.current();self.final=self.root/'done';self.final.mkdir()
        write(self.final/'STAGE_BINDING.json',self.binding);(self.final/'result.csv').write_text('x\n1\n')
        self.record=dict(binding=self.binding['fingerprints'],outputs_sha256={p.name:sha(p) for p in self.final.iterdir()})
        write(self.final/'COMPLETE.json',self.record)

    def tearDown(self):self.tmp.cleanup()

    def current(self):return make_binding(self.stage,self.parameters,self.code,self.runtime)

    def test_unchanged_binding_reused(self):self.assertTrue(verify_complete(self.final,self.current()))

    def test_same_size_and_mtime_input_change_rejected(self):
        p=Path(self.stage['feature_dir'])/'7/0.pt';st=p.stat();before=(self.final/'COMPLETE.json').read_bytes()
        p.write_bytes(b'cccc');os.utime(p,ns=(st.st_atime_ns,st.st_mtime_ns))
        with self.assertRaises(CacheMismatch):verify_complete(self.final,self.current())
        self.assertEqual(before,(self.final/'COMPLETE.json').read_bytes())

    def test_old_input_change_rejected(self):
        (Path(self.stage['buffer_dir'])/'t6__0.pt').write_bytes(b'cccc')
        with self.assertRaises(CacheMismatch):verify_complete(self.final,self.current())

    def test_code_change_rejected(self):
        self.code['source']='hash2'
        with self.assertRaises(CacheMismatch):verify_complete(self.final,self.current())

    def test_parameter_change_rejected(self):
        self.parameters['action_share']=.6
        with self.assertRaises(CacheMismatch):verify_complete(self.final,self.current())

    def test_runtime_change_rejected(self):
        self.runtime['torch']='other'
        with self.assertRaises(CacheMismatch):verify_complete(self.final,self.current())

    def test_output_change_rejected(self):
        (self.final/'result.csv').write_text('x\n2\n')
        with self.assertRaises(CacheMismatch):verify_complete(self.final,self.current())

    def test_membership_change_rejected(self):
        (Path(self.stage['buffer_dir'])/'t6__1.pt').write_bytes(b'xxxx')
        with self.assertRaises(CacheMismatch):self.current()

    def test_legacy_unbound_not_silently_upgraded(self):
        self.record.pop('binding');write(self.final/'COMPLETE.json',self.record)
        with self.assertRaisesRegex(CacheMismatch,'LEGACY_CACHE_UNBOUND'):verify_complete(self.final,self.current())
        self.assertNotIn('binding',json.loads((self.final/'COMPLETE.json').read_text()))

    def test_immutable_manifests_not_overwritten_on_error(self):
        p=self.root/'input.json';q=self.root/'not_created.json';write(p,{'sha':'old'})
        before=p.read_bytes()
        with self.assertRaises(CacheMismatch):freeze_many({q:{'stage':1},p:{'sha':'new'}})
        self.assertEqual(before,p.read_bytes());self.assertFalse(q.exists())

    def test_real_run_stage_checks_binding_before_skipping(self):
        from .run import run_stage
        # Only the early cache path executes: no load of these byte-only fixtures.
        target=self.root/'object_seed42_task7';self.final.rename(target);self.final=target
        with patch('abcr_pair_availability_a_v1.run.probe.run',side_effect=AssertionError('should not execute')):
            run_stage(self.stage,self.root,self.code,self.runtime,self.parameters)
            (Path(self.stage['buffer_dir'])/'t6__0.pt').write_bytes(b'cccc')
            with self.assertRaises(CacheMismatch):run_stage(self.stage,self.root,self.code,self.runtime,self.parameters)


if __name__=='__main__':
    unittest.main()
