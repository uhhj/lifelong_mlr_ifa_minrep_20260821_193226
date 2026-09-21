import random
import unittest
import torch
from sample_qp_optimized_v1.tests import original_functions, ToyGMM
from sample_qp_optimized_v1.core import Workspace, inner, project
from .common import selected_references,publish_eval_log

class Tests(unittest.TestCase):
    def test_completed_eval_reuse_does_not_append_duplicate_rows(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'attempt.log';dest=Path(d)/'canonical.log'
            body=''.join('[task %d epoch 99 and %d succ.] 0.50\n'%(t,s)
                         for s in range(6,10) for t in range(6,s+1))
            source.write_text(body);publish_eval_log(source,dest);publish_eval_log(source,dest)
            self.assertEqual(dest.read_text(),body)
            source.write_text(body.replace('0.50','0.55',1))
            with self.assertRaises(AssertionError):publish_eval_log(source,dest)
            self.assertEqual(dest.read_text(),body)
            source.write_text(body.splitlines()[0]+'\n')
            with self.assertRaises(ValueError):publish_eval_log(source,dest)

    def test_passive_trial_recorder_ignores_warmup_and_preserves_rng(self):
        import numpy as np
        from .evaluate import Observer
        from cb_ifa_experiment.gpu_runtime import digest,rng_state
        class Env:
            def set_init_state(self,x):return x
            def step(self,a):return {},None,[True]+[False]*9,{}
            def close(self):pass
        o=Observer(Env());before=digest(rng_state())
        o.set_init_state(np.zeros((10,2)))
        for _ in range(5):o.step(np.zeros((10,7)))
        self.assertFalse(any(o.active['successes']))
        o.step(np.zeros((10,7)));o.close()
        self.assertEqual(o.rounds[0]['first_success_step'][0],1)
        self.assertEqual(digest(rng_state()),before)

    def test_membership_is_private_reproducible_and_old_only(self):
        plan = dict(task=9,records=[dict(task=t,origin='%d/%03d.pt'%(t,i),sha256='x')
                                   for t in range(6,10) for i in range(50)])
        state = random.getstate(); a = selected_references(plan)
        self.assertEqual(state,random.getstate()); self.assertEqual(a,selected_references(plan))
        self.assertEqual(set(a),{'6','7','8'})
        self.assertTrue(all(len(v)==len({x['origin'] for x in v})==32 for v in a.values()))
        self.assertEqual(selected_references(dict(task=6,records=plan['records'])),{})

    def test_full_counts_exact_original_math(self):
        torch.set_num_threads(2); torch.manual_seed(4696); original=original_functions()
        for n in (32,64,96):
            H=torch.randn(65539,n); u=torch.randn(65539)
            K,f=inner(H,u,True); oldK,oldf=original['inner'](H,u,True)
            self.assertTrue((K==oldK).all()); self.assertTrue((f==oldf).all())
            c,s=project(H,K,u); oldc,olds=original['sample_project'](H,K,u,independent=False)
            self.assertTrue(torch.equal(c,oldc)); self.assertEqual(s['lam'],olds['lam'])

    def test_reference_matrix_96_columns_refreshes(self):
        torch.manual_seed(46); model=ToyGMM(); w=Workspace(model.parameters())
        batches=[(torch.randn(8,3),torch.randn(8,2)) for _ in range(12)]
        a=w.sample_gradients(batches,model.loss,96).clone()
        self.assertEqual(a.shape[1],96)
        with torch.no_grad():model.layers[0].weight.add_(.01)
        b=w.sample_gradients(batches,model.loss,96)
        self.assertFalse(torch.equal(a,b))

if __name__ == '__main__':unittest.main()
