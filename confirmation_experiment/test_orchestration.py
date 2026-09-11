"""Checks on frozen commands, native evaluation wrapping and audit copies."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from confirmation_experiment.commands import train_command,eval_command,feature_command
from confirmation_experiment.common import ROOT,sha

class OrchestrationTests(unittest.TestCase):
    def test_replay_origin_names(self):
        from confirmation_experiment.instrumentation import origin_for_record
        for source in ('/buffers/t6__1.pt','/buffers/t6__1__abcdef.pt','/buffers/t6__1__abcdef__2.pt','/features/6/1.pt'):
            self.assertEqual(origin_for_record(source,'/features'),Path('/features/6/1.pt'))

    def test_six_fixed_commands(self):
        destinations=set()
        for seed in (45,46,47):
            for arm in ('mlr','soft'):
                c=train_command(seed,arm)
                for required in ('train.n_epochs=100','train.batch_size=10','+past_data_rate=0.5',
                                 '+repro_scheduler_mode=batch_step','+start_task=6','+end_task=10',
                                 'train.optimizer.kwargs.lr=1e-4','+save_final_checkpoint=true'):
                    self.assertIn(required,c)
                self.assertNotIn('+max_optimizer_updates=8',c)
                self.assertEqual('lifelong.kappa=0.5' in c,arm=='soft')
                destinations.add(next(x for x in c if x.startswith('+experiment_root=')))
        self.assertEqual(len(destinations),6)

    def test_native_evaluation_recipe(self):
        for seed in (45,46,47):
            for arm in ('mlr','soft'):
                c=eval_command(seed,arm)
                for x in ('eval.n_eval=20','eval.num_procs=20','+eval_epochs=[99]',
                          '+start_task=6','+end_task=10','training=false'):
                    self.assertIn(x,c)
                self.assertFalse(any('best' in x or 'eval_task_ids=' in x or 'eval_stage=' in x for x in c))
            self.assertIn('train.batch_size=1',feature_command(seed))

    def test_frozen_runtime_matches_actual(self):
        folder=ROOT/'confirmation_experiment/frozen_runtime'
        manifest=json.loads((folder/'MANIFEST.json').read_text())
        self.assertGreater(len(manifest),20)
        for name,h in manifest.items():
            self.assertEqual(sha(folder/name),h,name+' snapshot')
            self.assertEqual(sha(ROOT/name),h,name+' active runtime')

    def test_eval_barrier_keeps_each_native_call_once(self):
        # A deterministic stub exercises wrapper bookkeeping only; this is not
        # an extra robot trial, nor a replacement for the formal task6 gate.
        import libero.lifelong.metric as metric
        import confirmation_experiment.eval_entry as entry
        from easydict import EasyDict
        calls=[];waits=[]
        def original(**kw):
            calls.append(kw['task_ids'][0]);return [.5]
        def native_main(*args,**kw):
            for stage in range(6,10):
                for task in range(6,stage+1):
                    self.assertEqual(metric.evaluate_success(cfg=EasyDict(eval=EasyDict(n_eval=20)),task_ids=[task]),[.5])
        try:
            with tempfile.TemporaryDirectory() as d, patch.dict('os.environ',CONFIRMATION_RUN_DIR=d), \
                 patch.object(metric,'evaluate_success',original), patch('runpy.run_path',native_main), \
                 patch('confirmation_experiment.common.wait_for',lambda *args:waits.append(args)), \
                 patch('confirmation_experiment.common.rng_state',lambda:{'fixed':1}):
                entry.main()
                self.assertEqual(calls,[6,6,7,6,7,8,6,7,8,9])
                self.assertEqual(len(waits),1)
                self.assertEqual(json.loads((Path(d)/'EVALUATION_COMPLETE.json').read_text())['native_evaluation_calls'],10)
        finally:
            pass

if __name__=='__main__':unittest.main()
