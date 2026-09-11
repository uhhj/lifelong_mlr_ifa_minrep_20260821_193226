import tempfile
import unittest
from pathlib import Path
from confirmation_experiment.commands import train_command
from confirmation_experiment.recovery.commands import train,rebuild_soft45

class RecoveryTests(unittest.TestCase):
    def test_launcher_only_changes_infrastructure(self):
        for seed in (45,46,47):
            for arm in ('mlr','soft'):
                old=train_command(seed,arm);new=train(seed,arm)
                self.assertNotIn('--standalone',new)
                self.assertIn('--rdzv_endpoint=localhost:0',new)
                # Hydra/scientific arguments remain byte-for-byte identical.
                self.assertEqual(old[old.index('seed=%d'%seed):],new[new.index('seed=%d'%seed):])
        self.assertNotEqual(next(x for x in train(46,'mlr') if x.startswith('--rdzv_id=')),
                            next(x for x in train(46,'soft') if x.startswith('--rdzv_id=')))
    def test_reconstruction_keeps_full_recipe(self):
        c=rebuild_soft45()
        for s in ('train.n_epochs=100','+start_task=6','+end_task=10','lifelong.kappa=0.5'):
            self.assertIn(s,c)
        self.assertFalse(any('max_optimizer_updates' in s for s in c))
    def segment(self,path):
        path.write_text(''.join('[task %d epoch 99 and %d succ.] 0.25\n'%(t,s) for s in range(6,10) for t in range(6,s+1)))
    def test_merge_preserves_first_cell_bytes(self):
        from confirmation_experiment.recovery.eval_entry import merge_segment
        with tempfile.TemporaryDirectory() as d:
            s=Path(d)/'segment';c=Path(d)/'canonical';self.segment(s)
            prefix=b'[task 6 epoch 99 and 6 succ.] 0.25\r\n';c.write_bytes(prefix)
            self.assertEqual(merge_segment(s,c),9)
            self.assertTrue(c.read_bytes().startswith(prefix))
            self.assertEqual(len(c.read_text().splitlines()),10)
    def test_merge_recovers_unlogged_completed_first_cell(self):
        from confirmation_experiment.recovery.eval_entry import merge_segment
        with tempfile.TemporaryDirectory() as d:
            s=Path(d)/'segment';c=Path(d)/'canonical';self.segment(s)
            self.assertEqual(merge_segment(s,c),10)
    def test_merge_refuses_disagreement_or_incomplete_segment(self):
        from confirmation_experiment.recovery.eval_entry import merge_segment
        with tempfile.TemporaryDirectory() as d:
            s=Path(d)/'segment';c=Path(d)/'canonical';self.segment(s)
            c.write_text('[task 6 epoch 99 and 6 succ.] 0.50\n');before=c.read_bytes()
            with self.assertRaises(AssertionError):merge_segment(s,c)
            self.assertEqual(c.read_bytes(),before)
            s.write_text('[task 6 epoch 99 and 6 succ.] 0.25\n')
            with self.assertRaises(ValueError):merge_segment(s,c)
            self.assertEqual(c.read_bytes(),before)

if __name__=='__main__':unittest.main()
