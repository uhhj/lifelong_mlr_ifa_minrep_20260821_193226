import copy,unittest
from .execution import recovery_records
from .queue import commands
class ExecutionTests(unittest.TestCase):
    def rows(self):
        rows=[]
        for b in range(309):
            rows.append(dict(batch=b,global_step=41800+b,optimizer_step=3400+b,scheduler_step=3400+b,
                fallback_triggered=b in (185,258),second_fallback_triggered=b==258,kkt=1e-9,
                projection_residual=1e-10))
        return rows
    def test_complete_epoch(self):
        self.assertEqual(recovery_records(self.rows())['after_B258_successful_batches'],50)
    def test_missing_epoch_end(self):
        with self.assertRaises(AssertionError):recovery_records(self.rows()[:-1])
    def test_wrong_fallback(self):
        rows=self.rows();rows[258]['second_fallback_triggered']=False
        with self.assertRaises(AssertionError):recovery_records(rows)
    def test_duplicate_optimizer_update(self):
        rows=self.rows();rows[185]['optimizer_step']+=1
        with self.assertRaises(AssertionError):recovery_records(rows)
    def test_strict_kkt(self):
        rows=self.rows();rows[258]['kkt']=1.01e-8
        with self.assertRaises(AssertionError):recovery_records(rows)
    def test_queue_scope(self):
        plan=commands();self.assertEqual([x[0] for x in plan],['train','evaluate','summarize'])
        self.assertIn('--resume',plan[0][1]);self.assertIn('+eval_epochs=[99]',plan[1][1])
        self.assertIn('eval.n_eval=20',plan[1][1])
if __name__=='__main__':unittest.main()
