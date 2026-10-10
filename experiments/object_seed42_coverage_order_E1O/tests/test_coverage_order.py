import random,sys,unittest
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from protective_e1.coverage_order import CoveragePlan,PlannedBatchSampler,CoverageOrder,OrderConfig,weighted_permutation

def plan(n=107):
    tasks=[6 if i%3 else 7 for i in range(n)]
    p=CoveragePlan(tasks,['id%d'%i for i in range(n)],10,7)
    p.begin_epoch(list(range(n)),0);return p

def order(p,config=None):
    o=CoverageOrder(None,42,7,{6:[i for i,t in zip(p.ids,p.tasks) if t==6]},[i for i,t in zip(p.ids,p.tasks) if t==7],lambda i:i,'test',config)
    def score(candidates,refs,new):
        return {t:[dict(id=i,score=k*.01,history_alignment=k*.01,current_alignment=1.,conflict_penalty=0.) for k,i in enumerate(ids)] for t,ids in candidates.items()},dict(reason='OK')
    o._score=score;return o

class CoverageTests(unittest.TestCase):
    def test_full_counter_not_unique_only(self):
        p=plan();bad=p.actual[:];bad[2]=bad[1]
        with self.assertRaises(AssertionError):p.verify(bad)
    def test_task_slot_rejected(self):
        p=plan();self.assertFalse(p.commit([1,3],[3,1],100)[0])
    def test_first_slot_pinned(self):
        p=plan();self.assertFalse(p.commit([10,11],[11,10],100)[0])
    def test_tail_pinned(self):
        p=plan();ps,end=p.candidates(0,random.Random(1));self.assertTrue(all(i<100 for i in ps[6]))
    def test_current_identity_pinned(self):
        p=plan();self.assertFalse(p.commit([3,6],[6,3],100)[0])
    def test_used_prefix_untouched(self):
        p=plan();next(iter(PlannedBatchSampler(p)));self.assertFalse(p.commit([1,11],[11,1],100)[0])
    def test_end_boundary(self):
        p=plan(10007);ps,end=p.candidates(0,random.Random(1));self.assertEqual(end,5000);self.assertTrue(all(i<end for i in ps[6]))
    def test_epoch_boundary(self):
        p=plan();_,end=p.candidates(0,random.Random(1));self.assertEqual(end,107)
    def test_weighted_without_replacement(self):
        ids=list(range(64));actual,permutation=weighted_permutation(ids,list(range(64)),random.Random(42));self.assertEqual(Counter(actual),Counter(ids));self.assertEqual(actual,[ids[i] for i in permutation])
    def test_weighted_rule_sequential_draws(self):
        class Draws:
            def choices(self,items,weights,k):self.seen.append((list(items),list(weights)));return [items[0]]
        r=Draws();r.seen=[];weighted_permutation([0,1,2],[0.,1.,2.],r)
        self.assertEqual([len(x[0]) for x in r.seen],[3,2,1]);self.assertAlmostEqual(r.seen[0][1][1]/r.seen[0][1][0],2.718281828459045)
    def test_invalid_candidate_all_task_fallback(self):
        p=plan();o=order(p);o._score=lambda c,h,n:({6:[dict(id=i,score=None) for i in c[6]]},dict(reason='OK'));o.schedule(p,0);self.assertEqual(p.actual,p.base)
    def test_invalid_history_all_fallback(self):
        p=plan();o=order(p);o._score=lambda c,h,n:({},dict(reason='INVALID_HISTORICAL_DIRECTION'));o.schedule(p,0);self.assertEqual(p.actual,p.base)
    def test_equal_scores_keep_order(self):
        p=plan();o=order(p);o._score=lambda c,h,n:({6:[dict(id=i,score=.2) for i in c[6]]},dict(reason='OK'));o.schedule(p,0);self.assertEqual(p.actual,p.base)
    def test_disabled_preserves_rng(self):
        p=plan();o=order(p,OrderConfig(enabled=False));before=o.rng.getstate();self.assertIsNone(o.schedule(p,0));self.assertEqual(o.rng.getstate(),before)
    def test_single_batch_skip(self):
        p=plan(10);o=order(p);o._score=lambda *args: self.fail('Must not score single-step interval');o.schedule(p,0);self.assertEqual(p.base,p.actual)
    def test_no_double_schedule(self):
        p=plan();o=order(p);o.schedule(p,0);state=o.state_dict();self.assertIsNone(o.schedule(p,0));self.assertEqual(state,o.state_dict())
    def test_scheduler_stage_not_epoch(self):
        o=order(plan());self.assertTrue(o.should_schedule(0));self.assertFalse(o.should_schedule(227));self.assertTrue(o.should_schedule(500))
    def test_task6_bypass(self):
        o=CoverageOrder(None,42,6,{},['a'],None,'x');self.assertFalse(o.should_schedule(0))
    def test_atomic_bad_permutation_rollback(self):
        p=plan();before=p.actual[:];self.assertFalse(p.commit([1,2],[1,1],100)[0]);self.assertEqual(p.actual,before)
    def test_committed_positions_not_reused(self):
        p=plan();self.assertTrue(p.commit([1,2],[2,1],100)[0]);self.assertFalse(p.commit([1,2],[1,2],100)[0])
    def test_global_rng_unchanged(self):
        p=plan();o=order(p);state=random.getstate();o.schedule(p,0);self.assertEqual(state,random.getstate())
    def test_actual_reader_and_epoch_counter(self):
        p=plan();o=order(p);iterator=iter(PlannedBatchSampler(p));o.schedule(p,0)
        for step,indices in enumerate(iterator):p.record_read([p.ids[i] for i in indices],[p.tasks[i] for i in indices],step)
        self.assertTrue(p.finish()['all_ids_once']);self.assertGreater(o.totals['cross_batch_moved'],0)
    def test_wrong_actual_ids_detected(self):
        p=plan();ids=next(iter(PlannedBatchSampler(p)))
        with self.assertRaises(AssertionError):p.record_read(['wrong']*10,[p.tasks[i] for i in ids],0)
    def test_future_mutation_read_lazily(self):
        p=plan();it=iter(PlannedBatchSampler(p));self.assertEqual(next(it),list(range(10)))
        self.assertTrue(p.commit([11,22],[22,11],100)[0]);self.assertEqual(next(it)[1],22)
    def test_reference_candidate_disjoint(self):
        p=plan();o=order(p);r=o.schedule(p,0);self.assertFalse(set(r['references'][6]) & set(r['tasks']['6']['candidates']))
        self.assertLessEqual(len(r['references'][6]),16);self.assertLessEqual(len(r['current_reference']),16)
    def test_resume_order_rng_next_epoch(self):
        p=plan();o=order(p);o.schedule(p,0);state=o.state_dict();p2=plan();a=o.schedule(p2,500)
        b_order=order(p2);b_order.load_state_dict(state);b=b_order.schedule(plan(),500)
        for event in (a,b):event.pop('scoring_seconds')
        self.assertEqual(a,b);self.assertEqual(o.rng.getstate(),b_order.rng.getstate())
    def test_nonfinite_weights_rejected(self):
        with self.assertRaises(AssertionError):weighted_permutation([1,2],[0.,float('nan')],random.Random())
    def test_protocol_config_fixed(self):
        with self.assertRaises(AssertionError):OrderConfig(refresh_interval_steps=100)

if __name__=='__main__':unittest.main()
