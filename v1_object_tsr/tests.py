import random
import unittest
import numpy as np
from .routing import RoutedSampler,probabilities


class Tests(unittest.TestCase):
    def test_probability_floor_signed_invalid(self):
        p0={6:.2,7:.3,8:.5};p=probabilities(p0,{6:-.8,7:-.2,8:None})
        self.assertEqual(p[8],.5);self.assertGreater(p[7],p[6])
        self.assertTrue(all(p[t]>=.5*p0[t] for t in p));self.assertAlmostEqual(sum(p.values()),1)
        self.assertEqual(probabilities(p0,{},False),p0)
        self.assertEqual(probabilities(p0,{},epsilon=1),p0)
    def test_native_fallback_and_slot_preservation(self):
        ids=[6]*300+[7]*300+[8]*700;ds=list(range(len(ids)))
        native=RoutedSampler(ds,8,42,{6:.5,7:.5},epsilon=1,task_ids=ids)
        route=RoutedSampler(ds,8,42,{6:.25,7:.75},task_ids=ids)
        state=random.getstate();ns=np.random.get_state()
        for epoch in [0,1,99]:
            native.set_epoch(epoch);route.set_epoch(epoch);a=list(native);b=list(route)
            self.assertEqual(a,list(native.native))
            self.assertEqual([i for i in a if ids[i]==8],[i for i in b if ids[i]==8])
            self.assertEqual([ids[i]==8 for i in a],[ids[i]==8 for i in b])
            self.assertEqual(b,list(route));self.assertEqual(len(b),len(ds))
        self.assertEqual(state,random.getstate());self.assertTrue(np.array_equal(ns[1],np.random.get_state()[1]))
    def test_frequency_without_seed_selection(self):
        ids=[6]*1000+[7]*1000+[8]*1000;ds=list(range(len(ids)))
        s=RoutedSampler(ds,8,42,{6:.25,7:.75},task_ids=ids);n=c=0
        for epoch in range(20):
            s.set_epoch(epoch);seq=list(s);c+=sum(ids[i]==6 for i in seq);n+=2000
        self.assertLess(abs(c/n-.25),6*(.25*.75/n)**.5)
    def test_future_rejected(self):
        with self.assertRaises(AssertionError):RoutedSampler([0,1],8,42,{},task_ids=[8,9])
    def test_trial_validation_rejects_duplicates(self):
        import json,tempfile
        from pathlib import Path
        from .queue import validate_trials
        rows=[dict(stage=6,epoch=99,task=6,initial_state_index=i) for i in range(20)]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'trials.jsonl';p.write_text('\n'.join(json.dumps(r) for r in rows))
            self.assertEqual(len(validate_trials(p,6,[6])),20)
            p.write_text('\n'.join(json.dumps(r) for r in rows[:-1]+[rows[0]]))
            with self.assertRaises(AssertionError):validate_trials(p,6,[6])
    def test_complete_report_pipeline_synthetic(self):
        import csv,json,tempfile
        from pathlib import Path
        import torch
        from . import queue
        from .io import write,sha
        saved=queue.ROOT,queue.OUT
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);out=root/'reports';rep=out/'seed42';run=root/'run';features=root/'features';buffer=run/'buffer'
                buffer.mkdir(parents=True);rep.mkdir(parents=True)
                queue.ROOT=root;queue.OUT=out
                matrix=[[35,None,None,None],[50,100,None,None],[40,95,100,None],[25,100,100,75]]
                budgets={'6':157,'7':227,'8':309,'9':362}
                write(out/'PROTOCOL.json',dict(implementation_files={},updates_per_epoch=budgets,baseline_results={'42':{'r':matrix}}))
                write(out/'CONFIG_seed42.json',dict(tsr_root=str(run),tsr_recovery=str(run/'recovery.pt'),feature_dir=str(features)))
                write(out/'INPUT_MANIFEST.json',dict(baseline_checkpoints={}))
                write(run/'TRAINING_COMPLETE.json',dict(status='PASS',synthetic_unit_fixture=True))
                epochs=[];files=[]
                for t in range(6,10):
                    (features/str(t)).mkdir(parents=True);(features/str(t)/'0.pt').touch()
                    name='t%d__0.pt'%t;(buffer/name).touch();files.append(name)
                    write(rep/('ROUTE_task%d.json'%t),dict(task=t,p={}))
                    write(rep/('SIGNATURE_task%d.json'%t),dict(norm=1.,seconds=1.,valid=True,reference_model_sha256='synthetic'))
                    for e in range(100):epochs.append(dict(stage=t,epoch=e,steps=budgets[str(t)]))
                torch.save(dict(stage_complete=True,task=9,totals={'steps':105500},buffer_files=files,buffer_dir=str(buffer)),run/'recovery.pt')
                with (rep/'training_log.csv').open('w',newline='') as f:w=csv.DictWriter(f,['stage','epoch','steps']);w.writeheader();w.writerows(epochs)
                for arm in ['v1','baseline']:
                    for t in range(6,10):
                        p=rep/'evaluation'/('%s_primary_stage%d'%(arm,t))/'trials.jsonl';p.parent.mkdir(parents=True)
                        rows=[dict(stage=t,epoch=99,task=j,initial_state_index=i,success=i<matrix[t-6][j-6]/5) for j in range(6,t+1) for i in range(20)]
                        p.write_text('\n'.join(json.dumps(r) for r in rows))
                    p=rep/'evaluation'/('%s_final_base_stage9'%arm)/'trials.jsonl';p.parent.mkdir(parents=True)
                    rows=[dict(stage=9,epoch=99,task=j,initial_state_index=i,success=i<10) for j in range(6) for i in range(20)]
                    p.write_text('\n'.join(json.dumps(r) for r in rows))
                queue.summarize()
                result=json.loads((out/'SUMMARY.json').read_text())
                self.assertEqual(result['screen'],'SCREEN_NOT_MET');self.assertEqual(result['rollouts'],640)
                self.assertEqual(result['historical_delta']['auc'],0.)
                self.assertIn('single', (out/'FINAL_REPORT.md').read_text().lower())
        finally:queue.ROOT,queue.OUT=saved


if __name__=='__main__':unittest.main()
