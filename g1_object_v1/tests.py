import json
import tempfile
import unittest
from pathlib import Path
import torch
from .gate import select_loss,gradient_statistics
from .summary import build_result,render
from .io import write


class Tests(unittest.TestCase):
    def test_sign_cases(self):
        for a,b,expected in [([1.,0.],[2.,0.],1),([1.,0.],[-2.,0.],0),([1.,0.],[0.,2.],1),([0.,0.],[1.,0.],1)]:
            p=torch.nn.Parameter(torch.zeros(2))
            bc=(p*torch.tensor(a)).sum();aux=1+(p*torch.tensor(b)).sum();native=bc+aux
            p.grad=torch.ones_like(p)*3;before=p.grad.clone()
            selected,row=select_loss(bc,native,[aux],(p,))
            self.assertEqual(row['w'],expected);self.assertTrue(torch.equal(p.grad,before))
            self.assertIs(selected,native if expected else bc)
    def test_unused_and_zero(self):
        p=torch.nn.Parameter(torch.ones(2));q=torch.nn.Parameter(torch.ones(2))
        bc=p.sum()+q.sum();aux=p.sum()
        _,r=select_loss(bc,bc+aux,[aux],(p,q))
        self.assertEqual(r['ifa_unused'],1)
        _,r=select_loss(bc,bc,[],(p,q));self.assertIsNone(r['dot']);self.assertEqual(r['reason'],'NO_IFA')
        z=p.sum()*0;_,r=select_loss(bc,bc+z,[z],(p,q));self.assertEqual(r['reason'],'HINGE_ZERO')
    def test_disconnected_nonfinite_stop(self):
        p=torch.nn.Parameter(torch.ones(2));q=torch.nn.Parameter(torch.ones(2))
        with self.assertRaisesRegex(AssertionError,'DISCONNECTED'):
            select_loss(p.sum(),p.sum()+q.sum(),[q.sum()],(p,))
        with self.assertRaisesRegex(AssertionError,'NONFINITE'):
            select_loss(p.sum()*float('nan'),p.sum(),[],(p,))
    def test_chunked_float64(self):
        # Binary fractions and an exactly representable sum isolate chunk
        # correctness from BLAS's different floating-point reduction order.
        a=torch.arange(300000,dtype=torch.float32).remainder(1024)/1024;b=a.flip(0)
        r=gradient_statistics((a,),(b,))
        self.assertEqual(r['dot'],float(a.double()@b.double()))
    def test_complete_summary_pipeline(self):
        base=[[35,None,None,None],[50,100,None,None],[40,95,100,None],[25,100,100,75]]
        equal=build_result(42,base,base);self.assertFalse(equal['run_additional_seeds'])
        higher=[row[:] for row in base];higher[3][3]+=.000001
        positive=build_result(42,higher,base)
        self.assertTrue(positive['run_additional_seeds'])
        self.assertEqual(round(positive['metrics']['auc'],3),round(equal['metrics']['auc'],3))
        lower=[row[:] for row in base];lower[3][3]-=5
        negative=build_result(42,lower,base);self.assertFalse(negative['run_additional_seeds'])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'RESULT.json';write(p,positive)
            self.assertEqual(json.loads(p.read_text()),positive)
            report=render([positive]);self.assertIn('POSITIVE_SINGLE_SEED_SIGNAL',report)
            self.assertIn('|42|',report)

    def test_real_queue_summary_entry(self):
        import csv
        from . import queue
        saved=queue.ROOT,queue.OUT
        matrix=[[35,None,None,None],[50,100,None,None],[40,95,100,None],[25,100,100,75]]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);out=root/'reports';run=root/'runs/g1_object_v1/seed42';rep=out/'seed42'
                rep.mkdir(parents=True);run.mkdir(parents=True);(run/'buffer').mkdir();(rep/'events').mkdir()
                queue.ROOT=root;queue.OUT=out
                write(out/'PROTOCOL.json',dict(implementation_files={},baseline_results={'42':{'r':matrix}}))
                write(out/'CONFIG_seed42.json',dict(g1_recovery=str(run/'recovery.pt')))
                torch.save(dict(stage_complete=True,task=9,buffer_files=[],buffer_dir=str(run/'buffer'),totals={}),run/'recovery.pt')
                write(run/'TRAINING_COMPLETE.json',dict(status='PASS'))
                trials=[];lines=[];epochs=[]
                for s in range(6,10):
                    for t in range(6,s+1):
                        value=matrix[s-6][t-6];lines.append('[task %d epoch 99 and %d succ.] %s'%(t,s,value/100))
                        trials.extend(dict(stage=s,epoch=99,task=t,initial_state_index=i,success=i<value/5,
                            checkpoint=str(run/'checkpoints'/('mlrifa_%d_model_ep99.pth'%s))) for i in range(20))
                    for e in range(100):
                        epochs.append(dict(stage=s,epoch=e,steps=1))
                        event=dict(step=e+1,has_ifa=False,hinge_zero=False,dot=None,w=None,weighted_terms=[],
                            retained_ifa=0.,reason='NO_IFA',probe_seconds=0.,dot_seconds=0.,peak_gpu_bytes=0)
                        (rep/'events'/('task%d_epoch%03d.jsonl'%(s,e))).write_text(json.dumps(event)+'\n')
                (rep/'trials_primary.jsonl').write_text('\n'.join(json.dumps(x) for x in trials))
                (run/'primary_ep99.log').write_text('\n'.join(lines))
                with (rep/'training_log.csv').open('w',newline='') as f:
                    writer=csv.DictWriter(f,['stage','epoch','steps']);writer.writeheader();writer.writerows(epochs)
                result=queue.summarize(42)
                self.assertFalse(result['run_additional_seeds']);self.assertEqual(result['delta']['auc'],0.)
                self.assertFalse(json.loads((out/'SEED42_DECISION.json').read_text())['run_additional_seeds'])
                (out/'FINAL_REPORT.md').write_text(render([result]))
                self.assertTrue((out/'FINAL_VERIFICATION_seed42.json').exists())
        finally:queue.ROOT,queue.OUT=saved


if __name__=='__main__':unittest.main()
