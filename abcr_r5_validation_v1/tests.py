import tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from .common import PROTOCOL,write,read,deadband
from .engine import aux
from .analysis import no_leak,predict,wilson

class TestR5(unittest.TestCase):
    def test_stop_gradient(self):
        z=torch.tensor([[1.,.2],[.3,1.]],requires_grad=True);pairs=[dict(eligible=True,label='SHARE',new_slot=0,old_slot=1)];l,_,_=aux(z,pairs,'SHARE_ONLY');l.backward();self.assertGreater(float(z.grad[0].norm()),0);self.assertEqual(float(z.grad[1].norm()),0)
    def test_empty_zero(self):
        z=torch.randn(10,8,requires_grad=True);l,_,c=aux(z,[],'ABCR');self.assertEqual(float(l),0);l.backward();self.assertEqual(float(z.grad.norm()),0);self.assertEqual(c,dict(SHARE=0,BIF=0))
    def test_ineligible_not_used(self):
        z=torch.randn(2,8,requires_grad=True);l,_,_=aux(z,[dict(eligible=False,label='BIF',new_slot=0,old_slot=1)],'ABCR');self.assertEqual(float(l),0)
    def test_class_loss(self):
        z=torch.tensor([[1.,0.],[1.,0.]],requires_grad=True);l,_,_=aux(z,[dict(eligible=True,label='BIF',new_slot=0,old_slot=1)],'BIF_ONLY');self.assertAlmostEqual(float(l),.025)
    def test_no_checkpoint_or_trajectory_leak(self):
        a=dict(start_id='a',new_trajectory='7:1',old_trajectory='6:2');b=dict(start_id='b',new_trajectory='8:3',old_trajectory='7:1');self.assertFalse(no_leak(a,b));b['old_trajectory']='7:2';self.assertTrue(no_leak(a,b));b['start_id']='a';self.assertFalse(no_leak(a,b))
    def test_regression_no_test_label(self):
        train=[dict(x=float(i),old_delta=2.*i) for i in range(30)];self.assertEqual(predict(train,dict(x=4.,old_delta=99),['x']),predict(train,dict(x=4.,old_delta=-99),['x']))
    def test_atomic_json_finite(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'x.json';write(p,dict(x=1));self.assertEqual(read(p),dict(x=1));self.assertRaises(ValueError,write,p,dict(x=float('nan')));self.assertEqual(read(p),dict(x=1))
    def test_frozen_grid(self):
        self.assertEqual(PROTOCOL['C']['batch_size'],10);self.assertEqual(PROTOCOL['C']['lambda_bif'],.05);self.assertGreater(deadband(-20),deadband(0));self.assertLess(wilson(1,8)[0],.8)

if __name__=='__main__':unittest.main()
