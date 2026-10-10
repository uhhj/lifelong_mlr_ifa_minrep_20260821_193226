"""Synthetic software tests only. Never evidence of OBJECT training/rollouts."""
import copy
import importlib.util
import json
import os
import random
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data._utils.collate import default_collate

ROOT = Path(__file__).resolve().parents[1]
def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result)
    return result
s=module('e1_protective_selector',ROOT/'source/libero/lifelong/replay_selection.py')
head=module('baseline_policy_head',ROOT/'baseline_evidence/policy_head.py')
sys.modules['e1_test_package']=types.ModuleType('e1_test_package')
sys.modules['e1_test_package.replay_selection']=s
hook=module('e1_test_package.replay_selection_hook',ROOT/'source/libero/lifelong/replay_selection_hook.py')
DEVICE=os.environ.get('E1_TEST_DEVICE','cpu')

class FixturePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder=nn.Sequential(nn.Linear(4,4),nn.Dropout(.25))
        self.policy_head=head.GMMHead(4,2,hidden_size=8,num_layers=1,low_eval_noise=True)
    def temporal_encode(self,x):return self.decoder(x[:,:,0,:])

class NativeFixtureUpdate:
    def __init__(self,policy):
        self.policy=policy;self.cfg=types.SimpleNamespace(seed=42)
        self.optimizer=torch.optim.AdamW(policy.parameters(),lr=1e-4)
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,lambda n:1.)
        self._optimizer_step_count=0
    def observe(self,data,current_task_id,epoch,ifa_info=None):
        self.optimizer.zero_grad()
        x=data[0]['feature'].squeeze(1).to(DEVICE);y=data[0]['actions'].squeeze(1).to(DEVICE)
        d=self.policy.policy_head(self.policy.temporal_encode(x)[:,-1,:])
        loss=self.policy.policy_head.loss_fn(d,y);loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(),100.)
        self.optimizer.step();self.scheduler.step();self._optimizer_step_count+=1
        return loss.item()

class HookedFixtureUpdate(hook.ProtectiveSelectionMixin,NativeFixtureUpdate):pass

class TestSelection(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.policy=FixturePolicy().to(DEVICE)
        self.payload={}
        for task in (6,7,8):
            for i in range(9):
                x=torch.randn(1,3,2,4);x[0,0,0]=task
                self.payload['%d/%d'%(task,i)]={'feature':x,'actions':torch.randn(1,3,2)}
        self.old={t:[k for k in self.payload if k.startswith(str(t)+'/')] for t in (6,7)}
        self.new=[k for k in self.payload if k.startswith('8/')]
        self.loaded=[]
        self.selector=self.make()
        self.ids=['6/0','8/0','7/0','6/1','8/1']
        self.batch=[default_collate([self.payload[k] for k in self.ids]),tuple(k.split('/')[0] for k in self.ids)]
    def load(self,k):self.loaded.append(k);return self.payload[k]
    def make(self,**kwargs):
        return s.ProtectiveReplaySelector(self.policy,42,8,self.old,self.new,self.load,'fixture-only',s.SelectionConfig(**kwargs))
    def test_score_formula(self):
        g=torch.tensor([1.,0.]);n=-g
        self.assertEqual(s.protective_score(g,g,n),0.)
        self.assertEqual(s.protective_score(g,g,None),1.)
        self.assertIsNone(s.protective_score(torch.zeros(2),g,n))
    def test_schedule_stage_steps(self):
        self.assertTrue(self.selector.should_refresh(0));self.selector.refresh(0)
        self.assertFalse(self.selector.should_refresh(499));self.assertTrue(self.selector.should_refresh(500))
        empty=s.ProtectiveReplaySelector(self.policy,42,6,{},self.new,self.load,'empty')
        self.assertFalse(empty.should_refresh(0))
    def test_reference_candidate_disjoint_and_legal(self):
        r=self.selector.refresh(0)
        for t in ('6','7'):
            self.assertFalse(set(r['references'][t])&set(r['candidates'][t]))
            self.assertTrue(set(r['references'][t]+r['candidates'][t])<=set(self.old[int(t)]))
            self.assertAlmostEqual(sum(r['probabilities'][t]),1.)
        self.assertTrue(set(self.loaded)<=set(self.payload))
    def test_parameters_grads_modes_and_rng_unchanged(self):
        self.policy.decoder[1].eval()
        for p in self.policy.parameters():p.grad=torch.ones_like(p)
        params=copy.deepcopy(self.policy.state_dict());grads=[p.grad.clone() for p in self.policy.parameters()]
        modes=[m.training for m in self.policy.modules()]
        py=random.getstate();np_state=np.random.get_state();cpu=torch.get_rng_state().clone()
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        self.selector.refresh(0)
        self.assertTrue(all(torch.equal(v,self.policy.state_dict()[k]) for k,v in params.items()))
        self.assertTrue(all(torch.equal(p.grad,g) for p,g in zip(self.policy.parameters(),grads)))
        self.assertEqual(modes,[m.training for m in self.policy.modules()])
        self.assertEqual(py,random.getstate());self.assertTrue(np.array_equal(np_state[1],np.random.get_state()[1]))
        self.assertTrue(torch.equal(cpu,torch.get_rng_state()))
        if cuda:self.assertTrue(all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all())))
    def test_head_only_gradient_and_full_density(self):
        batch=[self.payload['6/0'],self.payload['6/1']]
        with s.isolated_scoring(self.policy):
            g=s.head_gradient(self.policy,batch)
            data=default_collate(batch)
            x=data['feature'].squeeze(1).to(DEVICE);y=data['actions'].squeeze(1).to(DEVICE)
            z=self.policy.temporal_encode(x)[:,-1,:]
            distribution=self.policy.policy_head(z)
            loss=self.policy.policy_head.loss_fn(distribution,y)
            expected=torch.cat([v.reshape(-1) for v in torch.autograd.grad(loss,tuple(self.policy.policy_head.parameters()))])
            self.assertTrue(torch.allclose(g,expected,rtol=1e-6,atol=1e-7))
            self.assertGreater(distribution.component_distribution.base_dist.scale.max().item(),1e-4)
        self.assertTrue(all(p.grad is None for p in self.policy.decoder.parameters()))
    def test_task_balanced_history(self):
        self.selector.old[6]=self.selector.old[6][:3]
        seen=[]
        def gradient(policy,payloads):
            return torch.tensor([float(payloads[0]['feature'][0,0,0,0]),1.],device=DEVICE)
        native=s.protective_score
        def score(g,h,n,*args):seen.append(h.clone());return native(g,h,n,*args)
        with patch.object(s,'head_gradient',side_effect=gradient),patch.object(s,'protective_score',side_effect=score):self.selector.refresh(0)
        self.assertTrue(all(torch.allclose(h,torch.tensor([6.5,1.],device=DEVICE)) for h in seen))
    def test_old_same_task_replacement_preserves_current(self):
        sel=self.make(priority_replace_probability=1.);sel.refresh(0)
        before=copy.deepcopy(self.batch);result,events=sel.replace_old_slots(self.batch)
        self.assertEqual(result[1],before[1]);self.assertEqual(len(result[1]),5)
        for key in result[0]:
            self.assertTrue(torch.equal(self.batch[0][key],before[0][key]))
            self.assertTrue(torch.equal(result[0][key][[1,4]],before[0][key][[1,4]]))
        for event in events:
            self.assertTrue(event['candidate'].startswith(str(event['task'])+'/'))
            self.assertTrue(event['applied'])
            for key in result[0]:self.assertTrue(torch.equal(result[0][key][event['slot']],self.payload[event['candidate']][key]))
    def test_first_language_reference_veto(self):
        sel=self.make(priority_replace_probability=1.);sel.refresh(0)
        for k in self.old[6]:self.payload[k]['feature'][0,0,0]=99
        result,events=sel.replace_old_slots(self.batch)
        self.assertFalse(events[0]['applied']);self.assertEqual(events[0]['reason'],'IFA_LANGUAGE_REFERENCE')
        self.assertTrue(torch.equal(result[0]['feature'][0],self.batch[0]['feature'][0]))
    def test_disabled_and_zero_probability_identity(self):
        for kw in ({'enabled':False},{'priority_replace_probability':0.}):
            sel=self.make(**kw);before=sel.rng.getstate()
            self.assertFalse(sel.should_refresh(0));self.assertIs(sel.replace_old_slots(self.batch)[0],self.batch)
            self.assertEqual(sel.rng.getstate(),before)
    def test_json_resume_matches_next_refresh_and_replacements(self):
        self.selector.refresh(0);state=json.loads(json.dumps(self.selector.state_dict()))
        other=self.make();other.load_state_dict(state)
        a,e=self.selector.replace_old_slots(self.batch);b,f=other.replace_old_slots(self.batch)
        self.assertEqual(e,f);self.assertTrue(all(torch.equal(a[0][k],b[0][k]) for k in a[0]))
        self.assertEqual(self.selector.refresh(500),other.refresh(500))
    def test_restore_rejects_wrong_catalog(self):
        state=self.selector.state_dict();state['catalog_fingerprint']='wrong'
        with self.assertRaises(AssertionError):self.selector.load_state_dict(state)
    def test_invalid_history_original_fallback(self):
        with patch.object(s,'head_gradient',return_value=torch.zeros(2,device=DEVICE)):r=self.selector.refresh(0)
        self.assertEqual(r['reason'],'INVALID_HISTORICAL_DIRECTION')
        self.assertIs(self.selector.replace_old_slots(self.batch)[0],self.batch)
    def test_invalid_current_history_only(self):
        def gradient(policy,payloads):
            task=int(payloads[0]['feature'][0,0,0,0])
            return None if task==8 else torch.tensor([1.,2.],device=DEVICE)
        with patch.object(s,'head_gradient',side_effect=gradient):r=self.selector.refresh(0)
        self.assertEqual(r['reason'],'HISTORY_ONLY_INVALID_CURRENT_DIRECTION');self.assertEqual(len(self.selector.distributions),2)
    def test_invalid_candidates_excluded(self):
        def gradient(policy,payloads):
            return None if len(payloads)==1 else torch.tensor([1.,2.],device=DEVICE)
        with patch.object(s,'head_gradient',side_effect=gradient):self.selector.refresh(0)
        self.assertFalse(self.selector.distributions);self.assertGreater(self.selector.totals['invalid_candidates'],0)
    def test_exception_restores_modes_rng(self):
        before=torch.get_rng_state();modes=[m.training for m in self.policy.modules()]
        with self.assertRaises(RuntimeError):
            with s.isolated_scoring(self.policy):torch.rand(3);raise RuntimeError('fixture')
        self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertEqual(modes,[m.training for m in self.policy.modules()])
    def test_future_task_rejected(self):
        with self.assertRaises(AssertionError):s.ProtectiveReplaySelector(self.policy,42,8,{9:['9/0']},[],self.load,'invalid')
    def test_nonfinite_payload_is_not_a_score(self):
        bad=copy.deepcopy(self.payload['6/0']);bad['feature'].fill_(float('nan'))
        with s.isolated_scoring(self.policy):self.assertIsNone(s.head_gradient(self.policy,[bad]))
    def test_hook_zero_probability_optimizer_parity(self):
        native=NativeFixtureUpdate(copy.deepcopy(self.policy));e1=HookedFixtureUpdate(copy.deepcopy(self.policy))
        e1.configure_e1_stage(8,self.old,self.new,self.load,'fixture',s.SelectionConfig(priority_replace_probability=0.))
        torch.manual_seed(77);a=native.observe(copy.deepcopy(self.batch),8,0)
        cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(77);b=e1.observe(copy.deepcopy(self.batch),8,0)
        self.assertEqual(a,b)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(native.policy.parameters(),e1.policy.parameters())))
        for p,q in zip(native.policy.parameters(),e1.policy.parameters()):
            for k in native.optimizer.state[p]:self.assertTrue(torch.equal(native.optimizer.state[p][k],e1.optimizer.state[q][k]))
        self.assertEqual(native.scheduler.state_dict(),e1.scheduler.state_dict())
        self.assertTrue(torch.equal(cpu,torch.get_rng_state()))
        if cuda:self.assertTrue(all(torch.equal(x,y) for x,y in zip(cuda,torch.cuda.get_rng_state_all())))
        self.assertEqual(e1.e1_stage_step,1)
    def test_hook_enabled_one_update_and_sidecar(self):
        e1=HookedFixtureUpdate(self.policy)
        e1.configure_e1_stage(8,self.old,self.new,self.load,'fixture')
        e1.observe(copy.deepcopy(self.batch),8,0)
        self.assertEqual(e1._optimizer_step_count,1)
        self.assertEqual(e1.scheduler.last_epoch,1)
        state=json.loads(json.dumps(e1.e1_sidecar()));e1.restore_e1_sidecar(state)
        self.assertEqual(e1.e1_stage_step,1)
        self.assertEqual(e1.e1_selector.last_refresh_step,0)
        self.assertEqual(len(e1.pop_e1_events()),1);self.assertEqual(e1.pop_e1_events(),[])
    def test_hook_requires_stage_catalog(self):
        e1=HookedFixtureUpdate(self.policy)
        with self.assertRaises(RuntimeError):e1.observe(self.batch,8,0)

if __name__=='__main__':
    assert DEVICE=='cpu' or torch.cuda.is_available(), 'No CPU fallback for CUDA tests'
    print('SYNTHETIC_SOFTWARE_ONLY device=%s torch=%s'%(DEVICE,torch.__version__),flush=True)
    unittest.main(verbosity=2)
