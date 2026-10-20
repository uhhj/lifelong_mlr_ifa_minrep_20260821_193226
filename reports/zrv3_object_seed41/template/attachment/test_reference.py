"""Synthetic CPU tests only; no real embeddings, demonstrations, or policies."""
import copy
import json
import tempfile
import unittest
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import torch
from reference_core import extend_gates, normalize_rows, MeanCalibrationBank, corrected_gmm
from compare_metrics import metrics, compare, LOCKED

torch.set_num_threads(1)


def make_bank():
    e = np.eye(512, dtype=np.float64)[:8]
    w = e[6:8].copy()
    return MeanCalibrationBank([6, 7], w), e


def raw_params(n):
    gen = torch.Generator().manual_seed(912)
    return torch.randn(n, 5, 7, generator=gen), torch.ones(n,5,7)*0.2, torch.randn(n,5,generator=gen)


class GateTests(unittest.TestCase):
    def test_recursive_identity_and_old_responses(self):
        gen = np.random.default_rng(3)
        e = normalize_rows(gen.normal(size=(10, 512)))
        w = np.empty((0,512))
        for k in range(6,10):
            old = w.copy()
            w, report = extend_gates(e[:k], e[k], w)
            np.testing.assert_allclose(e[:k+1] @ w.T, np.eye(k+1)[:,6:], atol=1e-10)
            if len(old):
                np.testing.assert_allclose(e[:k] @ old.T, e[:k] @ w[:-1].T, atol=1e-10)
            self.assertLessEqual(max(report['gate_gains']),100)

    def test_recursive_update_really_zeros_old_to_new(self):
        e = np.eye(512)[:8].copy()
        e[7] = (e[6]+e[7]) / np.sqrt(2)
        w,_ = extend_gates(e[:6],e[6],np.empty((0,512)))
        self.assertGreater(abs(e[7] @ w[0]),0.7)
        w2,_ = extend_gates(e[:7],e[7],w)
        self.assertLess(abs(e[7] @ w2[0]),1e-10)
        self.assertAlmostEqual(float(e[6]@w2[0]),1)

    def test_duplicate_current_rejected(self):
        e = np.eye(512)[:6]
        with self.assertRaises(ValueError): extend_gates(e,e[0],np.empty((0,512)))

    def test_near_span_rejected(self):
        e = np.eye(512)[:7]
        with self.assertRaises(ValueError): extend_gates(e[:6],e[0]+1e-4*e[6],np.empty((0,512)))

    def test_old_dependence_allowed(self):
        e = np.eye(512)[:7].copy()
        e[1] = e[0]
        w,_ = extend_gates(e[:6],e[6],np.empty((0,512)))
        self.assertEqual(w.shape,(1,512))

    def test_zero_language_rejected(self):
        with self.assertRaises(ValueError): normalize_rows(np.zeros((2,512)))

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError): extend_gates(np.eye(512)[:7],np.eye(512)[7],np.empty((0,512)))


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.bank,self.e = make_bank()
        self.q = torch.randn(4,512,generator=torch.Generator().manual_seed(45))
        self.lang = torch.tensor(self.e[[6,7,6,7]],dtype=torch.float32)
        self.ids = torch.tensor([6,7,6,7])

    def test_local_initialization_preserves_torch_rng(self):
        state = torch.random.get_rng_state().clone()
        make_bank()
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_zero_init(self):
        self.assertTrue(torch.equal(self.bank(self.q,self.lang),torch.zeros(4,7)))

    def test_nonzero_own_and_zero_other(self):
        with torch.no_grad(): self.bank.branches['6'].U.fill_(0.2)
        d = self.bank(self.q,self.lang)
        self.assertGreater(d[[0,2]].abs().max().item(),0)
        self.assertEqual(d[[1,3]].abs().max().item(),0)

    def test_id_restriction_does_not_change_forward_values(self):
        with torch.no_grad():
            for p in self.bank.parameters(): p.add_(0.1)
        self.assertTrue(torch.equal(self.bank(self.q,self.lang),self.bank(self.q,self.lang,self.ids)))

    def test_mean_bound_and_distribution_identity(self):
        b,s,l=raw_params(4)
        with torch.no_grad():
            for branch in self.bank.branches.values(): branch.U.fill_(1e4)
        delta=self.bank(self.q,self.lang)
        out=corrected_gmm(b,s,l,delta)
        self.assertLessEqual(delta.abs().max().item(),0.100001)
        self.assertLessEqual((out['means']-b.tanh()).abs().max().item(),0.100001)
        self.assertTrue(torch.equal(out['scales'],s))
        self.assertTrue(torch.equal(out['logits'],l))
        self.assertTrue(torch.equal(out['selected'],l.argmax(-1)))

    def test_absent_branch_not_updated_including_adam_momentum(self):
        opt=torch.optim.AdamW(self.bank.parameters(),lr=1e-3,weight_decay=0.1)
        b,s,l=raw_params(4); target=torch.ones(4,7)*0.2
        for owner in (6,7):
            opt.zero_grad(set_to_none=True)
            ee=torch.tensor(self.e[owner],dtype=torch.float32).expand(4,-1)
            d=self.bank(self.q,ee,torch.full((4,),owner))
            loss=-corrected_gmm(b,s,l,d)['gmm'].log_prob(target).mean()
            loss.backward()
            other='7' if owner==6 else '6'
            pars=list(self.bank.branches[other].parameters())
            self.assertTrue(all(p.grad is None for p in pars))
            values=[p.detach().clone() for p in pars]
            states=[copy.deepcopy(opt.state.get(p,{})) for p in pars]
            opt.step()
            for p,v,state in zip(pars,values,states):
                self.assertTrue(torch.equal(p,v))
                now=opt.state.get(p,{})
                self.assertEqual(set(now),set(state))
                for key in state:
                    self.assertTrue(torch.equal(now[key],state[key]) if torch.is_tensor(state[key]) else now[key]==state[key])

    def test_fixed_inputs_get_no_grad(self):
        # q is deliberately differentiable: bank must stop it as a fixed input.
        q=self.q.clone().requires_grad_(True)
        b,s,l=raw_params(4)
        out=corrected_gmm(b,s,l,self.bank(q,self.lang,self.ids))
        (-out['gmm'].log_prob(torch.zeros(4,7)).mean()).backward()
        self.assertIsNone(q.grad)
        self.assertIsNotNone(self.bank.branches['6'].U.grad)

    def test_real_language_not_replaced_by_ids(self):
        with torch.no_grad(): self.bank.branches['6'].U.fill_(0.1)
        # Deliberately mismatch labels: values still come from language, not labels.
        d=self.bank(self.q,self.lang,torch.full((4,),7))
        self.assertTrue(torch.equal(d,self.bank(self.q,self.lang)))

    def test_reload_keeps_gates_and_output(self):
        with torch.no_grad(): self.bank.branches['6'].U.fill_(0.1)
        b2,_=make_bank(); b2.load_state_dict(self.bank.state_dict())
        self.assertTrue(torch.equal(b2(self.q,self.lang),self.bank(self.q,self.lang)))
        self.assertEqual(b2.gate_w.dtype,torch.float64)

    def test_downcast_rejected(self):
        self.bank.float()
        with self.assertRaises(RuntimeError): self.bank(self.q,self.lang)

    def test_unknown_training_label_rejected(self):
        with self.assertRaises(ValueError): self.bank(self.q,self.lang,torch.full((4,),9))


MATRIX = [[40,None,None,None],[95,100,None,None],[75,100,85,None],[65,95,100,55]]


def rates_matrix():
    return {(s,t):F(MATRIX[s-6][t-6],100) for s in range(6,10) for t in range(6,s+1)}


def write_records(folder,is_new,improve=False):
    folder.mkdir()
    for s in range(6,10):
        for t in range(6,s+1):
            successes=MATRIX[s-6][t-6]//5
            if improve and s==9 and t in (6,9): successes+=1
            cp=('%064x' % (100+s))
            rows=[]
            for trial in range(20):
                r=dict(seed=43,stage=s,task=t,trial=trial,initial_state_index=trial,
                       initial_state_sha256='%064x' % (1000+trial),modal_sha256='a'*64,
                       checkpoint_sha256='b'*64 if is_new else cp,
                       arm='coreset_ifa_zrv2' if is_new else 'coreset',success=trial<successes)
                if is_new: r.update(parent_epoch=99,calibration_epoch=9,parent_checkpoint_sha256=cp)
                else: r['epoch']=99
                rows.append(r)
            (folder/('trials_stage%d_task%d.jsonl'%(s,t))).write_text('\n'.join(json.dumps(r) for r in rows)+'\n')


class MetricsTests(unittest.TestCase):
    def test_exact_locked_baseline(self): self.assertEqual(metrics(rates_matrix()),LOCKED)
    def test_all_success(self):
        m=metrics({k:F(1) for k in rates_matrix()})
        self.assertEqual(m['AUC'],1);self.assertEqual(m['FWT'],1);self.assertEqual(m['NBT'],0)
    def test_missing_matrix_cell_rejected(self):
        r=rates_matrix();r.pop((6,6))
        with self.assertRaises(ValueError): metrics(r)
    def test_record_comparison_and_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/'base';new=Path(tmp)/'new'
            write_records(base,False);write_records(new,True,True)
            out=compare(base,new)
            self.assertTrue(out['metric_gate']);self.assertFalse(out['auto_extend'])
            self.assertEqual(out['engineering_acceptance'],'NOT_CHECKED_BY_THIS_TOOL')
    def test_equal_results_do_not_pass_strict_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/'base';new=Path(tmp)/'new'
            write_records(base,False);write_records(new,True)
            self.assertFalse(compare(base,new)['metric_gate'])
    def test_wrong_parent_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/'base';new=Path(tmp)/'new'
            write_records(base,False);write_records(new,True)
            p=new/'trials_stage6_task6.jsonl';s=p.read_text();p.write_text(s.replace('%064x'%106,'c'*64))
            with self.assertRaises(ValueError): compare(base,new)
    def test_nonboolean_success_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/'base';new=Path(tmp)/'new'
            write_records(base,False);write_records(new,True)
            p=new/'trials_stage6_task6.jsonl';p.write_text(p.read_text().replace('"success": false','"success": 0'))
            with self.assertRaises(ValueError): compare(base,new)

if __name__=='__main__': unittest.main(verbosity=2)
