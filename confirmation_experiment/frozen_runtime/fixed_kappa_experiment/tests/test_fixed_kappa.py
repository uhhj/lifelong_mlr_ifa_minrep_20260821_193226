import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


soft = load('soft_core', ROOT/'src/fixed_kappa.py')
hard = load('hard_fixture', ROOT/'tests/legacy_core.py')


def equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b), (a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal(x, y)
    else:
        assert a == b, (a, b)


class CoreTests(unittest.TestCase):
    def project(self, g, r, kappa=.5, **kwargs):
        p = torch.nn.Parameter(torch.zeros(len(g), dtype=torch.float64))
        p.grad = torch.tensor(g, dtype=torch.float64)
        m = soft.project_parameter_grads_([p], [torch.tensor(r, dtype=torch.float64)], kappa=kappa, **kwargs)
        return p.grad, m

    def test_known_half_correction(self):
        g, m = self.project([-2., 3.], [1., 0.])
        equal(g, torch.tensor([-1., 3.], dtype=torch.float64))
        self.assertEqual(m['dot_before'], -2.)
        self.assertEqual(m['dot_after'], -1.)
        self.assertTrue(m['projected'])

    def test_repeated_soft_is_not_idempotent(self):
        g, _ = self.project([-2., 3.], [1., 0.])
        again, _ = self.project(g.tolist(), [1., 0.])
        self.assertEqual(again[0], -.5)

    def test_zero_aligned_tiny_disabled(self):
        for g, r, kw in [([-2,3],[1,0],{'kappa':0}), ([2,3],[1,0],{}),
                         ([-2,3],[0,0],{}), ([-2,3],[1e-15,0],{}),
                         ([-2,3],[1,0],{'enabled':False}), ([0,3],[1,0],{})]:
            out, m = self.project(g, r, **kw)
            equal(out, torch.tensor(g,dtype=torch.float64))
            self.assertFalse(m['projected'])

    def test_global_not_blockwise(self):
        p, q = [torch.nn.Parameter(torch.zeros(1)) for _ in range(2)]
        p.grad, q.grad = torch.tensor([-1.]), torch.tensor([2.])
        m = soft.project_parameter_grads_([p,q],[torch.ones(1),torch.ones(1)])
        self.assertFalse(m['projected']); self.assertEqual(p.grad.item(), -1.)

    def test_invalid_coefficients_before_mutation(self):
        for k in (True, -.1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError): self.project([-2,3],[1,0],kappa=k)

    def test_nonfinite_aborts(self):
        with self.assertRaises(FloatingPointError): self.project([float('nan'),3],[1,0])

    def test_none_gradient_preserved_at_zero(self):
        p, q = [torch.nn.Parameter(torch.zeros(1)) for _ in range(2)]
        p.grad = torch.tensor([-1.])
        soft.project_parameter_grads_([p,q],[torch.ones(1),torch.ones(1)],kappa=0)
        self.assertIsNone(q.grad)
        self.assertEqual(p.grad.item(), -1.)

    def test_endpoint_one_bitwise_original_on_forced_conflicts(self):
        torch.manual_seed(81)
        for dtype in (torch.float32, torch.float64):
            for _ in range(20):
                r = torch.randn(20,dtype=dtype)
                g = -r + .1*torch.randn_like(r)
                a, b = [torch.nn.Parameter(torch.zeros_like(g)) for _ in range(2)]
                a.grad = g.clone(); b.grad = g.clone()
                ma = hard.project_parameter_grads_([a],[r])
                mb = soft.project_parameter_grads_([b],[r],kappa=1)
                equal(a.grad,b.grad)
                for key in ma: equal(ma[key],mb[key])
                self.assertTrue(mb['projected'])

    def test_linear_mixed_loss_no_double_replay(self):
        # Three new gradients -6, one old gradient +2 => mixed -4.
        # Half correction yields -2. A second replay addition would be wrong.
        p = torch.nn.Parameter(torch.tensor(0.))
        losses = torch.stack([-6*p,-6*p,-6*p,2*p])
        mix, _, ref, nn, no = soft.mixed_losses(losses,[7,7,7,6],7)
        soft.backward_mixed_with_reference(mix,ref,[p],kappa=.5)
        self.assertEqual(p.grad.item(), -2.)
        self.assertEqual((nn,no),(3,1))

    def test_no_replay_or_only_replay(self):
        for ids in ([7,7],[6,6]):
            p = torch.nn.Parameter(torch.tensor(1.))
            mix, _, ref, _, _ = soft.mixed_losses(torch.stack([p*p,2*p*p]),ids,7)
            m=soft.backward_mixed_with_reference(mix,ref,[p],loss_scale=2,kappa=.5)
            self.assertEqual(p.grad.item(),6.); self.assertFalse(m['projected'])


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = torch.nn.Parameter(torch.tensor(3.),requires_grad=False)
        self.temporal_transformer=torch.nn.Sequential(torch.nn.Linear(4,6),torch.nn.Dropout(.25))
        self.policy_head=torch.nn.Linear(6,10)
        self.policy_head.unused=torch.nn.Parameter(torch.tensor(.5))

    def compute_loss(self,data,*args,reduction='mean'):
        z=self.temporal_transformer(data['feature']); out=self.policy_head(z)
        mu=out[:,:4].reshape(-1,2,2).tanh()
        sd=torch.nn.functional.softplus(out[:,4:8].reshape(-1,2,2))+1e-4
        dist=torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=out[:,8:]),
            torch.distributions.Independent(torch.distributions.Normal(mu,sd),1))
        v=-dist.log_prob(data['actions'])
        return (v.mean() if reduction=='mean' else v),z


class IntegrationTests(unittest.TestCase):
    def test_gmm_adamw_rng_endpoints_multiple_updates(self):
        for kappa in (0,1):
            torch.manual_seed(91)
            models=[TinyPolicy()]; models.append(copy.deepcopy(models[0]))
            opts=[torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=.1) for m in models]
            sched=[torch.optim.lr_scheduler.LambdaLR(o,lambda s:min((s+1)/4.,1)) for o in opts]
            for ids in ([7]*6,[6,7]*3,[6]*6,[6,6,7,7,7,7]):
                data={'feature':torch.randn(6,4),'actions':torch.randn(6,2)}
                state=torch.get_rng_state()
                for index in range(2):
                    torch.set_rng_state(state); opts[index].zero_grad()
                    model=models[index]
                    if index==0 and kappa==0:
                        loss,_=model.compute_loss(data); (1.7*loss).backward()
                    else:
                        v,_=model.compute_loss(data,reduction='none')
                        mix,_,ref,_,_=soft.mixed_losses(v,ids,7)
                        core=hard if index==0 else soft
                        kw={} if index==0 else {'kappa':kappa}
                        core.backward_mixed_with_reference(mix,ref,[p for p in model.parameters() if p.requires_grad],loss_scale=1.7,**kw)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),.4)
                    opts[index].step(); sched[index].step()
                    if index==0: rng=torch.get_rng_state()
                    else: equal(rng,torch.get_rng_state())
                equal(models[0].state_dict(),models[1].state_dict())
                equal(opts[0].state_dict(),opts[1].state_dict()); equal(sched[0].state_dict(),sched[1].state_dict())
                for a,b in zip(models[0].parameters(),models[1].parameters()): equal(a.grad,b.grad)

    def test_real_adapter_logs_fixed_coefficient_and_rejects_change(self):
        fake=ModuleType('libero.lifelong.algos.mlr')
        class MLR(torch.nn.Module):
            def __init__(self,n_tasks,cfg,**kwargs):
                super().__init__(); self.cfg=cfg; self.experiment_dir=cfg.experiment_dir; self.loss_scale=1.
                self.policy=TinyPolicy(); self.optimizer=torch.optim.AdamW(self.policy.parameters(),lr=.001)
            def map_tensor_to_device(self,data): return data
            def optimizer_step_and_maybe_schedule(self): self.optimizer.step()
        fake.MLR=MLR
        keys={'libero.lifelong.algos.mlr':fake,'libero.lifelong.algos.fixed_kappa':soft}
        saved={k:sys.modules.get(k) for k in keys}; sys.modules.update(keys)
        try:
            adapter=load('fixed_adapter',ROOT/'src/mlr_fixed_kappa.py')
            with tempfile.TemporaryDirectory() as tmp:
                cfg=NS(ifa_enabled=False,seed=42,experiment_dir=tmp,train=NS(grad_clip=100.),
                    lifelong=NS(kappa=.5,projection_enabled=True,reference_norm_floor=1e-12,
                                synchronize_timing=False,step_log_every=1))
                algo=adapter.MLRFixedKappa(1,cfg)
                batch=[{'feature':torch.randn(4,4),'actions':torch.randn(4,2)},['6','7','6','7']]
                algo.observe(copy.deepcopy(batch),7,0); algo._flush_epoch()
                line=json.loads((Path(tmp)/'fixed_kappa_steps.jsonl').read_text())
                self.assertEqual(line['kappa'],.5)
                self.assertTrue((Path(tmp)/'fixed_kappa_epochs.csv').is_file())
                cfg.lifelong.kappa=.25
                with self.assertRaises(ValueError): algo.observe(batch,7,1)
        finally:
            for k,v in saved.items():
                if v is None: sys.modules.pop(k,None)
                else: sys.modules[k]=v


class ProtocolTests(unittest.TestCase):
    def test_launcher_isolated_full_protocol_and_eval_order(self):
        from run import commands, configuration
        a=NS(repo=Path('/repo'),tag='fixed_v1',kappa=.5,seed=43,smoke=False,feature_root=None,
             data_root=None,checkpoint=None,python='python',action='train',scheduler_mode='batch_step')
        cmd=commands(a)
        for flag in ('lifelong.kappa=0.5','lifelong=mlr_fixed_kappa','+start_task=6','+end_task=10',
                     'train.n_epochs=100','+past_data_rate=0.5','+ifa_enabled=false'):
            self.assertIn(flag,cmd)
        self.assertIn('fixed_kappa/fixed_v1/formal/kappa_0p5/seed43',configuration(a)[1].as_posix())
        a.action='eval'; cmd=commands(a)
        for flag in ('+eval_epochs=[99]','eval.n_eval=20','eval.num_procs=20','+aug=true'):
            self.assertIn(flag,cmd)
        a.kappa=float('nan')
        with self.assertRaises(ValueError): commands(a)

    def test_additive_installer_idempotence_and_conflict(self):
        from install import plan
        snippets={'libero/lifelong/algos/mlr.py':'class MLR: pass\n# build_dataset_for_task save_final_checkpoint',
            'libero/lifelong/algos/base.py':'def observe(): pass\n# optimizer_step_and_maybe_schedule batch_step',
            'libero/lifelong/models/base_policy.py':'def compute_loss(reduction=None): pass',
            'libero/lifelong/models/brief.py':'class BriefPolicy: pass\n# temporal_transformer policy_head',
            'libero/lifelong/models/policy_head.py':'# reduction == "none"',
            'libero/lifelong/algos/__init__.py':'# get_algo_class', 'libero/lifelong/test.py':'# eval_epochs'}
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d)
            for name,value in snippets.items():
                p=repo/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(value)
            changes=plan(repo); self.assertEqual(len(changes),4)
            for path,value in changes: path.parent.mkdir(parents=True,exist_ok=True); path.write_text(value,encoding='utf-8')
            self.assertEqual(plan(repo),[])
            (repo/'libero/lifelong/algos/fixed_kappa.py').write_text('# user changes')
            with self.assertRaises(ValueError): plan(repo)

    def test_matrix_rejects_missing_duplicate_or_wrong_epoch(self):
        from metrics import read_fixed_matrix
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'eval.log'
            lines=['[task %d epoch 99 and %d succ.] 0.5' % (t,s) for s in range(6,10) for t in range(6,s+1)]
            p.write_text('\n'.join(lines)); self.assertEqual(read_fixed_matrix(p)[3][3],50.)
            for bad in (lines[:-1],lines+[lines[0]],[x.replace('epoch 99','epoch 98') for x in lines]):
                p.write_text('\n'.join(bad))
                with self.assertRaises(ValueError): read_fixed_matrix(p)


class SummaryTests(unittest.TestCase):
    """Synthetic files live only in a temporary directory, never in experiment reports."""
    def fixture(self, repo):
        from protocol import common_recipe, endpoint_dirs, sha
        checkpoint=repo/'initial.pth'; checkpoint.write_bytes(b'synthetic fixture, not a model')
        for seed in (42,43,44):
            cfg={'seed':seed,'benchmark_name':'LIBERO_OBJECT','start_task':6,'end_task':10,
                 'aug':False,'past_data_rate':.5,'ifa_enabled':False,'repro_scheduler_mode':'batch_step',
                 'save_final_checkpoint':True,'load_previous_model':False,'task_embedding_format':'clip',
                 'data':{'seq_len':8,'demo_limit':10},'policy':{},
                 'train':{'n_epochs':100,'batch_size':10,'num_workers':0,'grad_clip':100.,'loss_scale':1.,
                          'resume':False,'optimizer':{'name':'torch.optim.AdamW','kwargs':{
                              'lr':1e-4,'betas':[.9,.999],'eps':1e-8,'weight_decay':.1}}},
                 'lifelong':{'algo':'MLRFixedKappa','projection_enabled':True,'kappa':.5,'reference_norm_floor':1e-12}}
            soft_dir=repo/'runs/fixed_kappa/fixed_v1/formal/kappa_0p5'/('seed%d'%seed)
            refs={}
            for arm,d in dict(endpoint_dirs(repo,seed),soft=soft_dir).items():
                (d/'run_001').mkdir(parents=True)
                (d/'run_001/config.json').write_text(json.dumps(cfg))
                success={'mlr':.50,'hard':.45,'soft':.55}[arm]
                log=d/'primary_ep99.log'
                log.write_text('\n'.join('[task %d epoch 99 and %d succ.] %s'%(t,s,success)
                                         for s in range(6,10) for t in range(6,s+1)))
                if arm!='soft': refs[arm]={'log_sha256':sha(log),'config_sha256':sha(d/'run_001/config.json')}
            manifest={'kappa':.5,'smoke':False,'seed':seed,'checkpoint':str(checkpoint),'source_sha256':{},
                'references':{'initial_checkpoint_sha256':sha(checkpoint),'common_recipe':common_recipe(cfg),'endpoints':refs}}
            (soft_dir/'experiment.json').write_text(json.dumps(manifest))
            for name in ('TRAINING_COMPLETE','PRIMARY_EVAL_COMPLETE'): (soft_dir/name).write_text('test fixture')
            for s in range(6,10): (soft_dir/'run_001'/('mlrifa_%d_model_ep99.pth'%s)).write_bytes(b'test fixture')
            (soft_dir/'train_execution.json').write_text('{"wall_seconds": 1}')
            (soft_dir/'evaluation_protocol.json').write_text('{}')
            (soft_dir/'evaluation_complete.json').write_text(json.dumps({
                'log_sha256':sha(soft_dir/'primary_ep99.log'),'protocol_sha256':sha(soft_dir/'evaluation_protocol.json')}))
            with (soft_dir/'run_001/fixed_kappa_epochs.csv').open('w',newline='') as f:
                template={'stage':6,'epoch':0,'kappa':.5,'enabled':True,'batches':1,'reference_batches':1,
                    'valid_reference_batches':1,'projected_batches':0,'mean_correction_ratio':0,
                    'mean_projected_correction_ratio':'','observe_seconds':.01,'stage_peak_allocated_bytes':0}
                w=csv.DictWriter(f,fieldnames=list(template)); w.writeheader()
                for s in range(6,10):
                    for e in range(100): w.writerow(dict(template,stage=s,epoch=e))
        return repo/'runs/fixed_kappa/fixed_v1/formal/kappa_0p5/seed42'

    @patch('validate_run.source_hashes',return_value={})
    def test_complete_aggregation_and_pairing(self,_):
        from summarize import summarize
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d); self.fixture(repo)
            r=summarize(repo,'fixed_v1',.5,repo/'output')
            self.assertAlmostEqual(r['paired']['soft_minus_mlr']['fwt']['mean'],5.)
            self.assertAlmostEqual(r['paired']['soft_minus_hard']['auc']['mean'],10.)
            self.assertEqual(r['aggregate']['soft']['fwt']['n'],3)

    @patch('validate_run.source_hashes',return_value={})
    def test_missing_seed_refused(self,_):
        from summarize import summarize
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d); run=self.fixture(repo)
            (run.parent/'seed44/TRAINING_COMPLETE').unlink()
            with self.assertRaises(ValueError): summarize(repo,'fixed_v1',.5,repo/'output')

    @patch('validate_run.source_hashes',return_value={})
    def test_changed_coefficient_or_log_refused(self,_):
        from validate_run import validate
        from summarize import summarize
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d); run=self.fixture(repo)
            with self.assertRaises(ValueError): validate(run,kappa=1)
            with (run/'primary_ep99.log').open('a') as f: f.write('\nchanged')
            with self.assertRaises(ValueError): summarize(repo,'fixed_v1',.5,repo/'output')


if __name__=='__main__': unittest.main()
