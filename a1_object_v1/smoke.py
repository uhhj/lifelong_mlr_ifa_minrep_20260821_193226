"""Technical checks only; isolated buffers and short disposable updates."""
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from easydict import EasyDict
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from .train import build
from .prepare import ROOT,OUT,config,code_hashes
from .state import capture,restore,cpu,rng_state,restore_rng
from .references import ensure_stage
from .io import write,save,sha
from .dataset import identified_dataset
from . import tests
from libero.lifelong.algos.mlr import MLR,update_buffer_equal_quota_10,build_ifa_similarity_matrix
from libero.lifelong.utils import torch_save_model,torch_load_model


def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    assert result.wasSuccessful()
    dist.init_process_group('nccl',init_method='env://');torch.cuda.set_device(0)
    checks={};details={'unit_tests':result.testsRun}
    try:
        with tempfile.TemporaryDirectory(prefix='a1_smoke_',dir=str(ROOT/'runs')) as tmp:
            cfg=EasyDict(config(42,tmp));cfg.a1_report=str(Path(tmp)/'reports');cfg.a1_protocol=str(Path(tmp)/'PROTOCOL.json')
            write(cfg.a1_protocol,dict(implementation_files=code_hashes()))
            algo,benchmark=build(cfg);model=algo.policy.module
            update_buffer_equal_quota_10(cfg.feature_dir,6,cfg.buffer_dir,seed=42,rate=.5)
            ds=identified_dataset(cfg.feature_dir,7,cfg.buffer_dir,42,start=6)
            ids=[i for i,p in enumerate(ds.file_paths) if Path(p).name.startswith('t6__')][:5]
            ids += [i for i,p in enumerate(ds.file_paths) if not Path(p).name.startswith('t')][:5]
            batch=lambda:default_collate([ds[i] for i in ids])
            info=build_ifa_similarity_matrix(7,cfg.feature_dir,cfg.buffer_dir)
            algo.a1_stage=7;algo.start_task(-1,DataLoader(ds,batch_size=10));algo.policy.train()
            before=capture(algo)
            ensure_stage(algo,6);ensure_stage(algo,7)
            checks['reference_preparation_preserves_all_training_state']=equal(before,capture(algo))
            ufile=Path(tmp)/'subspaces/SUBSPACE_stage7.pt';h=sha(ufile)
            U=algo.a1_U.clone();ensure_stage(algo,7)
            checks['same_stage_load_no_reorthogonalization']=h==sha(ufile) and torch.equal(U,algo.a1_U)
            checks['only_arrived_reference_ids']=algo.a1_subspace_state['task_ids']==list(range(8))
            # A few disposable updates initialize native moments; no experiment start/checkpoint is used.
            for _ in range(2):MLR.observe(algo,batch(),7,0,info)
            start=capture(algo)
            loss0=MLR.observe(algo,batch(),7,0,info);state0=capture(algo)
            grads0=[p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()]
            restore(algo,start);cfg.a1_enabled=False
            loss1=algo.observe(batch(),7,0,info)
            checks['a1_disabled_corrected_native_exact_state']=loss0==loss1 and equal(state0,capture(algo))
            checks['a1_disabled_exact_gradients']=all(equal(g,p.grad) for g,p in zip(grads0,model.parameters()))
            checks['corrected_own_reference_enabled']=cfg.ifa_own_reference_fix is True
            restore(algo,start);cfg.a1_enabled=True;algo.a1_U=torch.eye(512,device='cuda')
            lossI=algo.observe(batch(),7,0,info)
            checks['identity_projection_native_state']=lossI==loss0 and equal(state0,capture(algo))
            restore(algo,start);algo.a1_U=U
            # Identical forward and BC head outputs; projection is downstream of compute_loss.
            data=algo.map_tensor_to_device(batch()[0]);rng=rng_state()
            l,z=model.compute_loss(data,None,None);d0=model.policy_head(z)
            restore_rng(rng);l2,z2=model.compute_loss(data,None,None);d1=model.policy_head(z2)
            checks['bc_and_action_distribution_unchanged']=torch.equal(l,l2) and torch.equal(z,z2) and torch.equal(d0.mean,d1.mean)
            restore(algo,start);before_params=[(n,p.shape,p.requires_grad) for n,p in model.named_parameters()]
            lossA=algo.observe(batch(),7,0,info);after=capture(algo)
            checks['projected_step_finite']=np.isfinite(lossA) and all(torch.isfinite(p).all().item() for p in model.parameters())
            checks['single_optimizer_scheduler_step']=after['optimizer_step']==start['optimizer_step']+1 and after['scheduler_step']==start['scheduler_step']+1
            checks['parameter_set_unchanged']=before_params==[(n,p.shape,p.requires_grad) for n,p in model.named_parameters()]
            checks['rng_sequence_unchanged']=equal(state0['rng'],after['rng'])
            save(Path(tmp)/'state.pt',dict(state=after,subspace=algo.a1_subspace_state))
            saved=torch.load(Path(tmp)/'state.pt',map_location='cpu');restore(algo,saved['state'])
            checks['full_state_reload_exact']=equal(saved['state'],capture(algo)) and equal(saved['subspace'],algo.a1_subspace_state)
            details.update(native_loss=loss0,projected_loss=lossA,stage7_rank=U.shape[1],actual_geometry=algo._a1_diag,
                subspace_errors=algo.a1_subspace_state['metadata']['errors'])
            checks['subspace_not_policy_state']=not any('subspace' in k or 'a1_U' in k for k in model.state_dict())
            # Parameter-scope failure evidence must use PRE-update model/moments/RNG.
            original=model.compute_loss
            def nan_loss(*args,**kwargs):
                l,z=original(*args,**kwargs);return l*torch.tensor(float('nan'),device=l.device),z
            start=capture(algo);model.compute_loss=nan_loss
            try:
                algo.observe(batch(),7,1,info)
                raise AssertionError('Expected fail-stop')
            except FloatingPointError:
                dumps=list(Path(cfg.a1_report).glob('FAILED*.pt'));assert len(dumps)==1
                dump=torch.load(dumps[0],map_location='cpu')
                checks['nonfinite_pre_update_dump']=equal(dump['pre_update_state'],start) and 'batch' in dump and 'subspace' in dump
            finally:model.compute_loss=original;restore(algo,start)
            # Exercise the actual epoch recovery constructor and optimizer bootstrap.
            algo.checkpoint(1)
            restored,bench2=build(cfg);restored.a1_stage=7
            ensure_stage(restored,7);restored.start_task(-1,DataLoader(ds,batch_size=10))
            checks['actual_recovery_bootstrap_exact']=equal(start,capture(restored))
            checks['actual_recovery_basis_exact']=torch.equal(restored.a1_U,U)
            # Use the canonical checkpoint format and native load path.
            path=Path(tmp)/'native_model.pth';torch_save_model(restored.policy,str(path),cfg=cfg)
            loaded,_,_=torch_load_model(str(path))
            checks['native_model_checkpoint_format']=equal(loaded,cpu(restored.policy.module.state_dict()))
            # Stage 6 naturally has no IFA pairs, including with A1 enabled.
            ds6=identified_dataset(cfg.feature_dir,6,cfg.buffer_dir,42,start=6)
            data6=default_collate([ds6[i] for i in range(10)])
            before6=capture(restored)
            native6=MLR.observe(restored,copy.deepcopy(data6),6,0,None);state6=capture(restored)
            restore(restored,before6)
            a16=restored.observe(copy.deepcopy(data6),6,0,None)
            checks['task6_native_noop_exact']=native6==a16 and equal(state6,capture(restored)) and not restored._a1_diag
    finally:
        dist.destroy_process_group()
    checks['geometry_unit_tests']=result.wasSuccessful()
    write(OUT/'ENGINEERING_CHECK.json',dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,details=details,source_files=code_hashes()))
    assert all(checks.values()),checks


if __name__=='__main__':main()
