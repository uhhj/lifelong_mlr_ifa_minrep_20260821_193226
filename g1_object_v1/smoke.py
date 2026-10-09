"""Small real-batch correctness checks, never a baseline training run."""
import ast
import inspect
import json
import tempfile
import textwrap
import unittest
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from easydict import EasyDict
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from . import tests
from .train import build
from .prepare import ROOT,OUT,config,code_hashes
from .io import write,save,sha
from .dataset import identified_dataset
from .state import capture,restore,cpu
from libero.lifelong.algos.mlr import MLR,update_buffer_equal_quota_10,build_ifa_similarity_matrix


def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def normalized_prefix(method):
    tree=ast.parse(textwrap.dedent(inspect.getsource(method)))
    class N(ast.NodeTransformer):
        def visit_FunctionDef(self,n):n.body=self.block(n.body);return n
        def block(self,body):
            out=[]
            for n in body:
                if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('bc_loss','aux_terms','weighted_term') for t in n.targets):continue
                if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call):
                    f=n.value.func
                    if isinstance(f,ast.Name) and f.id=='print':continue
                    if isinstance(f,ast.Attribute) and f.attr=='append' and isinstance(f.value,ast.Name) and f.value.id=='aux_terms':continue
                    if isinstance(f,ast.Attribute) and f.attr=='backward':break
                if isinstance(n,ast.Return) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='finish_native':break
                if isinstance(n,ast.If):n.body=self.block(n.body);n.orelse=self.block(n.orelse)
                if isinstance(n,ast.For):n.body=self.block(n.body)
                class Substitute(ast.NodeTransformer):
                    def visit_Name(self,x):
                        if x.id=='weighted_term':return ast.parse('ifa_weight * loss_hinge_angle.mean()',mode='eval').body
                        return x
                out.append(Substitute().visit(n))
            return out
    return ast.dump(N().visit(tree),include_attributes=False)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    assert result.wasSuccessful()
    dist.init_process_group('nccl',init_method='env://');torch.cuda.set_device(0)
    assert dist.get_world_size()==1
    checks={};details={}
    try:
        with tempfile.TemporaryDirectory(prefix='g1_engineering_',dir=str(ROOT/'runs')) as tmp:
            tmp=Path(tmp);cfg=EasyDict(config(42,tmp));cfg.g1_report=str(tmp/'reports');cfg.g1_protocol=str(tmp/'protocol.json')
            write(cfg.g1_protocol,dict(implementation_files=code_hashes()))
            algo,benchmark=build(cfg);model=algo.policy.module
            named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
            decoder=[('temporal_transformer.'+n,p) for n,p in model.temporal_transformer.named_parameters() if p.requires_grad]
            write(OUT/'PARAMETER_MANIFEST.json',dict(gate_module='temporal_transformer',
                gate_parameters=[dict(name=n,shape=list(p.shape),count=p.numel()) for n,p in decoder],
                trainable_parameters=[dict(name=n,shape=list(p.shape),count=p.numel()) for n,p in named],
                gate_count=sum(p.numel() for n,p in decoder),trainable_count=sum(p.numel() for n,p in named)))
            update_buffer_equal_quota_10(cfg.feature_dir,6,cfg.buffer_dir,seed=42,rate=.5)
            ds=identified_dataset(cfg.feature_dir,7,cfg.buffer_dir,42,start=6)
            info=build_ifa_similarity_matrix(7,cfg.feature_dir,cfg.buffer_dir)
            old=[i for i,p in enumerate(ds.file_paths) if Path(p).name.startswith('t6__')]
            new=[i for i,p in enumerate(ds.file_paths) if not Path(p).name.startswith('t')]
            ids=old[:5]+new[:5];data=lambda:default_collate([ds[i] for i in ids])
            algo.g1_stage=7;algo.start_task(-1,DataLoader(ds,batch_size=10));algo.policy.train()
            for _ in range(2):MLR.observe(algo,data(),7,0,info)
            start=capture(algo)
            native_loss=MLR.observe(algo,data(),7,0,info);native=capture(algo)
            restore(algo,start);cfg.g1_enabled=False;loss=algo.observe(data(),7,0,info)
            checks['disabled_exact_native']=loss==native_loss and equal(native,capture(algo));cfg.g1_enabled=True
            restore(algo,start);algo.smoke_force='open';loss=algo.observe(data(),7,0,info)
            checks['forced_open_with_probes_exact_native']=loss==native_loss and equal(native,capture(algo)) and algo.last_row['dot'] is not None
            restore(algo,start);bc_loss=MLR.observe(algo,data(),7,0,None);bconly=capture(algo)
            restore(algo,start);algo.smoke_force='closed';loss=algo.observe(data(),7,0,info)
            checks['forced_closed_exact_BC_only']=loss==bc_loss and equal(bconly,capture(algo))
            del algo.smoke_force
            restore(algo,start);MLR.observe(algo,data(),7,0,None);noifa=capture(algo)
            restore(algo,start);algo.observe(data(),7,0,None)
            checks['no_IFA_no_probe_no_stale_gate']=equal(noifa,capture(algo)) and algo.last_row['dot'] is None and algo.last_row['w'] is None
            restore(algo,start)
            frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
            counter=[0];original=model.compute_loss
            def counted(*a,**kw):counter[0]+=1;return original(*a,**kw)
            model.compute_loss=counted
            import g1_object_v1.gate as gate
            real_grad=gate.torch.autograd.grad;probes=[]
            def checked_grad(*a,**kw):
                before=[None if p.grad is None else p.grad.clone() for n,p in named]
                state=capture(algo)
                value=real_grad(*a,**kw)
                probes.append(all(equal(g,p.grad) for g,(n,p) in zip(before,named)) and equal(state,capture(algo)))
                return value
            gate.torch.autograd.grad=checked_grad
            try:algo.observe(data(),7,0,info)
            finally:gate.torch.autograd.grad=real_grad;model.compute_loss=original
            candidate=capture(algo)
            checks['one_forward_shared_graph']=counter[0]==1 and len(probes)==2
            checks['probes_preserve_grad_model_optimizer_scheduler_RNG']=all(probes)
            checks['one_optimizer_scheduler_step']=candidate['optimizer_step']==start['optimizer_step']+1 and candidate['scheduler_step']==start['scheduler_step']+1
            checks['native_RNG_parity']=equal(native['rng'],candidate['rng'])
            checks['frozen_parameters_unchanged']=all(torch.equal(v,dict(model.named_parameters())[n]) for n,v in frozen.items())
            checks['parameter_boundary']=all(n.startswith(('temporal_transformer.','policy_head.')) for n,p in named) and len(decoder)>0
            save(tmp/'state.pt',start);restore(algo,torch.load(tmp/'state.pt',map_location='cpu'));algo.observe(data(),7,0,info)
            checks['full_restore_candidate_exact_parity']=equal(candidate,capture(algo))
            checks['data_identity_order_content_parity']=all(equal(ds[i][:2],ds.original[i]) for i in ids)
            from .native_observe import NativeLossMixin
            checks['original_BC_IFA_construction_AST']=normalized_prefix(MLR.observe)==normalized_prefix(NativeLossMixin.observe)
            from . import eval_metric
            from libero.lifelong import metric
            source=inspect.getsource(eval_metric.evaluate_one_task_success)
            source='\n'.join(l for l in source.splitlines() if 'record_trial(' not in l)
            checks['original_eval_scalar_source']=ast.dump(ast.parse(textwrap.dedent(source)))==ast.dump(ast.parse(textwrap.dedent(inspect.getsource(metric.evaluate_one_task_success))))
            save(Path(cfg.experiment_dir)/'mlrifa_7_model_ep99.pth',dict(state_dict=algo.policy.state_dict(),cfg=cfg))
            from libero.lifelong.utils import torch_load_model
            loaded,_,_=torch_load_model(str(Path(cfg.experiment_dir)/'mlrifa_7_model_ep99.pth'))
            # Native loader strips DDP's module. prefix; evaluate underlying
            # policy keys exactly as the original evaluation loader does.
            checks['eval_actual_checkpoint_load']=equal(loaded,cpu(model.state_dict()))
            precommit=capture(algo);algo.commit_stage();cp=torch.load(cfg.g1_recovery,map_location='cpu')
            files=sorted(p.name for p in Path(cfg.buffer_dir).glob('*.pt'));algo.commit_stage()
            checks['stage_buffer_commit_idempotent']=files==sorted(p.name for p in Path(cfg.buffer_dir).glob('*.pt')) and equal(precommit,capture(algo))
            checks['recovery_full_state']=cp['stage_complete'] and equal(cp['state'],capture(algo)) and cp['buffer_files']==files
            mem=[]
            for _ in range(4):
                algo.observe(data(),7,0,info);torch.cuda.synchronize();mem.append(torch.cuda.memory_allocated())
            checks['no_cross_batch_graph_retention']=len(set(mem[1:]))==1 and all(not torch.is_tensor(v) for v in algo.last_row.values())
            details=dict(unit_tests=result.testsRun,ifa_pairs=info['selected_pairs'],memory_bytes=mem,source_files=code_hashes(),tolerance='bitwise state parity; no relaxed tolerance')
        write(OUT/'ENGINEERING_CHECK.json',dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,details=details))
        assert all(checks.values()),checks
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
