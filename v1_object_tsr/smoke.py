"""Actual native MLR parity and source-routing checks before the full run."""
import ast,inspect,json,tempfile,textwrap,unittest
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
from .state import capture,restore,cpu,rng_state
from .dataset import identified_dataset
from .signatures import compute_signature,parameter_manifest,planned_cache,bc_eval
from .routing import RoutedSampler
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
    checks={};details={}
    try:
        with tempfile.TemporaryDirectory(prefix='tsr_smoke_',dir=str(ROOT/'runs')) as tmp:
            tmp=Path(tmp);cfg=EasyDict(config(42,tmp));cfg.tsr_report=str(tmp/'reports');cfg.tsr_protocol=str(tmp/'protocol.json')
            write(cfg.tsr_protocol,dict(implementation_files=code_hashes()))
            algo,benchmark=build(cfg);model=algo.policy.module
            manifest=parameter_manifest(model);write(OUT/'PARAMETER_MANIFEST.json',manifest)
            update_buffer_equal_quota_10(cfg.feature_dir,6,cfg.buffer_dir,seed=42,rate=.5)
            checks['planned_cache_exact_native_membership']={p.name for p in Path(cfg.buffer_dir).glob('*.pt')}=={'t6__'+p.name for p in planned_cache(cfg.feature_dir,6,42)}
            ds=identified_dataset(cfg.feature_dir,7,cfg.buffer_dir,42,start=6)
            info=build_ifa_similarity_matrix(7,cfg.feature_dir,cfg.buffer_dir)
            old=[i for i,p in enumerate(ds.file_paths) if Path(p).name.startswith('t6__')]
            new=[i for i,p in enumerate(ds.file_paths) if not Path(p).name.startswith('t')]
            ids=old[:5]+new[:5];data=lambda:default_collate([ds[i] for i in ids])
            algo.tsr_stage=7;algo.start_task(-1,DataLoader(ds,batch_size=10));algo.policy.train()
            for _ in range(2):MLR.observe(algo,data(),7,0,info)
            start=capture(algo)
            loss0=MLR.observe(algo,data(),7,0,info);state0=capture(algo)
            grads0=[None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
            restore(algo,start);loss1=algo.observe(data(),7,0,info)
            checks['exact_native_loss_model_optimizer_scheduler_rng']=loss0==loss1 and equal(state0,capture(algo))
            checks['exact_native_gradients']=all(equal(g,p.grad) for g,p in zip(grads0,model.parameters()))
            checks['one_optimizer_scheduler_step']=algo._optimizer_step_count==start['optimizer_step']+1 and algo._scheduler_step_count==start['scheduler_step']+1
            restore(algo,start);before=capture(algo)
            sig=compute_signature(algo,7)
            checks['signature_preserves_student_optimizer_scheduler_rng']=equal(before,capture(algo))
            sig2=compute_signature(algo,7)
            checks['signature_repeat_deterministic']=torch.equal(sig['mean_gradient'],sig2['mean_gradient']) and sig['metadata']['reference_model_sha256']==sig2['metadata']['reference_model_sha256']
            checks['signature_parameter_boundary']=manifest==sig['metadata']['parameter_manifest'] and all(r['name'].startswith(('temporal_transformer.decoder.h.5.','policy_head.')) for r in manifest['signature_parameters'])
            checks['signature_100_cache_only']=len(sig['metadata']['probe_files'])==100 and set(Path(p).name for p in sig['metadata']['probe_files'])<=set(sig['metadata']['planned_cache_files'])
            # Eval adapter equals the official GMM loss under identical deterministic settings.
            model.train()
            for m in model.modules():
                if isinstance(m,torch.nn.Dropout):m.eval()
            batch={k:v.cuda() for k,v in data()[0].items()}
            with torch.no_grad():official=model.compute_loss(batch,None,None)
            # compute_loss returns (loss,last_feature) in this project.
            if isinstance(official,tuple):official=official[0]
            model.eval()
            with torch.no_grad():adapter=bc_eval(model,batch)
            checks['signature_adapter_exact_official_bc']=torch.equal(official,adapter)
            restore(algo,start)
            save(tmp/'state.pt',start);restore(algo,torch.load(tmp/'state.pt',map_location='cpu'));algo.observe(data(),7,0,info)
            checks['legal_restore_reproduces_native_update']=equal(state0,capture(algo))
            checks['data_identity_order_content_parity']=all(equal(ds[i][:2],ds.original[i]) for i in ids)
            s=RoutedSampler(ds,7,42,{6:1.},epsilon=1)
            checks['epsilon1_calls_native_sampler']=all((s.set_epoch(e) is None and list(s)==list(s.native)) for e in [0,99])
            from .native_observe import NativeLossMixin
            checks['original_bc_ifa_ast']=normalized_prefix(MLR.observe)==normalized_prefix(NativeLossMixin.observe)
            from . import eval_metric
            from libero.lifelong import metric
            source='\n'.join(l for l in inspect.getsource(eval_metric.evaluate_one_task_success).splitlines() if 'record_trial(' not in l)
            checks['original_evaluation_ast']=ast.dump(ast.parse(textwrap.dedent(source)))==ast.dump(ast.parse(textwrap.dedent(inspect.getsource(metric.evaluate_one_task_success))))
            save(Path(cfg.experiment_dir)/'mlrifa_7_model_ep99.pth',dict(state_dict=algo.policy.state_dict(),cfg=cfg))
            from libero.lifelong.utils import torch_load_model
            loaded,_,_=torch_load_model(str(Path(cfg.experiment_dir)/'mlrifa_7_model_ep99.pth'))
            checks['eval_checkpoint_load']=equal(loaded,cpu(model.state_dict()))
            precommit=capture(algo);algo.commit_stage();cp=torch.load(cfg.tsr_recovery,map_location='cpu');algo.commit_stage()
            checks['stage_commit_idempotent_full_state']=equal(precommit,capture(algo)) and equal(cp['state'],capture(algo))
            counts={t:len(list((Path(cfg.feature_dir)/str(t)).glob('*.pt'))) for t in range(6,10)}
            budgets={str(t):(counts[t]+sum((counts[k]+1)//2 for k in range(6,t))+9)//10 for t in counts}
            checks['baseline_full_budget_preserved']=budgets=={'6':157,'7':227,'8':309,'9':362}
            details=dict(unit_tests=result.testsRun,ifa_pairs=info['selected_pairs'],signature_metadata=sig['metadata'],
                updates_per_epoch=budgets,source_files=code_hashes(),parity_tolerance='bitwise; no tolerance relaxation')
        write(OUT/'ENGINEERING_CHECK.json',dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,details=details))
        assert all(checks.values()),checks
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
