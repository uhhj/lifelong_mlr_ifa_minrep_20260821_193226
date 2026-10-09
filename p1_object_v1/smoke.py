"""Engineering-only real model checks; temporary states never enter P1 runs."""
import ast
import inspect
import json
import tempfile
import textwrap
import unittest
from pathlib import Path

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
from .policy_distill import forward_distribution,fields,sample_teacher,loss_from_samples
from .teacher_store import TeacherStore
from e1_object_v1.runtime import capture,restore
from e1_object_v1.smoke import equal
from libero.lifelong.algos.mlr import MLR,update_buffer_equal_quota_10,build_ifa_similarity_matrix


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    assert result.wasSuccessful()
    dist.init_process_group('nccl',init_method='env://');torch.cuda.set_device(0)
    checks={};details={}
    try:
        with tempfile.TemporaryDirectory(prefix='p1_engineering_',dir=str(ROOT/'runs')) as tmp:
            tmp=Path(tmp);cfg=EasyDict(config(42,tmp));cfg.p1_report=str(tmp/'reports');cfg.p1_protocol=str(tmp/'protocol.json')
            write(cfg.p1_protocol,dict(implementation_files=code_hashes()))
            algo,benchmark=build(cfg);model=algo.policy.module
            update_buffer_equal_quota_10(cfg.feature_dir,6,cfg.buffer_dir,seed=42,rate=.5)
            # Engineering fixture only: temporary model and checkpoint, NOT a formal teacher.
            ck=Path(cfg.experiment_dir)/'engineering_model.pt';save(ck,dict(state_dict=algo.policy.state_dict()))
            algo.store.build_stage(model,cfg.buffer_dir,6,ck,algo.report)
            ds=identified_dataset(cfg.feature_dir,7,cfg.buffer_dir,42,start=6)
            info=build_ifa_similarity_matrix(7,cfg.feature_dir,cfg.buffer_dir)
            old=[i for i,p in enumerate(ds.file_paths) if Path(p).name.startswith('t6__')]
            new=[i for i,p in enumerate(ds.file_paths) if not Path(p).name.startswith('t')]
            ids=old[:5]+new[:5];data=lambda:default_collate([ds[i] for i in ids])
            algo.p1_stage=7;algo.start_task(-1,DataLoader(ds,batch_size=10));algo.policy.train()
            for _ in range(2):MLR.observe(algo,data(),7,0,info)
            start=capture(algo);gen=algo.generator.get_state()
            MLR.observe(algo,data(),7,0,info);native=capture(algo)
            restore(algo,start);cfg.p1_beta=0.;algo.observe(data(),7,0,info)
            checks['beta_zero_exact_native_parity']=equal(native,capture(algo)) and torch.equal(gen,algo.generator.get_state())
            cfg.p1_beta=.05
            # No old windows uses original objective and does not consume distill RNG.
            newdata=lambda:default_collate([ds[i] for i in new[:10]])
            restore(algo,start);MLR.observe(algo,newdata(),7,0,info);newnative=capture(algo)
            restore(algo,start);algo.observe(newdata(),7,0,info)
            checks['no_old_exact_native_parity']=equal(newnative,capture(algo)) and torch.equal(gen,algo.generator.get_state())
            restore(algo,start)
            fdata=algo.map_tensor_to_device(default_collate([ds[i] for i in old[:5]])[0])
            before=capture(algo);student=forward_distribution(model,fdata);sfields=fields(student)
            teacher=algo.store.lookup([ds[i][2] for i in old[:5]],[6]*5)
            actions,lp=sample_teacher(teacher,algo.generator)
            loss,_=loss_from_samples(student,actions.cuda(),lp.cuda())
            params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
            grads=torch.autograd.grad(loss,[p for n,p in params],allow_unused=True)
            checks['student_decoder_and_head_gradients']=all(any(g is not None and bool(g.abs().sum()>0) for (n,p),g in zip(params,grads) if n.startswith(prefix)) for prefix in ['temporal_transformer.','policy_head.'])
            checks['teacher_actions_stop_gradient']=not actions.requires_grad and not lp.requires_grad and all(not t.requires_grad for t in teacher.values())
            checks['forward_modes_buffers_rng_unchanged']=equal(before,capture(algo))
            frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
            restore(algo,start);algo.generator.set_state(gen);algo.observe(data(),7,0,info);candidate=capture(algo);aftergen=algo.generator.get_state()
            checks['one_native_optimizer_scheduler_step']=candidate['optimizer_step']==start['optimizer_step']+1 and candidate['scheduler_step']==start['scheduler_step']+1
            checks['native_rng_unchanged_by_distill']=equal(native['rng'],candidate['rng'])
            checks['frozen_parameters_unchanged']=all(torch.equal(frozen[n],p) for n,p in model.named_parameters() if n in frozen)
            checks['parameter_boundary']=all(n.startswith(('temporal_transformer.','policy_head.')) for n,p in params)
            save(tmp/'state.pt',dict(state=start,distill_rng=gen))
            replay=torch.load(tmp/'state.pt',map_location='cpu');restore(algo,replay['state']);algo.generator.set_state(replay['distill_rng'])
            algo.observe(data(),7,0,info)
            checks['candidate_resume_exact_parity']=equal(candidate,capture(algo)) and torch.equal(aftergen,algo.generator.get_state())
            checks['batch_identity_content_parity']=all(equal(ds[i][:2],list(ds.original[i])) or equal(ds[i][:2],ds.original[i]) for i in ids)
            original_rows=json.loads(algo.store.manifest.read_text())['windows']
            checks['teacher_identity_errors_stop']=False
            errors=0
            for keys,tasks in [(['missing'],[6]),([ds[old[0]][2]],[7])]:
                try:algo.store.lookup(keys,tasks)
                except AssertionError:errors+=1
            mismatch=TeacherStore(tmp/'teacher_targets',43,algo.store.manifest)
            try:mismatch.lookup([ds[old[0]][2]],[6])
            except AssertionError:errors+=1
            checks['teacher_identity_errors_stop']=errors==3
            # Interrupted stage transaction: original selected buffer committed only
            # after teacher tables; retry must leave immutable tables unchanged.
            algo.p1_stage=7
            save(Path(cfg.experiment_dir)/'mlrifa_7_model_ep99.pth',dict(state_dict=algo.policy.state_dict()))
            precommit=capture(algo);pregen=algo.generator.get_state()
            algo.commit_stage()
            checks['teacher_build_no_training_state_mutation']=equal(precommit,capture(algo)) and torch.equal(pregen,algo.generator.get_state())
            old_preserved=all(algo.store.rows[k]==v for k,v in original_rows.items())
            snapshots={p.name:sha(p) for p in (tmp/'teacher_targets').glob('*.pt')}
            algo.commit_stage()
            checks['teacher_no_refresh_and_transaction_idempotent']=old_preserved and snapshots=={p.name:sha(p) for p in (tmp/'teacher_targets').glob('*.pt')}
            cp=torch.load(cfg.p1_recovery,map_location='cpu')
            checks['recovery_teacher_buffer_rng_complete']=cp['stage_complete'] and cp['teacher_manifest_sha256']==sha(algo.store.manifest) and equal(cp['state'],capture(algo)) and torch.equal(cp['distill_rng'],algo.generator.get_state())
            # Source equivalence excluding only the explicit insertion/update tail.
            from .native_observe import NativeLossMixin
            def native_prefix(method):
                tree=ast.parse(textwrap.dedent(inspect.getsource(method)))
                class Normalize(ast.NodeTransformer):
                    def visit_FunctionDef(self,n):
                        n.body=self.block(n.body);return n
                    def block(self,body):
                        result=[]
                        for n in body:
                            if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='print':continue
                            if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='base_bc' for x in n.targets):continue
                            if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='backward':break
                            if isinstance(n,ast.Return) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='finish_native':break
                            if isinstance(n,ast.If):n.body=self.block(n.body)
                            result.append(n)
                        return result
                return ast.dump(Normalize().visit(tree),include_attributes=False)
            checks['original_BC_IFA_construction_AST']=native_prefix(MLR.observe)==native_prefix(NativeLossMixin.observe)
            from .legacy_metrics import metrics
            old=json.loads((ROOT/'reports/e1_object_v1/PROTOCOL.json').read_text())['baseline_results']
            checks['baseline_unrounded_matrix_metrics']=all(metrics(old[str(s)]['r'])['auc']==old[str(s)]['auc'] for s in [42,43,44])
            details=dict(unit_tests=result.testsRun,trainable_parameters=sum(p.numel() for n,p in params),
                initial_engineering_teacher_count=len(original_rows),ifa_pairs=info['selected_pairs'],source_files=code_hashes())
        write(OUT/'ENGINEERING_CHECK.json',dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,details=details))
        assert all(checks.values()),checks
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
