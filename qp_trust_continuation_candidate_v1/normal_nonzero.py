"""Real-policy normal-path parity at nonzero LR, following one native warmup step."""
import contextlib,gc,io,traceback
from pathlib import Path
import torch
from . import step,core
from .io import write_json
from qp_lbfgsb_kkt_driven_v1.offline import OUT
from sample_qp_lifelong_v1_continuation_v3 import common as c
from sample_qp_lifelong_v1_continuation_v3.step import update as oldupdate

def main():
    from functional_geometry_diagnostic_v1 import geometry as g
    from cb_ifa_experiment.gpu_runtime import bootstrap_train,build_train
    assert c.read(OUT/'GPU_CHECK.json')['status']=='PASS'
    dest=Path('/root/qp_trust_continuation_candidate_v1/normal_nonzero');dest.mkdir(exist_ok=True)
    g.init_ddp(dest/'ddp');torch.set_num_threads(2);torch.set_num_interop_threads(1)
    try:
        cfg=c.config(dest/'bootstrap');bootstrap_train(cfg);algo=build_train(cfg);initial=g.cpu_state(algo.policy.state_dict());results=[]
        for task in (7,8,9):
            write_json(OUT/'STATUS.json',dict(stage='NORMAL_NONZERO_PARITY',task=task,formal_training_resumed=False))
            plan,ds,sampler,loader=c.dataset_loader(task);algo.policy.load_state_dict(initial)
            with contextlib.redirect_stdout(io.StringIO()):algo.start_task(-1,loader)
            algo.policy.train();refs=c.reference_batches(algo,task);order=list(iter(sampler))
            w=core.Workspace(g.params(algo));warm=oldupdate(algo,g.batch(ds,order[:10]),task,0,refs,w);del w
            base=g.capture(algo);data=g.batch(ds,order[10:20]);records={};states={}
            for arm,fn in [('v3',oldupdate),('candidate',step.update)]:
                g.restore(algo,base);w=core.Workspace(g.params(algo));records[arm]=fn(algo,data,task,0,refs,w);states[arm]=g.capture(algo);del w
            cmp=g.compare_states(states['v3'],states['candidate']);r=records['candidate']
            checks=dict(exact_state=all(cmp.values()),nonzero_lr=r['lr']>0,nonzero_native_displacement=r['native_norm']>0,standard_pass=r['kkt']<=1e-8,no_lbfgsb=not r['fallback_triggered'],no_trust=not r['second_fallback_triggered'])
            results.append(dict(task=task,count=32*(task-6),warmup=warm,record_indices=order[10:20],records=records,states=cmp,checks=checks,status='PASS' if all(checks.values()) else 'FAIL'))
            write_json(OUT/'NORMAL_NONZERO_GPU_PARITY.json',dict(status='RUNNING',results=results,formal_training_resumed=False));assert all(checks.values()),results[-1]
            print('NORMAL_NONZERO_PASS',task,r['native_norm'],r['kkt'],flush=True)
            del ds,loader,refs,base,states,data;gc.collect();torch.cuda.empty_cache()
        write_json(OUT/'NORMAL_NONZERO_GPU_PARITY.json',dict(status='PASS',results=results,formal_training_resumed=False))
    finally:
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
if __name__=='__main__':
    try:main()
    except BaseException:
        write_json(OUT/'NORMAL_NONZERO_ERROR.json',dict(traceback=traceback.format_exc(),formal_training_resumed=False));raise
