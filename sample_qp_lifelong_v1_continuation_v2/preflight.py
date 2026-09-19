import contextlib
import gc
import io
import time
import traceback
from .common import *

def reference_update(algo,data,task,epoch,refs):
    """Slow independent reference extends only row count, using original projection math."""
    from sample_step_closedloop_v1 import train as legacy
    g=legacy.g; torch=g.torch
    before=g.capture(algo); modes={n:m.training for n,m in algo.policy.named_modules()}
    columns=[]; algo.policy.eval(); ps=g.params(algo)
    try:
        for batch in refs:
            losses=g.module(algo).policy_head.loss_fn(g.gmm(algo,batch),batch['actions'].squeeze(1),'none')
            for i in range(8):
                grads=torch.autograd.grad(losses[i],ps,retain_graph=i<7,allow_unused=True)
                columns.append(torch.cat([(torch.zeros_like(p) if z is None else z).reshape(-1)
                                          for p,z in zip(ps,grads)]))
        H=torch.stack(columns,dim=1) if columns else None
    finally:
        for n,m in algo.policy.named_modules():m.training=modes[n]
    del columns
    K=legacy.inner(H,gram=True)[0] if H is not None else None
    g.native(algo,data,task,epoch)
    if H is not None:
        native=g.capture(algo); d=g.displacement(algo,before).cuda()
        c,sol=legacy.sample_project(H,K,d,independent=False)
        g.assign_displacement(algo,native,-c)
        after=g.capture(algo)
        assert all(g.digest(after[k])==g.digest(native[k]) for k in native if k!='model')

def main():
    from functional_geometry_diagnostic_v1 import geometry as g
    from cb_ifa_experiment.gpu_runtime import bootstrap_train,build_train,atomic_save
    from .step import update,Workspace
    import torch
    assert not (OUT/'PREFLIGHT_COMPLETE.json').exists()
    protocol=verify(); torch.set_num_threads(2);torch.set_num_interop_threads(1)
    assert torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3090'
    dest=OUT/'preflight';dest.mkdir(exist_ok=True);g.init_ddp(dest/'ddp')
    guard=Guard(); results=[]
    try:
        cfg=config(dest); bootstrap_train(cfg); algo=build_train(cfg)
        write(OUT/'INITIALIZATION_EXPECTED.json',dict(model_hash=g.digest(algo.policy.state_dict()),
              rng_hash=g.digest(g.rng_state()),official_sha256=sha(cfg.pretrain)))
        initial=g.cpu_state(algo.policy.state_dict())
        for task in TASKS:
            write(OUT/'STATUS.json',dict(status='PREFLIGHT',task=task,time=time.time()))
            plan,ds,sampler,loader=dataset_loader(task)
            algo.policy.load_state_dict(initial)
            with contextlib.redirect_stdout(io.StringIO()):algo.start_task(-1,loader)
            algo.policy.train(); base=g.capture(algo); refs=reference_batches(algo,task)
            assert g.digest(base['rng'])==g.digest(g.rng_state())
            w=Workspace(g.params(algo)); states={}; times={}
            indices=list(iter(sampler)); batches=[indices[i*10:(i+1)*10] for i in range(5)]
            for arm in ('reference','optimized'):
                g.restore(algo,base); states[arm]=[]; times[arm]=[]
                for i,ix in enumerate(batches):
                    tick=time.perf_counter()
                    if arm=='reference':reference_update(algo,g.batch(ds,ix),task,0,refs)
                    else:update(algo,g.batch(ds,ix),task,0,refs,w,guard)
                    torch.cuda.synchronize();times[arm].append(time.perf_counter()-tick)
                    s=g.capture(algo);states[arm].append({k:g.digest(v) for k,v in s.items()})
                    if arm=='optimized' and i==0:
                        write(dest/('logging_task%d.json'%task),dict(task=task,step=i,state_hashes=states[arm][-1]))
                        assert all(g.compare_states(s,g.capture(algo)).values()),'Logging changed state'
            assert states['reference']==states['optimized'], ('Full-stage GPU parity',task)
            # Serialize at a real epoch boundary, then reproduce iterator RNG consumption.
            boundary=g.capture(algo); atomic_save(dest/'boundary.pt',boundary)
            def continuation():
                sampler.set_epoch(1);algo.policy.train()
                data=next(iter(loader))
                rec=update(algo,data,task,1,refs,w,guard)
                return g.capture(algo),rec
            first,rec=continuation()
            g.restore(algo,torch.load(dest/'boundary.pt',map_location='cpu'))
            assert all(g.compare_states(boundary,g.capture(algo)).values())
            second,rec2=continuation()
            assert all(g.compare_states(first,second).values()),('Epoch iterator recovery',task)
            results.append(dict(task=task,references=32*(task-6),paired5_exact=True,
                recovery_iterator_exact=True,optimized_step_seconds=sum(times['optimized'][1:])/4,
                peak_cuda_allocated=torch.cuda.max_memory_allocated()))
            write(dest/'GPU_PARITY.json',dict(status='RUNNING',results=results))
            print('STAGE_GPU_PARITY_PASS',task,results[-1],flush=True)
            del ds,loader,refs,w,base,boundary,first,second,s;gc.collect();torch.cuda.empty_cache()
        del algo;gc.collect();torch.cuda.empty_cache()
        # Regression against the actual unchanged historical 64-window update implementation.
        from sample_step_closedloop_v1 import train as legacy
        name='s46_t8_e33_b148';spec,algo,ds,plan,base,Pold,lineage=legacy.assets(name)
        refs=[b for t in sorted({t for half,t in Pold}) for half in ['P0','P1'] for b in Pold[half,t]]
        hashes={}
        for arm in ['legacy','new']:
            g.restore(algo,base);w=Workspace(g.params(algo));hashes[arm]=[]
            for h in range(1,4):
                if arm=='legacy':legacy.update(algo,ds,plan,spec,Pold,lineage,name,'SAMPLE_STEP_QP',h,guard)
                else:
                    stage=spec['point']['stage_step']+h-1
                    if h>1:g.ensure_next_epoch_rng(algo,ds,plan,stage-1,stage)
                    update(algo,g.batch(ds,spec['batches'][h-1]),8,stage//plan['batches_per_epoch'],refs,w,guard)
                s=g.capture(algo);hashes[arm].append({k:g.digest(v) for k,v in s.items()})
        assert hashes['legacy']==hashes['new'],'Actual historical update regression'
        estimated=sum(r['optimized_step_seconds']*p['updates'] for r,p in zip(results,protocol['stage_plans']))
        report=dict(status='PASS',results=results,actual_legacy64_paired3_exact=True,
            gpu=torch.cuda.get_device_name(0),estimated_training_hours=estimated/3600,
            estimate_scope='Five-step engineering timings; references fixed, full states checked outside timer. Long-run runtime may vary.',
            frozen_protocol_sha256=sha(OUT/'PROTOCOL_FROZEN.json'))
        write(OUT/'PREFLIGHT_COMPLETE.json',report)
        (OUT/'PRECHECK_REPORT.md').write_text('# Preflight\n\nPASS: CPU tests; all four reference counts '
            '0/32/64/96 GPU state parity, iterator-boundary serialized recovery, and real historical64 regression.\n\n'+
            'Estimated update-only training hours: %.2f. Not a guaranteed runtime.\n'% (estimated/3600))
        print('PREFLIGHT_PASS',report,flush=True)
    finally:
        torch.distributed.destroy_process_group()

if __name__=='__main__':
    try:main()
    except BaseException:
        write(OUT/'PREFLIGHT_FAILED.json',dict(time=time.time(),error=traceback.format_exc()))
        raise
