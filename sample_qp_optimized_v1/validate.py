"""Read-only asset audit and real-policy GPU parity/speed acceptance runner.

python -m sample_qp_optimized_v1.validate --audit-only
python -m sample_qp_optimized_v1.validate --run-gpu
"""
import argparse
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import time
import traceback

ROOT = Path.cwd()
OUT = ROOT/'reports/sample_qp_optimized_v1'
OLD = ROOT/'reports/sample_step_closedloop_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(str(tmp),str(path))


def emit(path,value):
    with Path(path).open('a',encoding='utf-8') as f:
        f.write(json.dumps(value,allow_nan=False)+'\n');f.flush()


def manifest():
    original=read(OLD/'FROZEN_INPUTS.json')
    for name,h in original.items():assert sha(ROOT/name)==h,('Historical source changed',name)
    files={str(OLD/'PROTOCOL_FROZEN.json'):sha(OLD/'PROTOCOL_FROZEN.json')}
    files.update({str(ROOT/name):h for name,h in original.items()})
    files.update({str(f):sha(f) for f in (ROOT/'sample_qp_optimized_v1').glob('*.py')})
    return files


def asset_audit():
    import torch
    files=manifest();protocol=read(OLD/'PROTOCOL_FROZEN.json')
    members=read(OLD/'TRAINING_MEMBERSHIP.json');points=[]
    for point in protocol['points']:
        item=members[point];anchor=item['origin']['anchor'];path=Path(anchor['path'])
        assert sha(path)==anchor['sha256']
        state=torch.load(path,map_location='cpu')
        required=['model','optimizer','scheduler','rng','optimizer_step','scheduler_step','training_modes']
        assert all(k in state for k in required)
        assert len(item['batches'])==300
        endpoint=read(OLD/'training'/point/'SAMPLE_STEP_QP'/'COMPLETE.json')
        assert endpoint['status']=='PASS' and endpoint['steps']==300
        cp=endpoint['checkpoint'];assert sha(cp['path'])==cp['sha256']
        files[str(path)]=anchor['sha256'];files[cp['path']]=cp['sha256']
        for f in [OLD/'training'/point/'SAMPLE_STEP_QP'/'COMPLETE.json',
                  OLD/'training'/point/'SAMPLE_STEP_QP'/'events.jsonl']:
            files[str(f)]=sha(f)
        points.append(dict(point=point,complete_state_fields=required,
                           optimizer_step=state['optimizer_step'],future_batches=300))
        del state;gc.collect()
    result=dict(status='PASS',points=points,torch=torch.__version__,python=platform.python_version(),
                cuda_available=torch.cuda.is_available(),files=files,
                scope='CPU assets only; no claim of GPU numerical parity or speed')
    write(OUT/'ASSET_AUDIT.json',result)
    print('ASSET_AUDIT_PASS',len(points),'CUDA',result['cuda_available'],flush=True)
    return result


class Guard:
    def __init__(self,point,queue_start):
        self.point=point;self.start=time.monotonic();self.last=0.;self.queue_start=queue_start
    def check(self,force=False):
        import torch
        from functional_geometry_diagnostic_v1.runtime_guard import memory,size
        now=time.monotonic()
        if not force and now-self.last<1:return
        self.last=now;m=memory()
        assert now-self.start<7200,'Per-origin two-hour budget exceeded'
        assert now-self.queue_start<6*3600,'Six-hour total validation budget exceeded'
        assert m['rss']<28*2**30 and m['available']>2*2**30
        assert torch.cuda.memory_allocated()<20*2**30 and torch.cuda.memory_reserved()<22*2**30
        assert shutil.disk_usage(ROOT).free>1*2**30 and size(OUT)<512*2**20
        write(OUT/'HEARTBEAT.json',dict(point=self.point,pid=os.getpid(),time=time.time(),
            seconds=now-self.start,rss=m['rss'],cuda_allocated=torch.cuda.memory_allocated(),
            cuda_reserved=torch.cuda.memory_reserved()))


@contextlib.contextmanager
def legacy_profile(legacy):
    """Nested inclusive timings; only disjoint top-level phases may be added."""
    from .core import Timings
    t=Timings(True);patches=[]
    targets=[(legacy,'sample_gradients','reference_total'),(legacy,'inner','outer_inner'),
             (legacy,'sample_project','projection_total'),(legacy.g,'capture','capture_inclusive'),
             (legacy.g,'digest','hash_inclusive'),(legacy.g,'native','native_mlr'),
             (legacy.g,'assign_displacement','parameter_write'),
             (legacy.g,'displacement','cpu_displacement')]
    for obj,name,label in targets:
        original=getattr(obj,name)
        def wrapped(*a,_fn=original,_label=label,**kw):
            with t.phase(_label):return _fn(*a,**kw)
        patches.append((obj,name,original));setattr(obj,name,wrapped)
    try:yield t
    finally:
        for obj,name,value in reversed(patches):setattr(obj,name,value)


def run_gpu():
    import torch
    assert torch.cuda.is_available(),'GPU unavailable: no validation or speed claim can be made'
    run_dir=OUT/'gpu_validation'
    assert not run_dir.exists(),'Existing validation is preserved; do not silently rerun'
    audit=asset_audit()
    from sample_step_closedloop_v1 import train as legacy
    from sample_task_qp_v1.measure import sample_gradients as reference_gradients
    from .adapter import update
    from .core import Workspace
    g=legacy.g;torch.set_num_threads(2);torch.set_num_interop_threads(1)
    run_dir.mkdir();protocol=read(OLD/'PROTOCOL_FROZEN.json')
    frozen=dict(version='engineering_exact_v1',points=protocol['points'],short_steps=20,
        long_steps=300,comparison='every step exact state hash in paired20; exact original300 endpoint',
        scientific_changes=False,reference_windows_per_old_task=32,reference_refresh='each step',
        solver='unchanged SLSQP and polish',gram='unchanged float64 chunks65536',
        hardware=torch.cuda.get_device_name(0),files=audit['files'],
        run_order=['legacy,opt','opt,legacy','legacy,opt'],gpu_budget_hours=6,
        acceptance='all three points exact20 and historical300 parity; paired20 wall speedup >1')
    write(run_dir/'PROTOCOL_FROZEN.json',frozen)
    g.init_ddp(run_dir/'ddp');results=[];overall_start=time.monotonic()
    try:
        for index,name in enumerate(protocol['points']):
            assert time.monotonic()-overall_start<6*3600
            write(run_dir/'STATUS.json',dict(status='RUNNING',point=name,phase='restore'))
            spec,algo,ds,plan,base,Pold,lineage=legacy.assets(name)
            guard=Guard(name,overall_start);w=Workspace(g.params(algo));states={};timings={};records={}
            def do(which,h):
                if which=='legacy':return legacy.update(algo,ds,plan,spec,Pold,lineage,name,'SAMPLE_STEP_QP',h,guard)
                return update(algo,ds,plan,spec,Pold,name,h,w,guard)
            # Check reference gradients themselves against production, before optimizing states.
            g.restore(algo,base)
            oldH,_=reference_gradients(algo,Pold,lineage,name,guard)
            tasks=sorted({t for half,t in Pold});batches=[b for t in tasks for half in ['P0','P1'] for b in Pold[half,t]]
            modes={n:m.training for n,m in algo.policy.named_modules()};algo.policy.eval()
            try:
                def loss(data):return g.module(algo).policy_head.loss_fn(g.gmm(algo,data),data['actions'].squeeze(1),'none')
                H=w.sample_gradients(batches,loss,64)
            finally:
                for n,m in algo.policy.named_modules():m.training=modes[n]
            assert torch.equal(oldH,H),'Reference gradient bits differ'
            assert all(g.compare_states(base,g.capture(algo)).values())
            del oldH,H;gc.collect()
            # Separate synchronized profiling, never used as the speed comparison.
            g.restore(algo,base)
            with legacy_profile(legacy) as t:do('legacy',1)
            baseline_profile=dict(t.seconds)
            g.restore(algo,base);optimized_profile=update(algo,ds,plan,spec,Pold,name,1,w,guard,profile=True)
            write(run_dir/(name+'_PROFILE.json'),dict(legacy_inclusive=baseline_profile,
                optimized_disjoint=optimized_profile['profile_seconds'],
                note='Inclusive legacy timers overlap. Profiling synchronization is excluded from speed benchmark.'))
            # Equal warmup, same reference data and samples, alternate execution order by seed.
            order=['legacy','opt'] if index%2==0 else ['opt','legacy']
            for which in order:
                g.restore(algo,base);do(which,1);g.restore(algo,base)
                states[which]=[];timings[which]=[];records[which]=[]
                for h in range(1,21):
                    torch.cuda.synchronize();start=time.perf_counter();rec=do(which,h)
                    torch.cuda.synchronize();timings[which].append(time.perf_counter()-start)
                    # Same external hash audit for both arms, outside measured step time.
                    state=g.capture(algo);states[which].append({k:g.digest(v) for k,v in state.items()})
                    records[which].append(rec);del state
            assert states['legacy']==states['opt'],'Real-policy paired20 state mismatch'
            for x,y in zip(records['legacy'],records['opt']):
                for key in ['lr','batch_indices','trigger','active','kkt','projection_residual',
                            'actual_correction_norm','native_norm','actual_norm','actual_positive']:
                    assert x[key]==y[key],('Diagnostic mismatch',name,key)
            write(run_dir/(name+'_PAIRED20.json'),dict(state_hashes=states,seconds=timings,
                  all_states_exact=True,records=records))
            # Run unchanged frozen300 sequence and compare saved historical endpoint hashes.
            g.restore(algo,base);start=time.perf_counter()
            events=read(OLD/'training'/name/'SAMPLE_STEP_QP'/'COMPLETE.json')
            for h in range(1,301):
                rec=do('opt',h);emit(run_dir/(name+'_events.jsonl'),rec)
                if h%25==0:
                    write(run_dir/'STATUS.json',dict(status='RUNNING',point=name,phase='exact300',step=h,total=300))
                    print('OPTIMIZED_PROGRESS',name,h,flush=True)
            torch.cuda.synchronize();long_seconds=time.perf_counter()-start
            final=g.capture(algo);actual={k:g.digest(v) for k,v in final.items()}
            assert actual==events['full_state_hashes'],'Historical300 model/optimizer/RNG mismatch'
            result=dict(point=name,reference_gradient_exact=True,paired20_exact=True,
                original300_exact=True,legacy20_seconds=sum(timings['legacy']),
                optimized20_seconds=sum(timings['opt']),
                speedup=sum(timings['legacy'])/sum(timings['opt']),
                optimized300_seconds=long_seconds,peak_cuda_allocated=torch.cuda.max_memory_allocated())
            results.append(result);write(run_dir/(name+'_COMPLETE.json'),result)
            del algo,ds,base,Pold,lineage,w,final;gc.collect();torch.cuda.empty_cache()
        for name,h in frozen['files'].items():assert sha(name)==h,('Frozen input changed during run',name)
        speedup=sum(r['legacy20_seconds'] for r in results)/sum(r['optimized20_seconds'] for r in results)
        passed=all(r['speedup']>1 for r in results)
        write(run_dir/'COMPLETE.json',dict(status='PASS' if passed else 'NO_SPEED_GAIN',results=results,
            aggregate_speedup=speedup,scientific_changes=False,real_policy_gpu_verified=True))
        print('GPU_VALIDATION_COMPLETE',speedup,flush=True)
    finally:
        torch.distributed.destroy_process_group()


def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--audit-only',action='store_true');group.add_argument('--run-gpu',action='store_true')
    args=parser.parse_args()
    if args.audit_only:asset_audit()
    else:run_gpu()


if __name__=='__main__':
    try:main()
    except BaseException:
        write(OUT/'LAST_ERROR.json',dict(time=time.time(),error=traceback.format_exc()))
        if (OUT/'gpu_validation').exists():
            write(OUT/'gpu_validation/STATUS.json',dict(status='FAILED_STOP',time=time.time(),
                error_file=str(OUT/'LAST_ERROR.json')))
        raise
