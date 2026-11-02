"""Single-use isolated exact-check microbenchmark. No production writes or updates."""
import os,sys,time,json,hashlib,traceback,copy,contextlib,subprocess
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1')
D=Path('/data/mlr_ifa/diagnostics/lambda_3090_checks_20261102_v1')
assert not D.exists(),'single-use diagnostic; read RESULT instead of rerunning'
D.mkdir()
ledger=dict(status='RUNNING',started=time.time(),optimizer_calls=0,environments=0,target_signals=0,production_writes=0,full_training_equivalence='NOT_TESTED',production_activation=False)
def persist():
    p=D/'RESULT.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(ledger,indent=2));os.replace(tmp,p)
persist()
def main():
    os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    sys.path.insert(0,str(Q/'L005_41'))
    import runtime as rt
    torch=rt.torch
    assert torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3090'
    contract=rt.js(Q/'CONTRACT.json');assert all(rt.sha(p)==h for p,h in contract['files'].items())
    cp=Path('/data/mlr_ifa/experiments')/Q.name/'engineering_resume_3090_20261102_v1/interrupted_epoch260.pth'
    assert rt.sha(cp)=='65b8d3b4c9f589fb543ba5b7a41991a8c1dbc50daae309c21c386471178b4070'
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>12000
    loaded=torch.load(cp,map_location='cpu')
    rt.R=D;rt.OUTPUT['S']=D/'private';rt.ROOT_PRIVATE=D/'private_root';rt.DATA_PRIVATE=D/'private'
    torch.distributed.init_process_group('nccl',init_method='file://'+str(D/'store'),rank=0,world_size=1);torch.cuda.set_device(0)
    try:
        a=rt.make('S');ds=rt.Dataset(loaded['stage']);loader,sampler=rt.setup(a,ds);rt.restore(a,loaded['state']);del loaded
        original=rt.cpu
        def device(x):
            if torch.is_tensor(x):return x.detach().clone()
            if isinstance(x,dict):return {k:device(v) for k,v in x.items()}
            if isinstance(x,list):return [device(v) for v in x]
            if isinstance(x,tuple):return tuple(device(v) for v in x)
            return copy.deepcopy(x)
        def capture(fn):
            rt.cpu=fn
            try:return rt._reference_capture(a)
            finally:rt.cpu=original
        baseline=capture(original);selected=capture(device)
        def reference(x,y):return rt.equal(original(x),original(y))
        def candidate(x,y):return rt.equal(x,y)
        assert reference(selected,capture(original)) and candidate(selected,capture(device))
        rounds=[]
        for i in range(12):
            for mode in (['cpu','device'] if i%2==0 else ['device','cpu']):
                torch.cuda.synchronize();tic=time.perf_counter()
                if mode=='cpu':assert reference(selected,capture(original))
                else:assert candidate(selected,capture(device))
                torch.cuda.synchronize();rounds.append(dict(round=i,mode=mode,seconds=time.perf_counter()-tic))
            assert rt.equal(baseline,capture(original))
        mismatch_checks=[]
        for category in ['model','grads','optimizer']:
            def leaves(x):
                if torch.is_tensor(x):yield x
                elif isinstance(x,dict):
                    for v in x.values():yield from leaves(v)
                elif isinstance(x,(tuple,list)):
                    for v in x:yield from leaves(v)
            t=next(v for v in leaves(selected[category]) if v.numel() and v.is_floating_point())
            saved=t.flatten()[0].clone();t.view(-1)[0]=float('nan')
            assert not reference(selected,capture(original)) and not candidate(selected,capture(device))
            t.view(-1)[0]=saved;mismatch_checks.append(category)
        saved=selected['step'];selected['step']+=1
        assert not reference(selected,capture(original)) and not candidate(selected,capture(device))
        selected['step']=saved;mismatch_checks.append('step')
        assert candidate(selected,capture(device)) and rt.equal(baseline,capture(original))
        grads=[p.grad for p in a.policy.parameters()]
        loss=torch.tensor(1.,device='cuda')
        def finite_old(loss,grads):return bool(torch.isfinite(loss)) and all(g is None or bool(torch.isfinite(g).all()) for g in grads)
        def finite_new(loss,grads):return bool(torch.stack([torch.isfinite(loss).all()]+[torch.isfinite(g).all() for g in grads if g is not None]).all())
        assert finite_old(loss,grads)==finite_new(loss,grads)==True
        finite_rounds=[]
        for i in range(12):
            for mode in (['serial','batched'] if i%2==0 else ['batched','serial']):
                fn=finite_old if mode=='serial' else finite_new
                torch.cuda.synchronize();tic=time.perf_counter()
                for _ in range(20):assert fn(loss,grads)
                torch.cuda.synchronize();finite_rounds.append(dict(round=i,mode=mode,seconds_per_call=(time.perf_counter()-tic)/20))
        cases=0
        for val in [float('nan'),float('inf'),float('-inf')]:
            testloss=torch.tensor(val,device='cuda');assert finite_old(testloss,grads)==finite_new(testloss,grads)==False;cases+=1
            for pos in [0,len(grads)//2,len(grads)-1]:
                testgrads=list(grads);testgrads[pos]=testloss
                assert finite_old(loss,testgrads)==finite_new(loss,testgrads)==False;cases+=1
        assert rt.equal(baseline,capture(original)) and rt.CALLS=={'production':0,'admission':0}
        assert all(rt.sha(p)==h for p,h in contract['files'].items())
        ledger.update(status='PASS_MICROBENCH_EXACT_CHECKS_ONLY',ended=time.time(),state_check_rounds=rounds,finite_check_rounds=finite_rounds,
            nonfinite_cases=cases,mismatch_categories=mismatch_checks,gradient_tensor_count=sum(g is not None for g in grads),
            complete_state_unchanged=True,input_sha256=rt.sha(cp),max_memory_allocated=torch.cuda.max_memory_allocated(),
            limit='Concurrent GPU workload and microbench only; no full training-path equality or production speedup claimed.')
    finally:torch.distributed.destroy_process_group()
try:
    with (D/'diagnostic.log').open('x') as log,contextlib.redirect_stdout(log):main()
except BaseException:
    ledger.update(status='FAIL_DIAGNOSTIC_ONLY',ended=time.time(),error=traceback.format_exc())
persist();print(json.dumps(ledger))
