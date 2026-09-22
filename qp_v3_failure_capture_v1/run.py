"""Bounded diagnostic replay from the verified post-b185 state. Never resumes formal training."""
import contextlib, hashlib, inspect, io, json, os, subprocess, time, traceback
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from .capture import write_json
from . import solver, core, step
from sample_qp_lifelong_v1_continuation_v3 import common as c

ROOT=c.ROOT
OUT=ROOT/'reports/qp_v3_failure_capture_v1'
STORE=Path('/root/qp_v3_failure_capture_v1')
PARENT=c.OUT/'regression/b185_writeback/post_step_state.pt'
EVENT=c.RUN/'events/task8_epoch011_attempt001.jsonl'
EXPECTED_PARENT=c.read(c.OUT/'regression/b185_writeback/WRITEBACK_VALIDATION.json')['post_step_state_sha256']

def status(stage,**kw):
    write_json(OUT/'STATUS.json',dict(stage=stage,updated_at=time.time(),**dict(kw,formal_training_resumed=False)))

def fingerprint(obj,path=''):
    result=[]
    if torch.is_tensor(obj):
        v=obj.detach().cpu().contiguous()
        result.append(dict(key=path,shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest(),finite=bool(torch.isfinite(v).all()),minimum=float(v.min()) if v.numel() else None,maximum=float(v.max()) if v.numel() else None))
    elif isinstance(obj,dict):
        for k,v in obj.items():result.extend(fingerprint(v,path+'/'+str(k)))
    elif isinstance(obj,(list,tuple)):
        for i,v in enumerate(obj):result.extend(fingerprint(v,path+'/'+str(i)))
    return result

def atomic_state(path,value):
    assert not path.exists(),str(path)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def main():
    from functional_geometry_diagnostic_v1 import geometry as g
    from cb_ifa_experiment import gpu_runtime as runtime
    OUT.mkdir(exist_ok=True); STORE.mkdir(exist_ok=True)
    assert c.read(OUT/'CAPTURE_SMOKE.json')['status']=='PASS'
    assert core.solve is solver.solve and step.project is core.project
    assert '/qp_v3_failure_capture_v1/' in solver.solve.__code__.co_filename
    assert c.sha(PARENT)==EXPECTED_PARENT,'Parent checkpoint changed'
    hist=[json.loads(line) for line in EVENT.read_text().splitlines() if line.strip()]
    assert [r['batch'] for r in hist]==list(range(len(hist)))
    assert len(hist)==258 and hist[-1]['batch']==257
    target=hist[-1]['batch']+1
    write_json(OUT/'FAILURE_BATCH_LOCALIZATION.json',dict(event=str(EVENT),sha256=c.sha(EVENT),rows=len(hist),first_batch=hist[0]['batch'],last_success=hist[-1],failure_batch=target,failure_pre_update_global_step=hist[-1]['global_step'],reason='Event is written only after successful update; STATUS is a periodic progress snapshot, not a failure index.',formal_training_resumed=False))
    tracked={}
    for m in (solver,core,step,g,runtime):
        p=Path(inspect.getsourcefile(m));tracked[m.__name__]=dict(path=str(p),sha256=c.sha(p))
    write_json(OUT/'RUNTIME_BINDING.json',dict(core_solver_identity=core.solve is solver.solve,step_core_identity=step.project is core.project,modules=tracked,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),nvidia_smi=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True),diagnostic_override=True,formal_training_resumed=False))
    parent=torch.load(PARENT,map_location='cpu')
    assert (parent['task'],parent['epoch'],parent['batch'],parent['global_step'])==(8,11,185,41985)
    assert parent['optimizer_step']==parent['scheduler_step']==3585
    write_json(OUT/'PLAN.json',dict(parent=str(PARENT),parent_sha256=EXPECTED_PARENT,parent_global_step=41985,start_batch=186,target_batch=target,successful_prefix_updates=72,total_update_calls_max=73,diagnostic_storage=str(STORE),formal_storage_writable_by_diag=False,formal_training_resumed=False))
    status('BOOTSTRAP',target_batch=target)
    g.init_ddp(STORE/'ddp')
    try:
        cfg=c.config(STORE/'bootstrap')
        runtime.bootstrap_train(cfg);algo=runtime.build_train(cfg)
        plan,ds,sampler,loader=c.dataset_loader(8)
        with contextlib.redirect_stdout(io.StringIO()):algo.start_task(-1,loader)
        refs=c.reference_batches(algo,8)
        sampler.set_epoch(11); order=list(iter(sampler))
        w=core.Workspace(g.params(algo))
        g.restore(algo,parent['state'])
        restored=g.compare_states(parent['state'],g.capture(algo))
        assert all(restored.values()),restored
        write_json(OUT/'RESTORE.json',dict(checks=restored,optimizer_step=algo._optimizer_step_count,scheduler_step=algo._scheduler_step_count,lr=algo.optimizer.param_groups[0]['lr'],formal_training_resumed=False))
        del parent
        history={r['batch']:r for r in hist}
        global_step=41985
        logs=OUT/'short_replay_events.jsonl'
        assert not logs.exists(),'Do not overwrite previous diagnostic replay'
        for batch_index in range(186,target+1):
            ix=order[batch_index*10:(batch_index+1)*10]
            data=g.batch(ds,ix)
            sources=dict(Counter(str(int(v)) for v in data[1]))
            assert all(str(plan['records'][i]['task'])==str(int(s)) for i,s in zip(ix,data[1]))
            status('SHORT_REPLAY',batch=batch_index,pre_update_global_step=global_step,matched_updates=batch_index-186)
            diag=OUT/('t8_e11_b%03d'%batch_index)
            if batch_index==target:
                assert global_step==42057 and algo._optimizer_step_count==algo._scheduler_step_count==3657
                state=g.capture(algo)
                path=STORE/'PRE_FAILURE_STATE.pt'
                metadata=dict(task=8,epoch=11,batch=batch_index,global_step=global_step,optimizer_step=algo._optimizer_step_count,scheduler_step=algo._scheduler_step_count,lr=float(algo.optimizer.param_groups[0]['lr']),record_indices=ix,sources=sources,reference_sha=c.sha(c.OUT/'plans/references_task8.json'),protocol_sha=c.sha(c.OUT/'PROTOCOL_FROZEN.json'))
                atomic_state(path,dict(state=state,**metadata))
                loaded=torch.load(path,map_location='cpu')
                checks=g.compare_states(loaded['state'],g.capture(algo))
                assert all(checks.values()),checks
                g.restore(algo,loaded['state'])
                reload_checks=g.compare_states(state,g.capture(algo))
                assert all(reload_checks.values()),reload_checks
                write_json(diag/'PRE_FAILURE_STATE.json',dict(path=str(path),sha256=c.sha(path),size=path.stat().st_size,save_unchanged=checks,reload=reload_checks,**metadata))
                write_json(diag/'FAILURE_BATCH_INPUT.json',dict(**metadata,records=[plan['records'][i] for i in ix],reference_manifest=str(c.OUT/'plans/references_task8.json')))
                refprints=[]
                for ri,rb in enumerate(refs):
                    for wi in range(len(rb['feature'])):
                        refprints.extend(fingerprint({k:v[wi] for k,v in rb.items() if torch.is_tensor(v)},'ref/%d/%d'%(ri,wi)))
                write_json(diag/'UPDATE_INPUT_FINGERPRINT.json',dict(batch=fingerprint(data,'batch'),reference_windows=refprints,runtime=tracked))
                del loaded,state
            began=time.perf_counter()
            try:
                actual=step.update(algo,data,8,11,refs,w,guard=None,diag_dir=diag)
            except Exception as ex:
                err=dict(batch=batch_index,pre_update_global_step=global_step,error=repr(ex),traceback=traceback.format_exc(),seconds=time.perf_counter()-began,optimizer_step_after=algo._optimizer_step_count,scheduler_step_after=algo._scheduler_step_count,formal_training_resumed=False)
                write_json(diag/'UPDATE_EXCEPTION.json',err)
                if batch_index!=target:
                    status('SHORT_REPLAY_DIVERGED',**err);raise
                before=diag/'before_fallback.npz';after=diag/'after_fallback.npz'
                assert before.exists() and after.exists(),'Capture incomplete'
                z=np.load(after);p=z['b']-z['A']@z['x']
                parts=dict(primal_positive=float(np.maximum(p,0).max()),nonnegative_violation=float(np.maximum(-z['x'],0).max()),complementarity=float(np.abs(z['x']*p).max()))
                kkt=max(parts.values())
                assert abs(kkt-1.547045713756745e-8)<1e-18,('Historical failure not reproduced',kkt)
                assert algo._optimizer_step_count==algo._scheduler_step_count==3658
                status('HISTORICAL_FAILURE_CAPTURED',batch=target,matched_updates=72,kkt=kkt,parts=parts,pre_failure_state=str(STORE/'PRE_FAILURE_STATE.pt'),diagnostic_dir=str(diag))
                print('HISTORICAL_FAILURE_CAPTURED',kkt,flush=True)
                return
            global_step+=1
            actual.update(batch=batch_index,epoch=11,seed=46,task=8,global_step=global_step,record_indices=ix,sources=sources)
            if batch_index==target:
                write_json(diag/'UNEXPECTED_SUCCESS.json',actual)
                status('TARGET_FAILURE_NOT_REPRODUCED',batch=batch_index);return
            expected=history[batch_index]
            differences={}
            for key,value in expected.items():
                if key in ('seconds','fallback_seconds'):continue
                observed=actual.get(key)
                if observed!=value:
                    detail=dict(expected=value,actual=observed)
                    if isinstance(value,(float,int)) and isinstance(observed,(float,int)):
                        detail.update(absolute_difference=abs(observed-value),relative_difference=abs(observed-value)/max(abs(value),1e-300))
                    differences[key]=detail
            with logs.open('a') as f:
                f.write(json.dumps(dict(actual=actual,exact_event_match=not differences,differences=differences),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
            if differences:
                status('SHORT_REPLAY_DIVERGED',batch=batch_index,differences=differences)
                raise RuntimeError('SHORT_REPLAY_DIVERGED_AT_BATCH_%d'%batch_index)
            print('MATCH',batch_index,'seconds',actual['seconds'],flush=True)
    finally:
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()

if __name__=='__main__':
    try:main()
    except BaseException:
        write_json(OUT/'RUN_ERROR.json',dict(traceback=traceback.format_exc(),formal_training_resumed=False));raise
