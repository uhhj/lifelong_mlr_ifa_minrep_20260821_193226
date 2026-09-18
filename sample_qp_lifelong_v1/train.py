"""Native MLR stage schedule with explicit frozen data and recoverable epoch commits."""
import argparse
import contextlib
import gc
import io
import json
import subprocess
import time
import traceback
from collections import Counter
from .common import *

def run(resume=False):
    from functional_geometry_diagnostic_v1 import geometry as g
    from cb_ifa_experiment.gpu_runtime import bootstrap_train,build_train,atomic_save
    from libero.lifelong.utils import make_linear_schedule_with_warmup
    from .step import update,Workspace
    torch=g.torch
    protocol=verify();assert read(OUT/'PREFLIGHT_COMPLETE.json')['status']=='PASS'
    assert read(OUT/'PREFLIGHT_COMPLETE.json')['frozen_protocol_sha256']==sha(OUT/'PROTOCOL_FROZEN.json')
    assert torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3090'
    assert resume == RUN.exists(), 'Fresh run requires absent output; existing output requires explicit --resume'
    assert not (RUN/'TRAINING_COMPLETE.json').exists(), 'Completed training must not be rerun'
    RUN.mkdir(parents=True,exist_ok=True); (RUN/'events').mkdir(exist_ok=True)
    recovery=STORAGE/'recovery';recovery.mkdir(exist_ok=True)
    checkpoints=STORAGE/'checkpoints';checkpoints.mkdir(exist_ok=True)
    attempt=len(list(RUN.glob('ATTEMPT_*.json')))
    gpu_identity=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,driver_version','--format=csv,noheader']).decode().strip()
    if resume:assert read(RUN/'ATTEMPT_000.json')['gpu_identity']==gpu_identity,'Hardware changed: explicit recovery parity required'
    write(RUN/('ATTEMPT_%03d.json'%attempt),dict(pid=os.getpid(),time=time.time(),resume=resume,
          gpu_identity=gpu_identity,protocol_sha256=sha(OUT/'PROTOCOL_FROZEN.json')))
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    g.init_ddp(RUN/('ddp_%03d'%attempt));start=time.monotonic();elapsed_before=0
    accepted=[];finished=0;restore=None;guard=Guard()
    try:
        cfg=config(checkpoints);write(RUN/'config.json',dict(cfg))
        bootstrap_train(cfg);algo=build_train(cfg)
        expected=read(OUT/'INITIALIZATION_EXPECTED.json')
        assert g.digest(algo.policy.state_dict())==expected['model_hash']
        assert g.digest(g.rng_state())==expected['rng_hash']
        if resume:
            index=read(RUN/'RECOVERY_INDEX.json');assert sha(index['path'])==index['sha256']
            restore=torch.load(index['path'],map_location='cpu')
            assert restore['protocol_sha256']==sha(OUT/'PROTOCOL_FROZEN.json')
            accepted=restore['accepted_epochs'];finished=restore['global_step']
            elapsed_before=restore['elapsed_seconds']
            for row in accepted:assert sha(row['event_path'])==row['event_sha256']
        else:write(RUN/'INITIALIZATION.json',expected)
        stage_records=[]
        for task in TASKS:
            if restore and task<restore['task']:
                stage_records.append(read(RUN/('TASK%d_COMPLETE.json'%task)));continue
            plan,ds,sampler,loader=dataset_loader(task)
            with contextlib.redirect_stdout(io.StringIO()):algo.start_task(-1,loader)
            assert algo.scheduler_total_steps==plan['batches_per_epoch']*100
            refs=reference_batches(algo,task);w=Workspace(g.params(algo));first_epoch=0
            if restore and task==restore['task']:
                g.restore(algo,restore['state']);first_epoch=restore['next_epoch']
                assert all(g.compare_states(restore['state'],g.capture(algo)).values())
                del restore;restore=None
            lr_fn=make_linear_schedule_with_warmup(500,plan['batches_per_epoch']*100)
            for epoch in range(first_epoch,100):
                assert elapsed_before+time.monotonic()-start<120*3600,'Frozen training runtime budget'
                sampler.set_epoch(epoch);indices=list(iter(sampler));algo.policy.train()
                path=RUN/'events'/('task%d_epoch%03d_attempt%03d.jsonl'%(task,epoch,attempt))
                count=0;loss_sum=0;trigger_count=0;sum_correction=0.;epoch_start=time.monotonic()
                with path.open('x') as log:
                    for batch_index,data in enumerate(loader):
                        assert abs(algo.optimizer.param_groups[0]['lr']-1e-4*lr_fn(algo._optimizer_step_count))<1e-14
                        ids=[int(x) for x in data[1]]
                        ix=indices[batch_index*10:batch_index*10+len(ids)]
                        assert ids==[plan['records'][i]['task'] for i in ix]
                        rec=update(algo,data,task,epoch,refs,w,guard)
                        count+=1;finished+=1;loss_sum+=rec['mixed_bc'];trigger_count+=rec['trigger']
                        sum_correction+=rec.get('correction_ratio',0.)
                        rec.update(seed=SEED,task=task,epoch=epoch,batch=batch_index,
                            global_step=finished,record_indices=ix,sources=dict(Counter(map(str,ids))))
                        log.write(json.dumps(rec,allow_nan=False,separators=(',',':'))+'\n')
                        if count%20==0 or count==1:
                            log.flush()
                            write(OUT/'STATUS.json',dict(status='TRAINING',seed=SEED,task=task,
                                epoch=epoch,batch=count,batches_per_epoch=len(loader),global_step=finished,
                                total_steps=105500,pid=os.getpid(),time=time.time(),
                                elapsed_seconds=elapsed_before+time.monotonic()-start))
                assert count==plan['batches_per_epoch']
                assert algo._optimizer_step_count==(epoch+1)*count
                assert algo._scheduler_step_count==algo._optimizer_step_count
                # Match native model mode changes and checkpoint epochs exactly.
                if epoch in SAVED_EPOCHS:
                    algo.policy.eval();target=checkpoints/('mlrifa_%d_model_ep%d.pth'%(task,epoch))
                    state_dict=g.cpu_state(algo.policy.state_dict())
                    if target.exists():
                        assert g.digest(torch.load(target,map_location='cpu')['state_dict'])==g.digest(state_dict)
                    else:atomic_save(target,dict(state_dict=state_dict,cfg=cfg,previous_masks=None))
                    del state_dict
                algo.epoch_schedule_step(epoch)
                row=dict(task=task,epoch=epoch,steps=count,global_step=finished,
                    mean_bc=loss_sum/count,trigger_count=trigger_count,mean_correction_ratio=sum_correction/count,
                    seconds=time.monotonic()-epoch_start,event_path=str(path),event_sha256=sha(path))
                accepted.append(row)
                # Commit an epoch only after its event file and full state are durable.
                state=g.capture(algo);slot=((task-6)*100+epoch)%2
                state_path=recovery/('slot%d.pt'%slot)
                atomic_save(state_path,dict(state=state,task=task,next_epoch=epoch+1,global_step=finished,
                    accepted_epochs=accepted,elapsed_seconds=elapsed_before+time.monotonic()-start,
                    protocol_sha256=sha(OUT/'PROTOCOL_FROZEN.json'),
                    plan_sha256=sha(OUT/('plans/task%d.json'%task)),
                    reference_sha256=sha(OUT/('plans/references_task%d.json'%task)),
                    boundary='After native epoch_schedule_step and checkpoint mode changes; before next DataLoader iterator'))
                write(RUN/'RECOVERY_INDEX.json',dict(path=str(state_path),sha256=sha(state_path),
                    task=task,epoch=epoch,global_step=finished))
                write(RUN/'ACCEPTED_EPOCHS.json',accepted)
                print('EPOCH_COMPLETE',task,epoch,finished,'/',105500,'seconds',round(row['seconds'],2),flush=True)
                del state
            endpoint=checkpoints/('mlrifa_%d_model_ep99.pth'%task)
            stage=dict(task=task,epochs=100,updates=100*plan['batches_per_epoch'],
                       checkpoint=str(endpoint),sha256=sha(endpoint),completed_at=time.time())
            write(RUN/('TASK%d_COMPLETE.json'%task),stage);stage_records.append(stage)
            del ds,loader,refs,w;gc.collect();torch.cuda.empty_cache()
        assert finished==105500 and len(accepted)==400
        assert {(x['task'],x['epoch']) for x in accepted}=={(t,e) for t in TASKS for e in range(100)}
        verify()
        write(RUN/'TRAINING_COMPLETE.json',dict(status='PASS',seed=SEED,arm='SAMPLE_STEP_QP',
            updates=finished,epochs=len(accepted),stages=stage_records,
            seconds=elapsed_before+time.monotonic()-start,protocol_sha256=sha(OUT/'PROTOCOL_FROZEN.json')))
        write(OUT/'STATUS.json',dict(status='TRAINING_COMPLETE',time=time.time(),updates=finished))
    finally:torch.distributed.destroy_process_group()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    try:run(args.resume)
    except BaseException:
        write(OUT/'TRAINING_ERROR.json',dict(time=time.time(),error=traceback.format_exc()))
        raise
