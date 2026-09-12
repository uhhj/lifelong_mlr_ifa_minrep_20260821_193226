"""New controlled MLR carriers with nonperturbing 120-point diagnostics."""
import argparse
import contextlib
import gzip
import hashlib
import io
import struct
import time
import traceback
from gradient_relation_experiment.kernel import *
from gradient_relation_experiment.prepare import OUT,epoch_indices
from gradient_relation_experiment.preflight import load_banks
from confirmation_experiment.common import state_digest,rng_state as historical_rng


def verify_code():
    frozen=read(OUT/'FROZEN_EXECUTION.json')
    for rel,h in frozen['files'].items():assert sha(ROOT/rel)==h,('Frozen file changed',rel)
    assert frozen['carrier_source']=='new_controlled_3090_from_official'
    assert read(OUT/'PREFLIGHT_COMPLETE.json')['status']=='PASS'


def run(seed):
    verify_code();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    out=ROOT/f'runs/gradient_relation_v1/seed{seed}';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'STARTED.json').exists()
    write(out/'STARTED.json',{'seed':seed,'pid':os.getpid(),'time':time.time(),'carrier':'new_controlled_3090_from_official'})
    begin=time.monotonic();completed_points=0
    try:
        init_ddp(out/'ddp_store')
        with gzip.open(OUT/'DESIGN.json.gz','rt') as f:design=json.load(f)
        cfg=train_config(seed,out);bootstrap_train(cfg);algo=build_train(cfg);algo.train()
        write(out/'config.json',dict(cfg))
        streams={name:(out/(name+'.jsonl')).open('x',buffering=1024*1024) for name in
            ('scores','labels','losses','geometry','equal_norm','checks','unobserved','carrier_checks','epochs')}
        def emit(kind,row):
            streams[kind].write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n')
            if kind in ('scores','checks','carrier_checks'):streams[kind].flush()
        banks=load_banks(algo,design,(6,7,8));global_step=0
        initial_sha=digest(algo.policy.state_dict());pending={}
        for current in (6,7,8):
            plan_path=ROOT/f'runs/fixed_kappa_confirmation_v2/formal/seed{seed}/mlr/plans/task{current}.json'
            plan=read(plan_path);ds=CachedWindows(plan['immutable_feature_origins'],plan['record_sha256'])
            sampler=torch.utils.data.DistributedSampler(ds,**{k:plan['sampler'][k] for k in ('seed','num_replicas','rank','shuffle','drop_last')})
            dl=torch.utils.data.DataLoader(ds,batch_size=10,sampler=sampler,num_workers=0)
            with contextlib.redirect_stdout(io.StringIO()):algo.start_task(-1,dl)
            points={p['stage_step']:p for p in design['points'] if p['seed']==seed and p['task']==current}
            trace=hashlib.sha256()
            for epoch in range(100):
                sampler.set_epoch(epoch);algo.policy.train();loss_sum=0.;epoch_begin=time.monotonic()
                for batch_index,data in enumerate(dl):
                    step=epoch*len(dl)+batch_index
                    if step in points:
                        point=points[step]
                        assert [int(x) for x in data[1]]==point['first_task_ids']
                        assert all(torch.equal(data[0][k],batch(ds,point['future20_record_indices'][0])[0][k]) for k in ('feature','actions'))
                        if point['phase']=='30-69' and point['slot']==0:
                            state=capture(algo);state.update(point=point,plan_path=str(plan_path.relative_to(ROOT)),
                                plan_sha256=sha(plan_path),global_step=global_step,config=dict(cfg),
                                code_manifest_sha256=sha(OUT/'FROZEN_EXECUTION.json'),
                                boundary='pre-forward, current DataLoader iterator already created; next20 batches frozen')
                            atomic_save(out/f'presaved_task{current}_early_slot0.pt',state)
                        write(out/'STATUS.json',{'seed':seed,'task':current,'epoch':epoch,'batch':batch_index,'phase':'diagnostic',
                            'points_completed':completed_points,'point':point['point'],'global_step':global_step,'wall_seconds':time.monotonic()-begin})
                        expected=run_point(algo,ds,plan,point,banks,emit)
                        for h,witness in expected.items():pending.setdefault(step+h,[]).append((point['point'],h,witness))
                        completed_points+=1
                        for stream in streams.values():stream.flush()
                    loss=native(algo,data,current,epoch);assert np.isfinite(loss)
                    trace.update(struct.pack('!d',float(loss)));loss_sum+=loss;global_step+=1
                    if step+1 in pending:
                        state=capture(algo)
                        for point,h,expected in pending.pop(step+1):
                            matches={k:digest(state[k])==v for k,v in expected.items()}
                            emit('carrier_checks',{'point':point,'horizon':h,'matches':matches})
                            assert all(matches.values()),('Actual native carrier differs from diagnostic11',point,h,matches)
                if (epoch%5==0 and epoch>69) or epoch==99:algo.policy.eval()
                algo.epoch_schedule_step(epoch)
                row={'seed':seed,'task':current,'epoch':epoch,'global_step':global_step,'mean_bc':loss_sum/len(dl),
                    'points_completed':completed_points,'phase':'native_carrier','epoch_seconds':time.monotonic()-epoch_begin,
                    'wall_seconds':time.monotonic()-begin}
                emit('epochs',row);write(out/'STATUS.json',row)
                if epoch%10==9:
                    recovery=capture(algo);recovery.update(seed=seed,current_task=current,epoch=epoch,global_step=global_step,
                        plan_path=str(plan_path.relative_to(ROOT)),plan_sha256=sha(plan_path),config=dict(cfg),
                        pending_carrier_checks=pending,points_completed=completed_points,code_manifest_sha256=sha(OUT/'FROZEN_EXECUTION.json'),
                        boundary='end of epoch, before next DataLoader iterator; incomplete scientific outputs retained if interrupted')
                    atomic_save(out/'RECOVERY.pt',recovery)
                    print('CARRIER',seed,current,epoch,'points',completed_points,flush=True)
            assert not pending
            witness={'task':current,'batches':100*len(dl),'loss_sequence_sha256':trace.hexdigest(),
                'model_state_sha256':state_digest(algo.policy.state_dict()),'optimizer_sha256':state_digest(algo.optimizer.state_dict()),
                'scheduler_sha256':state_digest(algo.scheduler.state_dict()),'rng_sha256':state_digest(historical_rng()),
                'optimizer_step':algo._optimizer_step_count}
            write(out/f'task{current}_WITNESS.json',witness)
            if seed==45 and current==6:
                expected=read(OUT/'recovery_probe_seed45/WITNESS_COMPARISON.json')['rebuilt']
                assert witness==expected,'Current GPU native carrier differs from its independent no-diagnostic repeat'
            # Historical comparison is descriptive only for the explicitly new carrier.
            old=torch.load(ROOT/f'runs/fixed_kappa_confirmation_v2/formal/seed{seed}/mlr/run_001/mlrifa_{current}_model_ep99.pth',map_location='cpu')['state_dict']
            model=cpu_state(algo.policy.state_dict());maxdiff=max(float((v-old[k]).abs().max()) for k,v in model.items())
            write(out/f'task{current}_HISTORICAL_COMPARISON.json',{'max_abs_difference':maxdiff,'same_historical_trajectory':False,
                'not_an_acceptance_test':'explicitly new controlled 3090 carrier; no history substitution'})
            del ds,old,model
        assert completed_points==40
        for stream in streams.values():stream.close()
        write(out/'COMPLETE.json',{'status':'PASS','seed':seed,'carrier_source':'new_controlled_3090_from_official',
            'points_completed':completed_points,'native_steps':global_step,'time':time.time(),'wall_seconds':time.monotonic()-begin,
            'initial_model_sha256':initial_sha,'all_live_carrier_11_checks_passed':True,
            'files':{p.name:sha(p) for p in out.glob('*.jsonl')}})
        torch.distributed.destroy_process_group()
    except Exception:
        write(out/'FAILED.json',{'time':time.time(),'points_completed':completed_points,'traceback':traceback.format_exc()});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();run(a.seed)
