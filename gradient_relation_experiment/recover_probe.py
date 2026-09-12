"""Rebuild a historical MLR task6 boundary and require every stored witness."""
import argparse
import contextlib
import hashlib
import io
import struct
import time
import traceback

from cb_ifa_experiment.gpu_runtime import *
from confirmation_experiment.common import state_digest, rng_state as historical_rng

OUT=ROOT/'reports/gradient_relation_v1'


def run(seed):
    dest=OUT/f'recovery_probe_seed{seed}'
    dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'STARTED.json').exists()
    write(dest/'STARTED.json',{'time':time.time(),'pid':os.getpid(),'seed':seed,'scope':'historical state recovery only'})
    try:
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        init_ddp(dest/'ddp_store')
        historical=ROOT/f'runs/fixed_kappa_confirmation_v2/formal/seed{seed}/mlr'
        plan=read(historical/'plans/task6.json')
        ds=CachedWindows(plan['immutable_feature_origins'],plan['record_sha256'])
        sampler=torch.utils.data.DistributedSampler(ds,**{k:plan['sampler'][k] for k in ('num_replicas','rank','shuffle','seed','drop_last')})
        dl=torch.utils.data.DataLoader(ds,batch_size=plan['batch_size'],sampler=sampler,num_workers=0)
        cfg=train_config(seed,dest);bootstrap_train(cfg);algo=build_train(cfg)
        with contextlib.redirect_stdout(io.StringIO()):algo.start_task(-1,dl)
        trace=hashlib.sha256();count=0;begin=time.monotonic()
        for epoch in range(100):
            sampler.set_epoch(epoch);algo.policy.train()
            for batch in dl:
                with contextlib.redirect_stdout(io.StringIO()):loss=algo.observe(batch,6,epoch,None)
                assert np.isfinite(loss)
                trace.update(struct.pack('!d',float(loss)));count+=1
            if (epoch%5==0 and epoch>69) or epoch==99:algo.policy.eval()
            algo.epoch_schedule_step(epoch)
            write(dest/'STATUS.json',{'epoch':epoch,'optimizer_steps':count,'elapsed_seconds':time.monotonic()-begin})
            if epoch%10==9:print('RECOVERY',seed,epoch,count,flush=True)
        rebuilt={'task':6,'batches':count,'loss_sequence_sha256':trace.hexdigest(),
                 'model_state_sha256':state_digest(algo.policy.state_dict()),
                 'optimizer_sha256':state_digest(algo.optimizer.state_dict()),
                 'scheduler_sha256':state_digest(algo.scheduler.state_dict()),
                 'rng_sha256':state_digest(historical_rng()),'optimizer_step':algo._optimizer_step_count}
        expected=read(historical/'TASK6_TRAIN_READY.json')
        comparison={k:rebuilt[k]==v for k,v in expected.items()}
        old=torch.load(historical/'run_001/mlrifa_6_model_ep99.pth',map_location='cpu')['state_dict']
        model=cpu_state(algo.policy.state_dict())
        maxdiff=max(float((v-old[k]).abs().max()) for k,v in model.items() if v.numel())
        write(dest/'WITNESS_COMPARISON.json',{'expected':expected,'rebuilt':rebuilt,'equal':comparison,'model_max_abs_difference':maxdiff})
        assert all(comparison.values()),'Historical boundary mismatch; not permitted to label this a faithful recovery'
        atomic_save(dest/'BOUNDARY.pt',{'model':model,'optimizer':cpu_state(algo.optimizer.state_dict()),
            'scheduler':algo.scheduler.state_dict(),'rng':cpu_state(rng_state()),
            'optimizer_step':algo._optimizer_step_count,'scheduler_step':algo._scheduler_step_count,
            'training_modes':{n:m.training for n,m in algo.policy.named_modules()},'seed':seed,
            'boundary':'task6 native end, exact complete witness matched; before task7 optimizer initialization',
            'stage_plan_sha256':sha(historical/'plans/task6.json'),'witness':rebuilt})
        write(dest/'COMPLETE.json',{'status':'PASS','seed':seed,'full_historical_witness_bitwise_equal':True,
            'elapsed_seconds':time.monotonic()-begin,'state_sha256':sha(dest/'BOUNDARY.pt')})
        torch.distributed.destroy_process_group()
    except Exception:
        write(dest/'FAILED.json',{'time':time.time(),'traceback':traceback.format_exc()});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();run(a.seed)
