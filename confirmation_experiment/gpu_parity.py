"""Real GPU on/off observer parity, snapshot restoration, and timing.

Historical seed42 only. Full strict equality is required; never relax tolerances
after inspecting new-seed outcomes. No inference or formal confirmation run.
"""
import copy
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from easydict import EasyDict
from torch.utils.data import DataLoader,DistributedSampler
from libero.lifelong.algos.mlr import MLR,FileTrajectoryDataset
from libero.lifelong.algos.mlr_fixed_kappa import MLRFixedKappa
from confirmation_experiment.instrumentation import ConfirmationMLR,ConfirmationSoft
from confirmation_experiment.common import ROOT,REPORTS,read,write,rng_state,restore_rng,exact,state_to_cpu

def capture(obj):
    return {'model':state_to_cpu(obj.policy.state_dict()),'optimizer':state_to_cpu(obj.optimizer.state_dict()),
            'scheduler':state_to_cpu(obj.scheduler.state_dict()),'grads':[state_to_cpu(p.grad) for p in obj.policy.parameters()],
            'rng':rng_state(),'steps':(obj._optimizer_step_count,obj._scheduler_step_count)}

def run():
    torch.set_num_threads(2)
    out=REPORTS/'gpu_parity_final';out.mkdir(parents=True,exist_ok=False)
    os.environ.update(CONFIRMATION_BARRIERS='0',CONFIRMATION_EVENT_LOGGING='1',CONFIRMATION_SNAPSHOTS='1')
    cfg=EasyDict(read(ROOT/'runs/final_paper_text_faithful/mlr/seed42/run_001/config.json'))
    cfg.device='cuda:0';cfg.ifa_enabled=False;cfg.ifa_interaction_logging=False
    cfg.lifelong=EasyDict(algo='MLRFixedKappa',eval_in_train=False,projection_enabled=True,
                         reference_norm_floor=1e-12,synchronize_timing=True,step_log_every=0,kappa=.5)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    files=[str(p) for t in (6,7) for p in sorted((ROOT/f'features/object_seed42/{t}').glob('*.pt'))]
    ds=FileTrajectoryDataset(files);sampler=DistributedSampler(ds,num_replicas=1,rank=0,shuffle=True)
    loader=DataLoader(ds,batch_size=10,num_workers=0,sampler=sampler)
    sampler.set_epoch(35)
    batches=[]
    for i,b in enumerate(loader):
        if i==128:break
        batches.append(b)
    checkpoint=ROOT/'runs/fixed_kappa/fixed_v1/formal/kappa_0p5/seed42/run_001/mlrifa_7_model_ep95.pth'
    loaded=torch.load(checkpoint,map_location='cpu')['state_dict']
    weights={k[7:] if k.startswith('module.') else k:v for k,v in loaded.items()}
    def build(cls,name):
        c=copy.deepcopy(cfg);c.experiment_dir=str(out/name/'run_001');Path(c.experiment_dir).mkdir(parents=True)
        obj=cls(1,c).to('cuda:0');obj.policy.load_state_dict(weights,strict=True)
        if hasattr(obj,'_confirm_stage'):obj._confirm_stage=7
        obj.start_task(-1,loader);obj.policy.train()
        return obj
    results=[]
    for native_cls,wrapped_cls,label in ((MLR,ConfirmationMLR,'mlr'),(MLRFixedKappa,ConfirmationSoft,'soft')):
        a=build(native_cls,label+'_native');b=build(wrapped_cls,label+'_observer')
        off=build(wrapped_cls,label+'_observer_disabled')
        off.event_logging=False;off.snapshot_logging=False
        steps=8 if label=='mlr' else 128
        triggered=0;snapshot_expected={};timings={'native':0.,'observed':0.,'observer_disabled':0.}
        for i,batch in enumerate(batches[:steps]):
            before=rng_state();torch.cuda.synchronize();start=time.perf_counter()
            la=a.observe(copy.deepcopy(batch),7,35,None);torch.cuda.synchronize();timings['native']+=time.perf_counter()-start
            after=rng_state();restore_rng(before)
            torch.cuda.synchronize();start=time.perf_counter()
            lb=b.observe(copy.deepcopy(batch),7,35,None);torch.cuda.synchronize();timings['observed']+=time.perf_counter()-start
            exact(la,lb,'loss');exact(after,rng_state(),'rng')
            exact(a.policy.state_dict(),b.policy.state_dict(),'parameters')
            exact(a.optimizer.state_dict(),b.optimizer.state_dict(),'optimizer')
            exact(a.scheduler.state_dict(),b.scheduler.state_dict(),'scheduler')
            for p,q in zip(a.policy.parameters(),b.policy.parameters()):exact(p.grad,q.grad,'gradient')
            restore_rng(before);torch.cuda.synchronize();start=time.perf_counter()
            lo=off.observe(copy.deepcopy(batch),7,35,None);torch.cuda.synchronize()
            timings['observer_disabled']+=time.perf_counter()-start
            exact(la,lo,'disabled observer loss');exact(after,rng_state(),'disabled observer rng')
            exact(a.policy.state_dict(),off.policy.state_dict(),'disabled observer parameters')
            exact(a.optimizer.state_dict(),off.optimizer.state_dict(),'disabled observer optimizer')
            exact(a.scheduler.state_dict(),off.scheduler.state_dict(),'disabled observer scheduler')
            for p,q in zip(a.policy.parameters(),off.policy.parameters()):exact(p.grad,q.grad,'disabled observer gradient')
            if label=='soft':
                triggered+=int(b._rc_rows[-1]['projected'])
                for row in b._confirm_snapshot_rows:
                    if row['slot'] not in snapshot_expected:snapshot_expected[row['slot']]=(row,capture(b),lb)
        if b._confirm_events:b._confirm_events.flush()
        restored=[]
        for slot,(row,expected,expected_loss) in snapshot_expected.items():
            snap=torch.load(ROOT/row['path'],map_location='cpu')
            c=build(MLRFixedKappa,label+'_restore_'+slot)
            c.policy.load_state_dict(snap['model'],strict=True);c.optimizer.load_state_dict(snap['optimizer'])
            c.scheduler.load_state_dict(snap['scheduler'])
            c._optimizer_step_count=snap['optimizer_step'];c._scheduler_step_count=snap['scheduler_step']
            for name,module in c.policy.named_modules():module.training=snap['training_modes'][name]
            restore_rng(snap['rng']);lc=c.observe(copy.deepcopy(snap['batch']),7,snap['epoch'],None)
            exact(expected_loss,lc,'snapshot restored loss');exact(expected,capture(c),'snapshot next step')
            restored.append(slot);del c
        if label=='soft':
            assert triggered>0,'No real conflict exercised; cannot certify triggered logging path'
            assert any('trigger_bin' in x for x in restored),'Real trigger snapshot must be restored'
            assert any('ordinary_nontrigger' in x for x in restored),'Ordinary nontrigger snapshot must be restored'
        results.append({'arm':label,'steps':steps,'all_compared_states_bitwise_equal':True,'triggered_steps':triggered,
                        'comparison':'Untouched native vs observer flag off vs observer flag on with snapshot capture',
                        'restored_snapshot_slots':restored,'timings_seconds':timings,
                        'timing_scope':'Native observe vs observed observe including snapshot writes; not pure production algorithm cost.'})
        del a,b,off;torch.cuda.empty_cache()
    write(REPORTS/'GPU_PARITY.json',{'status':'PASS','comparisons':results,'torch':torch.__version__,
                                   'gpu':torch.cuda.get_device_name(0),'seed':42,'maximum_abs_difference':0.,
                                   'new_seed_training_or_selection':False})
    print(json.dumps(results,indent=2))
if __name__=='__main__':run()
