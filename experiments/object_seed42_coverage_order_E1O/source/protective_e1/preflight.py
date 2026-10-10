"""Five isolated real GPU optimizer calls; no environments or rollouts."""
import copy,hashlib,json,time
from pathlib import Path
import torch
import torch.distributed as dist
from easydict import EasyDict
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .prepare import O,S,config
from .train import build
from .runtime import capture,restore_rng,rng
from .coverage_order import CoveragePlan,CoverageOrder,OrderConfig,PlannedBatchSampler
from .identity import IdentityDataset
from .io import write,sha,save
from libero.lifelong.algos.mlr import MLR,build_dataset_for_task,update_buffer_equal_quota_10,BUFFER_CAPACITY,build_ifa_similarity_matrix
from libero.lifelong.utils import torch_load_model

def signature(value):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):h.update(str((x.dtype,tuple(x.shape))).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,dict):
            for k,v in x.items():h.update(str(k).encode());visit(v)
        elif isinstance(x,(list,tuple)):
            for v in x:visit(v)
        elif hasattr(x,'tobytes'):h.update(x.tobytes())
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()

def restore(algo,state):
    algo.policy.load_state_dict(state['model'],strict=True);algo.optimizer.load_state_dict(state['optimizer'])
    algo.scheduler.load_state_dict(state['scheduler']);algo._optimizer_step_count=state['optimizer_step'];algo._scheduler_step_count=state['scheduler_step']
    for n,m in algo.policy.named_modules():m.training=state['modes'][n]
    for n,p in algo.policy.named_parameters():p.grad=None if state['grads'][n] is None else state['grads'][n].to(p.device)
    restore_rng(state['rng'])

def main():
    root=O/'preflight';assert not root.exists(),'Do not silently repeat GPU checks'
    root.mkdir();dist.init_process_group('nccl');torch.cuda.set_device(0);begin=time.time()
    try:
        cfg=EasyDict(config(root));algo,benchmark=build(cfg)
        pretrained=torch_load_model(cfg.pretrain)[0]
        trainable={n:p for n,p in algo.policy.module.named_parameters() if p.requires_grad}
        assert all(n in pretrained and torch.equal(p.detach().cpu(),pretrained[n].cpu()) for n,p in trainable.items())
        loaded_count=len(trainable);del pretrained
        update_buffer_equal_quota_10(cfg.feature_dir,6,cfg.buffer_dir,capacity=BUFFER_CAPACITY,seed=42,drop_policy='random',rate=.5)
        old={6:[]};current=[];algo.catalog={};algo.stage=7
        for p in sorted(Path(cfg.buffer_dir).glob('*.pt')):
            key='buffer/'+p.name;old[6].append(key);algo.catalog[key]=dict(path=str(p),sha256=sha(p),task=6)
        for p in sorted((Path(cfg.feature_dir)/'7').glob('*.pt')):
            key='current/7/'+p.name;current.append(key);algo.catalog[key]=dict(path=str(p),sha256=sha(p),task=7)
        dataset=build_dataset_for_task(cfg.feature_dir,7,cfg.buffer_dir,42,start=6)
        sampler=DistributedSampler(dataset,num_replicas=1,rank=0,shuffle=True)
        native_loader=DataLoader(dataset,batch_size=10,num_workers=0,sampler=sampler)
        algo.configure_plan(IdentityDataset(dataset));plan=algo.plan
        loader=DataLoader(IdentityDataset(dataset),num_workers=0,batch_sampler=PlannedBatchSampler(plan))
        algo.start_task(-1,loader);algo.policy.train();base_rng=copy.deepcopy(rng())
        sampler.set_epoch(0);native=next(iter(native_loader));native_rng=signature(rng())
        restore_rng(base_rng);sampler.set_epoch(0);plan.begin_epoch(list(iter(sampler)),0)
        iterator=iter(loader);batch=next(iterator)
        assert batch[1]==native[1] and all(torch.equal(batch[0][k],native[0][k]) for k in native[0])
        assert native_rng==signature(rng()),'Planned loader changes original model RNG'
        algo.order=CoverageOrder(algo.policy,42,7,old,current,algo.payload,'preflight',OrderConfig(enabled=False));algo.order_stage_step=0
        ifa=build_ifa_similarity_matrix(current_task_id=7,current_feat_root=cfg.feature_dir,replay_dir=cfg.buffer_dir,language_idx=0,agent_idx=1,top_ratio=.5)
        initial=capture(algo)
        native_loss=MLR.observe(algo,copy.deepcopy(native),7,0,ifa_info=ifa);expected=capture(algo)
        restore(algo,initial);off_loss=algo.observe(copy.deepcopy(batch),7,0,ifa_info=ifa);actual=capture(algo)
        assert native_loss==off_loss and signature(expected)==signature(actual),'Disabled native update differs'
        algo.plan=CoveragePlan(plan.tasks,plan.ids,10,7);plan=algo.plan
        sampler.set_epoch(0);plan.begin_epoch(list(iter(sampler)),0)
        iterator=iter(DataLoader(IdentityDataset(dataset),num_workers=0,batch_sampler=PlannedBatchSampler(plan)))
        algo.order=CoverageOrder(algo.policy,42,7,old,current,algo.payload,'preflight');algo.order_stage_step=0
        before=capture(algo);event=algo.order.schedule(plan,0);after=capture(algo)
        assert signature(before)==signature(after),'Scoring changed params/buffers/optimizer/scheduler/grads/RNG/modes'
        assert event['reason']=='OK' and event['tasks']['6']['committed']
        assert algo.order.totals['cross_batch_moved']>0
        batch=next(iterator)
        for slot,task in enumerate(batch[1]):
            if int(task)==7 or slot==0:assert batch[2][slot]==plan.ids[plan.base[slot]]
        algo.observe(copy.deepcopy(batch),7,0,ifa_info=ifa) # GPU call3
        # Read remaining payloads without optimizing. This validates an epoch
        # cursor boundary, NOT a fully trained GPU epoch.
        for batch in iterator:plan.record_read(batch[2],batch[1],plan.last_batch[0]//10)
        evidence=plan.finish();algo.order_stage_step=len(plan);algo.order.events=[]
        boundary=dict(state=capture(algo),side=algo.order_sidecar());save(root/'boundary.pth',boundary)
        def next_epoch():
            sampler.set_epoch(1);plan.begin_epoch(list(iter(sampler)),1)
            it=iter(DataLoader(IdentityDataset(dataset),num_workers=0,batch_sampler=PlannedBatchSampler(plan)))
            algo.order.schedule(plan,algo.order_stage_step);return next(it)
        first=next_epoch();loss=algo.observe(copy.deepcopy(first),7,1,ifa_info=ifa);target=capture(algo);target_order=algo.order.state_dict()
        restored=torch.load(root/'boundary.pth',map_location='cpu');restore(algo,restored['state']);algo.restore_order_sidecar(restored['side'])
        algo.plan=CoveragePlan(plan.tasks,plan.ids,10,7);plan=algo.plan
        second=next_epoch();loss2=algo.observe(copy.deepcopy(second),7,1,ifa_info=ifa)
        assert first[2]==second[2] and loss==loss2 and signature(target)==signature(capture(algo))
        assert signature(target_order)==signature(algo.order.state_dict())
        assert signature(initial['model'])!=signature(target['model'])
        result=dict(status='PASS',optimizer_calls=5,real_payloads=True,formal_state_reused=False,rollouts=0,
          old_buffer_windows=len(old[6]),current_windows=len(current),trainable_pretrain_keys_loaded=loaded_count,
          disabled_native_update_exact=True,loader_payload_rng_parity=True,actual_ids_match_plan=True,
          complete_readonly_epoch_multiplicity=True,read_only_epoch_is_not_training=True,
          scoring_isolation_includes_parameters_buffers_optimizer_scheduler_grads_rng_modes=True,
          saved_epoch_boundary_next_update_exact=True,pinned_first_and_current=True,
          cross_batch_moved=algo.order.totals['cross_batch_moved'],ifa_pairs=ifa['selected_pairs'],
          seconds=time.time()-begin,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),schedule=event)
        write(root/'coverage_readcheck.json',evidence);write(O/'REAL_PREFLIGHT.json',result)
        print(json.dumps(result),flush=True)
    finally:dist.destroy_process_group()

if __name__=='__main__':main()
