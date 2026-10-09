"""Short checks using actual recovered OBJECT payloads and the actual E0 policy."""
import copy
import hashlib
import json
import os
import time
from pathlib import Path
import torch
import torch.distributed as dist
from easydict import EasyDict
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .prepare import O,S,config
from .train import build
from .runtime import capture,restore_rng,rng
from .io import write,sha,save
from libero.lifelong.algos.mlr import MLR,build_dataset_for_task,update_buffer_equal_quota_10,BUFFER_CAPACITY,build_ifa_similarity_matrix
from libero.lifelong.utils import torch_load_model
from libero.lifelong.replay_selection import SelectionConfig

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
    restore_rng(state['rng'])

def main():
    root=O/'preflight';assert not root.exists(),'Do not rerun existing real checks silently'
    root.mkdir();dist.init_process_group('nccl');torch.cuda.set_device(0)
    begin=time.time()
    try:
        cfg=EasyDict(config(root));algo,benchmark=build(cfg)
        pretrained=torch_load_model(cfg.pretrain)[0]
        trainable={n:p for n,p in algo.policy.module.named_parameters() if p.requires_grad}
        assert trainable and all(n in pretrained and torch.equal(p.detach().cpu(),pretrained[n].cpu()) for n,p in trainable.items())
        loaded_count=len(trainable);del pretrained
        update_buffer_equal_quota_10(cfg.feature_dir,6,cfg.buffer_dir,capacity=BUFFER_CAPACITY,seed=42,drop_policy='random',rate=.5)
        old={6:[]};current=[];catalog={}
        for p in sorted(Path(cfg.buffer_dir).glob('*.pt')):
            old[6].append(p.name);catalog[p.name]=dict(path=str(p),sha256=sha(p))
        for p in sorted((Path(cfg.feature_dir)/'7').glob('*.pt')):
            key='7/'+p.name;current.append(key);catalog[key]=dict(path=str(p),sha256=sha(p))
        algo.catalog=catalog;algo.stage=7;algo.configure_e1_stage(7,old,current,algo.payload,signature(catalog),SelectionConfig(pool_weighting='uniform',priority_replace_probability=0.))
        dataset=build_dataset_for_task(cfg.feature_dir,7,cfg.buffer_dir,42,start=6)
        loader=DataLoader(dataset,batch_size=10,num_workers=0,sampler=DistributedSampler(dataset,num_replicas=1,rank=0,shuffle=True))
        algo.start_task(-1,loader);algo.policy.train();iterator=iter(loader);batch=next(iterator);second=next(iterator)
        ifa=build_ifa_similarity_matrix(current_task_id=7,current_feat_root=cfg.feature_dir,replay_dir=cfg.buffer_dir,language_idx=0,agent_idx=1,top_ratio=.5)
        initial=capture(algo)
        native_loss=MLR.observe(algo,copy.deepcopy(batch),7,0,ifa_info=ifa);native=capture(algo)
        restore(algo,initial);e1_loss=algo.observe(copy.deepcopy(batch),7,0,ifa_info=ifa);off=capture(algo)
        assert native_loss==e1_loss and signature(native)==signature(off),'ZERO_PROBABILITY_UPDATE_DIVERGED'
        restore(algo,initial);algo.configure_e1_stage(7,old,current,algo.payload,signature(catalog),SelectionConfig(pool_weighting='uniform'))
        before=capture(algo);refresh=algo.e1_selector.refresh(0);after=capture(algo)
        assert signature(before)==signature(after),'SCORING_CHANGED_TRAINING_STATE'
        assert refresh['reason']=='OK' and len(algo.e1_selector.distributions[6]['ids'])==64
        assert refresh['probabilities']['6'] == [1./64]*64
        replacement,events=algo.e1_selector.replace_old_slots(copy.deepcopy(batch))
        assert replacement[1]==batch[1]
        for i,t in enumerate(batch[1]):
            if int(t)==7:
                for k in batch[0]:assert torch.equal(replacement[0][k][i],batch[0][k][i])
        algo.observe(copy.deepcopy(batch),7,0,ifa_info=ifa)
        middle=capture(algo);side=algo.e1_sidecar()
        save(root/'resume_check.pth',dict(state=middle,side=side))
        expected_loss=algo.observe(copy.deepcopy(second),7,0,ifa_info=ifa);expected=capture(algo)
        restored=torch.load(root/'resume_check.pth',map_location='cpu');restore(algo,restored['state']);algo.restore_e1_sidecar(restored['side'])
        restored_loss=algo.observe(copy.deepcopy(second),7,0,ifa_info=ifa);actual=capture(algo)
        assert expected_loss==restored_loss and signature(expected)==signature(actual),'RESUME_NEXT_UPDATE_DIVERGED'
        assert signature(initial['model'])!=signature(actual['model']),'NONZERO_LR_DID_NOT_UPDATE'
        from .identity import IdentityDataset
        saved_rng=copy.deepcopy(rng());loader.sampler.set_epoch(1);a=next(iter(loader));rng_a=signature(rng())
        restore_rng(saved_rng);loader.sampler.set_epoch(1);b=next(iter(loader))
        assert a[1]==b[1] and all(torch.equal(a[0][k],b[0][k]) for k in a[0]) and rng_a==signature(rng())
        restore_rng(saved_rng)
        identity_loader=DataLoader(IdentityDataset(dataset),batch_size=10,num_workers=0,sampler=DistributedSampler(dataset,num_replicas=1,rank=0,shuffle=True))
        identity_loader.sampler.set_epoch(1);c=next(iter(identity_loader))
        assert a[1]==c[1] and all(torch.equal(a[0][k],c[0][k]) for k in a[0]) and rng_a==signature(rng())
        assert len(c[2])==10 and all(k.startswith(('buffer/','current/')) for k in c[2])
        result=dict(identity_sidechannel_loader_payload_rng_parity=True,pool_weighting='uniform',status='PASS',optimizer_calls=5,actual_policy='E0 BriefPolicy/GMMHead',real_payloads=True,
          old_buffer_windows=len(old[6]),current_windows=len(current),zero_probability_native_update_parity=True,
          refresh_preserves_parameters_optimizer_scheduler_rng_modes=True,only_head_gradient=True,
          candidate_count=64,reference_count=16,current_reference_count=16,same_task_slots_current_content=True,
          saved_state_next_update_exact=True,next_epoch_loader_rng_exact=True,trainable_pretrain_keys_loaded=loaded_count,
          ifa_pairs=ifa['selected_pairs'],formal_state_reused=False,seconds=time.time()-begin,
          torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),refresh=refresh)
        write(O/'REAL_PREFLIGHT.json',result);print(json.dumps(result),flush=True)
    finally:dist.destroy_process_group()

if __name__=='__main__':main()
