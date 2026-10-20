"""Four isolated native updates and one zero-action environment step; no rollout."""
import copy
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import robomimic.utils.tensor_utils as TensorUtils
from libero.lifelong.models.complete_policy import CompletePolicy
from libero.lifelong.metric import OffScreenRenderEnv
from libero.lifelong.utils import make_linear_schedule_with_warmup,control_seed
from .common import O,SEED,verify,write,capture,restore
from .data import cfg,lifelong_datasets,tasks,native_task
from .train import build
from .grouped_ifa import loss

def main():
    verify();assert not (O/"GPU_PREFLIGHT.json").exists()
    assert torch.cuda.is_available()
    control_seed(SEED);torch.cuda.set_device(0)
    c=cfg("export");datasets,emb,shape=lifelong_datasets(c);c.shape_meta=shape
    policy=CompletePolicy(c,shape).cuda().eval()
    batches=[]
    for t in (0,1):
        data=next(iter(DataLoader(datasets[t],batch_size=10,shuffle=False,num_workers=0)))
        data=TensorUtils.map_tensor(data,lambda x:x.cuda())
        with torch.no_grad():
            native=policy.preprocess_input(copy.deepcopy(data),train_mode=False)
            feature=policy.spatial_encode(native)
        assert feature.shape==(10,8,4,512)
        batches.append((feature.detach(),data["actions"].detach(),t))
    dist.init_process_group("nccl")
    try:
        c=cfg("train");model=build(c);model.eval()
        for h in (1,4,8):
            with torch.no_grad():
                x=batches[0][0][:,:h]
                a=policy.policy_head.forward_fn(policy.temporal_encode(x)[:,-1])
                b=model.module.policy_head.forward_fn(model.module.temporal_encode(x)[:,-1])
            assert all(torch.equal(u,v) for u,v in zip(a,b)),h
        del policy;torch.cuda.empty_cache()
        opt=torch.optim.AdamW(model.parameters(),**c.train.optimizer.kwargs)
        scheduler=torch.optim.lr_scheduler.LambdaLR(opt,make_linear_schedule_with_warmup(500,1000))
        refs={t:x[0,0,0].detach() for x,target,t in batches}
        info=dict(current_tasks=list(range(5)),selected_pairs=[(0,1)])
        before=capture(model,opt,scheduler)
        def two():
            result=[];start=time.monotonic()
            for x,target,t in batches:
                model.train();opt.zero_grad()
                distribution,last=model(x)
                value=model.module.policy_head.loss_fn(distribution,target)+.1*loss(
                    last,torch.full((len(x),),t,device=x.device),info,refs,.1)
                assert torch.isfinite(value)
                (c.train.loss_scale*value).backward()
                gs=[p.grad for p in model.parameters() if p.grad is not None]
                assert gs and all(torch.isfinite(g).all() for g in gs)
                assert sum(float(g.abs().sum()) for g in gs)>0
                if c.train.grad_clip is not None:torch.nn.utils.clip_grad_norm_(model.parameters(),c.train.grad_clip)
                opt.step();scheduler.step();result.append(float(value.detach()))
            torch.cuda.synchronize()
            return result,capture(model,opt,scheduler),time.monotonic()-start
        a,after,seconds=two();restore(model,opt,scheduler,before);b,replay,seconds2=two()
        assert a==b
        assert all(torch.equal(after["model"][k],replay["model"][k]) for k in after["model"])
        assert any(not torch.equal(before["model"][k],after["model"][k]) for k in after["model"])
    finally:dist.destroy_process_group()
    task=native_task(tasks()[0]);c=cfg("eval")
    env=OffScreenRenderEnv(bddl_file_name=str(Path(c.bddl_folder)/task.problem_folder/task.bddl_file),
        camera_heights=c.data.img_h,camera_widths=c.data.img_w)
    try:
        env.seed(SEED);env.reset();env.seed(SEED)
        initial=torch.load(Path(c.init_states_folder)/task.problem_folder/task.init_states_file)
        assert len(initial)>=20
        obs=env.set_init_state(initial[0]);obs,reward,done,info=env.step(np.zeros(7))
        assert all(np.isfinite(np.asarray(obs[k])).all() for k in ("robot0_joint_pos","robot0_gripper_qpos"))
    finally:env.close()
    verify()
    write(O/"GPU_PREFLIGHT.json",dict(status="PASS",optimizer_calls=4,formal_updates=0,
        environment_steps=1,policy_rollouts=0,restore_tensor_exact=True,
        native_gmm_history_lengths=[1,4,8],batch_size=10,fp32=True,
        measured_two_update_seconds=[seconds,seconds2],gpu=torch.cuda.get_device_name()))
if __name__=="__main__":main()
