import copy,json,os,time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import robomimic.utils.tensor_utils as TU
from .common import O,write,save,sha,verify,rng,restore_rng
from .model import cfg,data_setup,complete,strict_downstream,angular_mean
from .train import build,capture,restore,cpu
from libero.lifelong.models.brief import BriefPolicy
from libero.lifelong.utils import control_seed

def main():
    verify();assert not (O/'GPU_PREFLIGHT.json').exists(),'Do not repeat successful preflight'
    start=time.time();assert torch.cuda.is_available();control_seed(42)
    c=cfg('export');b,ds,shape=data_setup(c,True);c.shape_meta=shape
    p=complete(c,shape);p.eval();tests={}
    raw=next(iter(DataLoader(ds[6],batch_size=1,num_workers=0)))
    data=TU.map_tensor(raw,lambda x:x.cuda());prepared=p.preprocess_input(copy.deepcopy(data))
    captured={};hooks=[]
    for name in ('state_encoder','state_scale','state_shift'):
        def hook(module,args,value,name=name):captured[name]=value.detach().clone()
        hooks.append(getattr(p,name).register_forward_hook(hook))
    with torch.no_grad():x=p.spatial_encode(copy.deepcopy(prepared))
    expected=((1+captured['state_scale'])*captured['state_encoder']+captured['state_shift']).reshape(1,8,512)
    assert torch.equal(x[:,:,3,:],expected);tests['state_token_exact']=True
    for h in hooks:h.remove()
    changed=copy.deepcopy(prepared);changed['obs']['joint_states']+=.37;changed['obs']['gripper_states']-=.21
    with torch.no_grad():y=p.spatial_encode(changed)
    assert torch.equal(x[:,:,:3],y[:,:,:3]);delta=float((x[:,:,3]-y[:,:,3]).abs().max());assert delta>1e-5
    tests['state_response_max']=delta;tests['other_tokens_invariant']=True
    p.cfg.policy.use_state=False
    with torch.no_grad():f=p.spatial_encode(copy.deepcopy(prepared));g=p.spatial_encode(copy.deepcopy(changed))
    assert torch.equal(f,g);tests['false_negative_control']=True;p.cfg.policy.use_state=True
    bc=cfg('random');brief=BriefPolicy(bc,None).cuda().eval();strict_downstream(brief,bc.pretrain)
    with torch.no_grad():
        left=p.policy_head.forward_fn(p.temporal_encode(x)[:,-1]);right=brief.policy_head.forward_fn(brief.temporal_encode(x)[:,-1])
    assert all(torch.equal(a,b_) for a,b_ in zip(left,right));tests['gmm_parameters_exact']=True
    payload=[]
    with torch.no_grad():
        for task in (6,7):
            for index in range(5):
                row=next(iter(DataLoader(torch.utils.data.Subset(ds[task],[index]),batch_size=1,num_workers=0)))
                row=TU.map_tensor(row,lambda x:x.cuda());latent=p.spatial_encode(p.preprocess_input(row))
                payload.append((dict(feature=latent.cpu(),actions=row['actions'].cpu()),str(task)))
    torch.save(payload,O/'preflight_real_payload.pth')
    # Render/seed/initial-state interface check only: one env, one zero-action physics step; no policy rollout.
    from libero.libero.envs import OffScreenRenderEnv,SubprocVectorEnv
    task=b.get_task(6)
    kwargs=dict(bddl_file_name=str(Path(c.bddl_folder)/task.problem_folder/task.bddl_file),camera_heights=128,camera_widths=128)
    env=SubprocVectorEnv([lambda:OffScreenRenderEnv(**kwargs)])
    try:
        env.seed(42);env.reset();init=torch.load(Path(c.init_states_folder)/task.problem_folder/task.init_states_file)
        assert len(init)>=20;env.set_init_state(init[:1]);obs,*_=env.step(__import__('numpy').zeros((1,7)))
        assert obs[0]['agentview_image'].shape==(128,128,3);tests['spawn_render_reset_one_zero_step']=True
    finally:env.close()
    del p,brief;torch.cuda.empty_cache()
    # Actual native-size latent batch, corrected shared IFA, four isolated optimizer calls.
    dist.init_process_group('nccl');torch.cuda.set_device(0)
    try:
        bc.run_root=str(O/'smoke');bc.experiment_dir=str(O/'smoke/models');bc.buffer_dir=str(O/'smoke/buffer')
        a,_=build(bc);loader=DataLoader(payload,batch_size=10,num_workers=0);a.start_task(-1,loader)
        a.refs={6:payload[0][0]['feature'][0,0,0].cuda(),7:payload[5][0]['feature'][0,0,0].cuda()}
        info={'selected_pairs':[(6,7)]}
        a.observe(next(iter(loader)),7,0,info);a.observe(next(iter(loader)),7,0,info)
        snap=capture(a);save(O/'smoke/recovery_test.pth',snap)
        first=a.observe(next(iter(loader)),7,0,info);expect=cpu(a.policy.state_dict())
        restore(a,torch.load(O/'smoke/recovery_test.pth',map_location='cpu'))
        second=a.observe(next(iter(loader)),7,0,info)
        assert first==second and all(torch.equal(v,expect[k]) for k,v in cpu(a.policy.state_dict()).items())
        tests['optimizer_scheduler_rng_recovery_exact']=True;tests['isolated_optimizer_calls']=4
        assert a.scheduler.last_epoch==3 and a.step==3
        g=torch.randn(3,512,device='cuda');own=a.refs[7];old=a.refs[6]
        one=angular_mean(g,own,[old],.3);two=angular_mean(g,own,[old,old],.3)
        assert torch.equal(one,two);assert angular_mean(g,own,[],.3).item()==0
        tests['ifa_pair_mean_and_empty']=True
    finally:dist.destroy_process_group()
    verify();write(O/'GPU_PREFLIGHT.json',dict(status='PASS',tests=tests,seconds=time.time()-start,
        formal_rollouts=0,environment_interface_checks=1,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        modal_sha256=sha(O/'MODAL_CONTRACT.json'),contract_sha256=sha(O/'CONTRACT.json')))
if __name__=='__main__':main()
