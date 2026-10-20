"""Fixed 20-episode paired teacher screen. No optimizer or residual training."""
import os,sys,json,time,hashlib,copy,subprocess
from pathlib import Path
import numpy as np
import torch
W=Path("/data/mlr_ifa/experiment_workspaces/OVR_OBJECT_RA_seed42_screen_20261020")
O=Path("/data/mlr_ifa/experiments/OVR_OBJECT_RA_seed42_screen_20261020")
P=Path("/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed42_20261019")
S=Path("/data/mlr_ifa/experiment_workspaces/RA_OBJECT_conditional_20261019/seed42/source")
A=Path("/data/mlr_ifa/analyses/OVR_OBJECT_seed42_20261020_admission_v3/RESOURCE_AUDIT.json")
sys.path.insert(0,str(S))
from coreset_exp.common import sha,write,save,rng,restore_rng
from coreset_exp.model import cfg,data_setup,complete,strict_downstream
from libero.lifelong.utils import control_seed
def js(p):return json.loads(Path(p).read_text())
def append(p,r):
    with Path(p).open("a") as f:
        f.write(json.dumps(r,allow_nan=False)+"\n");f.flush();os.fsync(f.fileno())
def cp(stage):return P/("ra_coreset/models/mlrifa_%d_model_ep99.pth"%stage)
def th(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def noise():
    return torch.stack([torch.randn(600,7,generator=torch.Generator(device="cpu").manual_seed(42000000+i)) for i in range(20,30)])
def setup():
    assert torch.cuda.is_available()
    torch.set_num_threads(4);control_seed(42);torch.backends.cudnn.benchmark=True
def flags():
    return dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,deterministic=torch.are_deterministic_algorithms_enabled())
def verify():
    c=js(O/"CONTRACT.json")
    for p,h in c["files"].items():assert sha(p)==h,("Frozen input changed",p)
    return c
def params(policy,data):
    data=policy.preprocess_input(data,train_mode=False)
    h=policy.spatial_encode(data);z=policy.temporal_encode(h)[:,-1]
    dist=policy.policy_head(z,train=False)
    assert not policy.policy_head.low_eval_noise
    assert isinstance(dist,torch.distributions.Independent)
    assert isinstance(dist.base_dist,torch.distributions.Normal)
    assert torch.isfinite(dist.base_dist.loc).all() and torch.isfinite(dist.base_dist.scale).all()
    return dist,z
def freeze():
    O.mkdir(exist_ok=False)
    a=js(A);files=dict(a["input_hashes"])
    for folder in (W,S):
        for f in sorted(folder.rglob("*")):
            if f.is_file() and f.suffix in (".py",".yaml",".yml",".sh") and "__pycache__" not in f.parts:
                files[str(f)]=sha(f)
    files[str(W/"PROTOCOL.md")]=sha(W/"PROTOCOL.md");files[str(A)]=sha(A)
    c=cfg("eval");bench,_,shape=data_setup(c,True)
    for path in (Path(c.pretrain),P/"CONTRACT.json",P/"MODAL_CONTRACT.json"):
        files[str(path)]=sha(path)
    task=bench.get_task(6)
    for path in (Path(c.init_states_folder)/task.problem_folder/task.init_states_file,
                 Path(c.bddl_folder)/task.problem_folder/task.bddl_file):files[str(path)]=sha(path)
    write(O/"CONTRACT.json",dict(files=files,seed=42,indices=list(range(20,30)),stages=[9,7],
        episodes=20,vector_attempts=2,optimizer_calls=0,preflight_environment_calls=0,
        noise_seed="42000000+initial_index",distribution="argmax-component Normal, unchanged scale",
        scope="exploratory reused states; screen only; no automatic distillation"))
    write(O/"RUN_STATUS.json",dict(status="FROZEN",time=time.time()))
    print("FROZEN",flush=True)
def preflight():
    verify();setup();assert not (O/"PREFLIGHT.json").exists()
    c=cfg("eval");c.experiment_dir=str(O/"preflight");bench,_,shape=data_setup(c,True);c.shape_meta=shape
    from libero.lifelong.datasets import get_dataset,SequenceVLDataset
    from torch.utils.data import DataLoader,Subset
    import robomimic.utils.tensor_utils as TU
    ds,_=get_dataset(str(Path(c.folder)/bench.get_task_demonstration(6)),c.data.obs.modality,initialize_obs_utils=False,seq_len=8,demo_limit=10,train=False)
    tok,mask=bench.get_task_emb(6);ds=SequenceVLDataset(ds,tok,mask)
    data=TU.map_tensor(next(iter(DataLoader(Subset(ds,[0]),batch_size=1))),lambda x:x.cuda())
    before=torch.get_rng_state().clone();table=noise();assert torch.equal(before,torch.get_rng_state())
    assert torch.equal(table,noise()) and len(set(th(t) for t in table))==10
    reports=[]
    for stage in (9,7):
        p=complete(c,shape).eval().requires_grad_(False);n=strict_downstream(p,cp(stage));assert n==100
        for length in (1,4,8):
            batch=copy.deepcopy(data);batch["obs"]={k:v[:,:length].contiguous() for k,v in batch["obs"].items()}
            r=copy.deepcopy(rng())
            with torch.no_grad():native=p.get_action(copy.deepcopy(batch))
            restore_rng(r)
            with torch.no_grad():
                dist,z=params(p,copy.deepcopy(batch));action=dist.sample().cpu().numpy()
                assert np.array_equal(native,action)
                head=p.policy_head;x=head.share(z)
                means=head.mean_layer(x).view(-1,5,7)
                scales=head.actv(head.logstd_layer(x).view(-1,5,7))+head.min_std
                logits=head.logits_layer(x);idx=logits.argmax(-1)
                assert torch.equal(dist.base_dist.loc,means[torch.arange(len(idx)),idx])
                assert torch.equal(dist.base_dist.scale,scales[torch.arange(len(idx)),idx])
                rr=copy.deepcopy(rng());eps=table[:1,0].cuda()
                matched=dist.base_dist.loc+dist.base_dist.scale*eps
                assert torch.isfinite(matched).all()
                assert torch.equal(rr["torch"],rng()["torch"]) if "torch" in rr else True
            reports.append(dict(stage=stage,history=length,native_sample_bitwise=True,argmax_mean_scale_exact=True))
        del p;torch.cuda.empty_cache()
    verify()
    write(O/"PREFLIGHT.json",dict(status="PASS",optimizer_calls=0,environment_calls=0,
        actual_gpu=True,checks=reports,numeric_flags=flags(),noise_deterministic=True,noise_global_cpu_rng_unchanged=True))
    print("PREFLIGHT PASS",flush=True)
def rollout(stage):
    assert stage in (9,7);verify();setup()
    assert js(O/"PREFLIGHT.json")["status"]=="PASS"
    assert flags()==js(O/"PREFLIGHT.json")["numeric_flags"]
    phase="parent9" if stage==9 else "teacher7";root=O/phase;root.mkdir(exist_ok=False)
    if stage==7:assert js(O/"parent9/COMPLETE.json")["episodes"]==10
    c=cfg("eval");c.experiment_dir=str(root);b,_,shape=data_setup(c,True);c.shape_meta=shape
    write(root/"resolved_actual.json",dict(c))
    from libero.lifelong.algos.basic import Basic
    from libero.lifelong.metric import SubprocVectorEnv,OffScreenRenderEnv,raw_obs_to_tensor_obs
    control_seed(42);algo=Basic(10,c).cuda();n=strict_downstream(algo.policy,cp(stage))
    algo.eval();algo.requires_grad_(False);assert n==100
    task=b.get_task(6);instr,mask=b.get_task_emb(6);a=js(A)
    states=torch.load(a["initial_file"]);ids=list(range(20,30));table=noise()
    save(root/"rng_before.pth",rng())
    write(root/"RNG_METADATA.json",dict(snapshot_sha256=sha(root/"rng_before.pth"),numeric_flags=flags(),
        global_seed=42,env_seed=42,noise=[dict(initial_index=i,seed=42000000+i,sha256=th(table[lane])) for lane,i in enumerate(ids)]))
    stepctx={"step":0}
    def action(data):
        algo.policy.eval();dist,z=params(algo.policy,data)
        eps=table[:,stepctx["step"]].to(z.device)
        out=dist.base_dist.loc+dist.base_dist.scale*eps
        append(root/"step_hashes.jsonl",dict(step=stepctx["step"],loc_sha256=th(dist.base_dist.loc),
            scale_sha256=th(dist.base_dist.scale),action_sha256=th(out),noise_sha256=th(eps)))
        return out.cpu().numpy()
    algo.policy.get_action=action
    def ledger(event,**kw):append(root/"attempts.jsonl",dict(event=event,time=time.time(),phase=phase,**kw))
    env=None
    try:
        ledger("environment_construct_start",lanes=10)
        args=dict(bddl_file_name=str(Path(c.bddl_folder)/task.problem_folder/task.bddl_file),
            camera_heights=c.data.img_h,camera_widths=c.data.img_w)
        env=SubprocVectorEnv([lambda:OffScreenRenderEnv(**args) for _ in range(10)])
        ledger("environment_construct_complete",lanes=10)
        ledger("vector_start",indices=ids,env_seed=42)
        env.seed(42);env.reset();env.seed(42);algo.reset()
        obs=env.set_init_state(states[np.asarray(ids)])
        for _ in range(5):obs,_,_,_=env.step(np.zeros((10,7)))
        done=[False]*10;first=[None]*10;history=[]
        with torch.no_grad():
            for step in range(600):
                stepctx["step"]=step
                data=raw_obs_to_tensor_obs(obs,instr,mask,c);history.append(data);history=history[-8:]
                batch=dict(obs={k:torch.stack([v["obs"][k] for v in history],dim=1) for k in history[0]["obs"]},
                    task_emb=history[0]["task_emb"],masks=history[0]["masks"])
                actions=algo.policy.get_action(batch);obs,_,now,_=env.step(actions)
                for lane in range(10):
                    if bool(now[lane]) and not done[lane]:first[lane]=step
                    done[lane]=done[lane] or bool(now[lane])
                if all(done):break
        for lane,i in enumerate(ids):
            append(root/"episodes.jsonl",dict(phase=phase,stage=stage,task=6,seed=42,initial_index=i,
                initial_sha256=a["initial_hashes"][i],success=bool(done[lane]),
                first_success_policy_step=first[lane],vector_steps=step+1,env_seed=42,
                checkpoint_sha256=sha(cp(stage)),noise_seed=42000000+i,noise_sha256=th(table[lane])))
        ledger("vector_complete",steps=step+1,successes=done)
    except BaseException as e:
        ledger("technical_exception",error=repr(e));raise
    finally:
        if env is not None:env.close();ledger("environment_closed")
    verify()
    write(root/"COMPLETE.json",dict(status="PASS",episodes=10,successes=sum(done),stage=stage))
    write(O/"RUN_STATUS.json",dict(status=phase.upper()+"_COMPLETE",time=time.time()))
    print(phase,sum(done),"/10",flush=True)
def audit():
    verify();rows={}
    for name in ("parent9","teacher7"):
        root=O/name;rows[name]=[json.loads(l) for l in (root/"episodes.jsonl").read_text().splitlines()]
        assert len(rows[name])==10 and [r["initial_index"] for r in rows[name]]==list(range(20,30))
        events=[json.loads(l)["event"] for l in (root/"attempts.jsonl").read_text().splitlines()]
        assert events==["environment_construct_start","environment_construct_complete","vector_start","vector_complete","environment_closed"]
        assert js(root/"COMPLETE.json")["status"]=="PASS"
    counts={k:0 for k in ("SS","SF","FS","FF")};pairs=[]
    for p,t in zip(rows["parent9"],rows["teacher7"]):
        for key in ("initial_index","initial_sha256","env_seed","noise_seed","noise_sha256"):assert p[key]==t[key]
        category=("S" if p["success"] else "F")+("S" if t["success"] else "F");counts[category]+=1
        pairs.append(dict(initial_index=p["initial_index"],parent_success=p["success"],teacher_success=t["success"],category=category,
            parent_first_success_step=p["first_success_policy_step"],teacher_first_success_step=t["first_success_policy_step"]))
    parent=counts["SS"]+counts["SF"];teacher=counts["SS"]+counts["FS"]
    write(O/"SCREEN_AUDIT.json",dict(status="PASS",episodes=20,vector_attempts=2,technical_exceptions=0,
        parent_successes=parent,teacher_successes=teacher,counts=counts,pairs=pairs,
        difference_pp=10*(teacher-parent),teacher_better_in_this_screen=teacher>parent,
        optimizer_calls=0,inputs_unchanged=True,next_action="STOP: full distillation/independent confirmation protocol not authorized"))
    print("SCREEN AUDIT",parent,teacher,counts,flush=True)
if __name__=="__main__":
    mode=sys.argv[1]
    if mode=="freeze":freeze()
    elif mode=="preflight":preflight()
    elif mode=="rollout":rollout(int(sys.argv[2]))
    elif mode=="audit":audit()
    else:raise ValueError(mode)
