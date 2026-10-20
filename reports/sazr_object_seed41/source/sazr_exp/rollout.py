"""Fixed-budget native LIBERO rollout; separate anchor/validation/formal ledgers."""
from .common import *
from .native import install
from libero.lifelong.metric import SubprocVectorEnv,OffScreenRenderEnv,raw_obs_to_tensor_obs
from libero.lifelong.algos.basic import Basic

def run(phase):
    verify();setup();root=O/phase;root.mkdir()
    assert phase in ("anchor","validation_parent","validation_candidate","formal")
    c=cfg("eval");c.experiment_dir=str(root);bench,_,shape=data_setup(c,True);c.shape_meta=shape
    write(root/"resolved_actual.json",dict(c))
    control_seed(41)
    algo=Basic(10,c).cuda();strict_downstream(algo.policy,CP);algo.eval();algo.requires_grad_(False)
    initial_rng=cpu(rng())
    bank=None;enabled=False
    if phase in ("validation_candidate","formal"):
        if phase=="formal":
            decision=js(O/"DECISION.json")
            assert decision["checkpoint_sha256"]==sha(O/"models/sazr_stage6_calibration_ep9.pth")
            for f,h in decision["files"].items():assert sha(f)==h
        snap=torch.load(O/"models/sazr_stage6_calibration_ep9.pth",map_location="cpu")
        assert snap["contract_sha256"]==sha(O/"CONTRACT.json") and snap["anchor_contract_sha256"]==sha(O/"ANCHOR_CONTRACT.json")
        bank=make_bank(snap["bank"]["Q"]);bank.load_state_dict(snap["bank"],strict=True);bank.requires_grad_(False)
        enabled=phase=="validation_candidate" or js(O/"DECISION.json")["enabled"]
    if phase=="anchor":indices=list(range(20,30))
    elif phase.startswith("validation"):indices=list(range(30,40))
    else:indices=list(range(20))
    table=noise_table(indices) if phase.startswith("validation") else None
    ctx=install(algo.policy,bank,enabled,table)
    restore_rng(initial_rng);save(root/"rng_before.pth",rng())
    task=bench.get_task(6);instr,mask=bench.get_task_emb(6);binding=js(O/"BINDING.json")
    states=torch.load(binding["initial_file"])
    envargs=dict(bddl_file_name=str(Path(c.bddl_folder)/task.problem_folder/task.bddl_file),camera_heights=c.data.img_h,camera_widths=c.data.img_w)
    def ledger(event,**kw):append(root/"attempts.jsonl",dict(event=event,phase=phase,time=time.time(),**kw))
    ledger("environment_construct_start",lanes=10)
    env=None;traces={};success_rows=[]
    try:
        env=SubprocVectorEnv([lambda:OffScreenRenderEnv(**envargs) for _ in range(10)])
        ledger("environment_construct_complete",lanes=10)
        for group in range(len(indices)//10):
            ids=indices[group*10:group*10+10];seed=41+group
            ledger("vector_start",group=group,indices=ids,env_seed=seed)
            env.seed(seed);env.reset();env.seed(seed);algo.reset()
            obs=env.set_init_state(states[np.asarray(ids)])
            for _ in range(5):obs,_,_,_=env.step(np.zeros((10,7)))
            done=[False]*10;first=[None]*10;history=[]
            if phase=="anchor":
                for i in ids:traces[i]=[]
            with torch.no_grad():
                for step in range(600):
                    ctx["step"]=step
                    data=raw_obs_to_tensor_obs(obs,instr,mask,c);history.append(data);history=history[-8:]
                    batch=dict(obs={k:torch.stack([x["obs"][k] for x in history],dim=1) for k in history[0]["obs"]},
                        task_emb=history[0]["task_emb"],masks=history[0]["masks"])
                    actions=algo.policy.get_action(batch)
                    if phase=="anchor":
                        zz=ctx["z"].detach().cpu()
                        for lane,i in enumerate(ids):
                            if not done[lane]:traces[i].append(zz[lane].clone())
                    obs,_,now,_=env.step(actions)
                    for lane in range(10):
                        if bool(now[lane]) and not done[lane]:first[lane]=step
                        done[lane]=done[lane] or bool(now[lane])
                    if all(done):break
            for lane,i in enumerate(ids):
                row=dict(phase=phase,initial_index=i,initial_sha256=binding["initial_hashes"][i],
                    success=bool(done[lane]),first_success_policy_step=first[lane],vector_steps=step+1,
                    env_seed=seed,parent_sha256=binding["checkpoint_sha256"],enabled=enabled)
                if table is not None:row.update(noise_seed=41000000+i,noise_sha256=hashlib.sha256(table[lane].numpy().tobytes()).hexdigest())
                append(root/"episodes.jsonl",row);success_rows.append(row)
            ledger("vector_complete",group=group,steps=step+1,successes=done)
        if phase=="anchor":
            successful={i:torch.stack(traces[i]) for i in indices if any(r["initial_index"]==i and r["success"] for r in success_rows)}
            save(O/"successful_parent_features.pth",successful)
        write(root/"COMPLETE.json",dict(status="PASS",episodes=len(success_rows),successes=sum(r["success"] for r in success_rows),enabled=enabled))
    except BaseException as exc:
        ledger("technical_exception",error=repr(exc));raise
    finally:
        if env is not None:env.close();ledger("environment_closed")
    verify()
    print(phase,"complete",sum(r["success"] for r in success_rows),"/",len(success_rows),flush=True)
if __name__=="__main__":run(sys.argv[1])
