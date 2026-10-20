from .common import *
from .native import install
from torch.utils.data import DataLoader,Subset
import robomimic.utils.tensor_utils as TU

def main():
    assert not (O/"PREFLIGHT.json").exists() and not (O/"preflight_calls.jsonl").exists()
    setup();p=new_public();ph=tensor_hash(p)
    paths=sorted((P/"features/6").glob("*.pt"))[:16]
    xs=[torch.load(f,map_location="cpu") for f in paths]
    h=torch.cat([x["feature"] for x in xs]).cuda();actions=torch.cat([x["actions"] for x in xs]).cuda()
    with torch.no_grad():z=p.temporal_encode(h)[:,-1]
    # Engineering-only projection, never claimed to come from successful rollouts.
    q,_=basis(z[:4]);bank=make_bank(q);opt,sch=optimizer(bank)
    def call():
        append(O/"preflight_calls.jsonl",dict(event="before_optimizer_call",time=time.time()))
        return update(p,bank,opt,sch,h,actions)
    a=call();b=call();snap=dict(bank=cpu(bank.state_dict()),optimizer=cpu(opt.state_dict()),scheduler=cpu(sch.state_dict()),rng=cpu(rng()))
    save(O/"preflight_recovery.pth",snap)
    c=call();after=cpu(bank.state_dict());afteropt=cpu(opt.state_dict())
    bank.load_state_dict(snap["bank"]);opt.load_state_dict(snap["optimizer"]);sch.load_state_dict(snap["scheduler"]);restore_rng(snap["rng"])
    d=call()
    assert all(torch.equal(v,bank.state_dict()[n].cpu()) for n,v in after.items())
    for n,st in afteropt["state"].items():
        for k,v in st.items():assert torch.equal(v,opt.state_dict()["state"][n][k].cpu())
    with torch.no_grad():
        assert float(bank(z[:4],h[:4,-1,0]).abs().max())<1e-6
        assert float(bank(z[4:],h[4:,-1,0]).abs().max())>0
    cfig=cfg("eval");bench,_,shape=data_setup(cfig,False);cfig.shape_meta=shape
    from libero.lifelong.datasets import get_dataset,SequenceVLDataset
    ds,_=get_dataset(str(Path(cfig.folder)/bench.get_task_demonstration(6)),cfig.data.obs.modality,initialize_obs_utils=False,seq_len=8,demo_limit=10,train=False)
    tok,mask=bench.get_task_emb(6);ds=SequenceVLDataset(ds,tok,mask)
    data=TU.map_tensor(next(iter(DataLoader(Subset(ds,[0]),batch_size=1))),lambda x:x.cuda())
    full=complete(cfig,shape).eval().requires_grad_(False);strict_downstream(full,CP)
    # Compare fallback with original policy itself before wrapper installation.
    expected=[]
    for length in (1,4,8):
        batch=copy.deepcopy(data);batch["obs"]={k:v[:,:length].contiguous() for k,v in batch["obs"].items()}
        r=cpu(rng())
        with torch.no_grad():act=full.get_action(batch)
        expected.append((batch,r,act))
    ctx=install(full,bank,False)
    reports=[]
    for batch,r,act in expected:
        restore_rng(r)
        with torch.no_grad():fallback=full.get_action(batch)
        assert np.array_equal(act,fallback)
        full.policy_head.enabled=True
        r=cpu(rng())
        with torch.no_grad():
            sampled=full.get_action(batch);o=ctx["out"];latent=decode(p,bank,ctx["h"])
            assert torch.equal(o["means"],latent["means"])
            restore_rng(r);pred=latent["dist"].sample().cpu().numpy();assert np.array_equal(sampled,pred)
        full.policy_head.enabled=False
        reports.append(dict(history=batch["obs"][next(iter(batch["obs"]))].shape[1],fallback_action_bitwise=True,native_latent_parameters_equal=True))
    assert tensor_hash(p)==ph
    write(O/"PREFLIGHT.json",dict(status="PASS",optimizer_calls=4,environment_steps=0,
        updates=[a,b,c,d],recovery_exact=True,history=reports,public_hash=ph,
        anchors_are_engineering_only=True,formal_initialization="fresh zero U and RNG"))
    print("PREFLIGHT PASS 4 isolated updates, 0 environment")
if __name__=="__main__":main()
