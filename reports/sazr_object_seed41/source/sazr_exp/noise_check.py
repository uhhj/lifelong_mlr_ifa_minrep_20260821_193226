"""Real-image GPU validation-adapter check: zero updates, zero simulator steps."""
from .common import *
from .native import install
from torch.utils.data import DataLoader,Subset
import robomimic.utils.tensor_utils as TU
def main():
    assert not (O/"NOISE_INTERFACE.json").exists()
    setup();p=new_public()
    saved=torch.load(O/"preflight_recovery.pth",map_location="cpu")
    bank=make_bank(saved["bank"]["Q"]);bank.load_state_dict(saved["bank"])
    c=cfg("eval");b,_,shape=data_setup(c,False);c.shape_meta=shape
    from libero.lifelong.datasets import get_dataset,SequenceVLDataset
    ds,_=get_dataset(str(Path(c.folder)/b.get_task_demonstration(6)),c.data.obs.modality,initialize_obs_utils=False,seq_len=8,demo_limit=10,train=False)
    tok,mask=b.get_task_emb(6);ds=SequenceVLDataset(ds,tok,mask)
    data=TU.map_tensor(next(iter(DataLoader(Subset(ds,[0]),batch_size=1))),lambda x:x.cuda())
    full=complete(c,shape).eval().requires_grad_(False);strict_downstream(full,CP)
    table=noise_table([30]);ctx=install(full,bank,True,table);rows=[]
    with torch.no_grad():
        for length in (1,4,8):
            batch=copy.deepcopy(data);batch["obs"]={k:v[:,:length].contiguous() for k,v in batch["obs"].items()}
            ctx["step"]=length-1;before=cpu(rng());action=full.get_action(batch)
            out=decode(p,bank,ctx["h"])
            expected=(out["dist"].base_dist.loc+out["dist"].base_dist.scale*table[:,length-1].cuda()).cpu().numpy()
            assert np.array_equal(action,expected)
            after=rng();assert torch.equal(before["cpu"],after["cpu"])
            assert all(torch.equal(a,b) for a,b in zip(before["cuda"],after["cuda"]))
            rows.append(dict(history=length,action_exact=True,global_torch_rng_unchanged=True))
    write(O/"NOISE_INTERFACE.json",dict(status="PASS",rows=rows,optimizer_calls=0,environment_steps=0,
        noise_hash=hashlib.sha256(table.numpy().tobytes()).hexdigest()))
    print("NOISE INTERFACE PASS: real history1/4/8, zero updates/environment")
if __name__=="__main__":main()
