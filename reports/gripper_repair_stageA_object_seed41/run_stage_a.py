"""Fixed Stage A: three isolated recovery calls, 1260 formal updates, no environment."""
import os,sys,json,time,hashlib,copy,inspect
from pathlib import Path
sys.dont_write_bytecode=True
from sazr_exp import common as old
from gripper_core import GripperHead,install
import numpy as np
import torch
from torch import nn
import h5py
import robomimic.utils.tensor_utils as TU
from torch.utils.data import DataLoader,Subset

W=Path(__file__).resolve().parent
O=Path("/data/mlr_ifa/experiments/GripperRepair_A_OBJECT_seed41_20261020")
def write(n,v):
    p=O/n;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,allow_nan=False)+"\n")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def head_new():
    return GripperHead(torch.from_numpy(np.load(old.W/"language.npz")["canonical"])).cuda()
def opt_new(h):
    op=torch.optim.AdamW(h.parameters(),lr=1e-4,weight_decay=.1,betas=(.9,.999),eps=1e-8)
    sc=torch.optim.lr_scheduler.LambdaLR(op,old.make_linear_schedule_with_warmup(500,1260))
    return op,sc
def cpu(v):return old.cpu(v)

def properties(device):
    canon=torch.eye(7,512,device=device)
    h=GripperHead(canon).to(device)
    z=torch.ones(9,512,device=device)
    e=torch.cat([canon,torch.ones(1,512,device=device),torch.zeros(1,512,device=device)])
    acts=torch.zeros(9,7,device=device);acts[:,6]=-1
    assert h.route(e).tolist()==[False]*6+[True,False,False]
    assert torch.equal(h.apply(acts,z,e),acts)
    with torch.no_grad():h.linear.bias.fill_(1)
    x=h.apply(acts,z,e)
    assert x[6,6]==1 and torch.equal(x[:,:6],acts[:,:6])
    assert torch.equal(x[[0,1,2,3,4,5,7,8]],acts[[0,1,2,3,4,5,7,8]])
    with torch.no_grad():h.linear.bias.fill_(-1)
    acts[:,6]=1;x=h.apply(acts,z,e);assert x[6,6]==-1
    h.zero_grad();nn.functional.binary_cross_entropy_with_logits(h(z),torch.ones(9,device=device)).backward()
    assert all(torch.isfinite(p.grad).all() for p in h.parameters())
    assert sum(float(p.grad.abs().sum()) for p in h.parameters())>0
    return dict(device=str(device),route=True,zero_fallback=True,positive_flip=True,negative_flip=True,
                protected_exact=True,motion_exact=True,finite_nonzero_gradient=True)

def native_check(head):
    c=old.cfg("eval");bench,_,shape=old.data_setup(c,False);c.shape_meta=shape
    from libero.lifelong.datasets import get_dataset,SequenceVLDataset
    from libero.lifelong.utils import get_task_embs
    ds,_=get_dataset(str(Path(c.folder)/bench.get_task_demonstration(6)),c.data.obs.modality,
                    initialize_obs_utils=False,seq_len=8,demo_limit=10,train=False)
    tok,mask=bench.get_task_emb(6)
    ds=SequenceVLDataset(ds,tok,mask)
    data=TU.map_tensor(next(iter(DataLoader(Subset(ds,[0]),batch_size=1))),lambda x:x.cuda())
    p=old.complete(c,shape).eval().requires_grad_(False);old.strict_downstream(p,old.CP)
    public_hash=old.tensor_hash(p)
    ctx,original,handle=install(p,head)
    unknown=get_task_embs(c,["move the invisible diagnostic marker to an unspecified place"])
    reports=[]
    for length in (1,4,8):
        for task in list(range(7))+["unknown"]:
            batch=copy.deepcopy(data)
            batch["obs"]={k:v[:,:length].contiguous() for k,v in batch["obs"].items()}
            if task=="unknown":
                batch["task_emb"]=unknown["input_ids"][0:1].cuda()
                batch["masks"]=unknown["attention_mask"][0:1].cuda()
            else:
                t,m=bench.get_task_emb(task)
                batch["task_emb"]=t[None].cuda();batch["masks"]=m[None].cuda()
            r=cpu(old.rng())
            with torch.no_grad():base=original(copy.deepcopy(batch))
            after_base=cpu(old.rng())
            old.restore_rng(r)
            with torch.no_grad():actual=p.get_action(copy.deepcopy(batch))
            assert np.array_equal(actual[:,:6],base[:,:6])
            assert torch.equal(after_base["cpu"],cpu(old.rng())["cpu"])
            for a,b in zip(after_base["cuda"],cpu(old.rng())["cuda"]):assert torch.equal(a,b)
            if task!=6:assert np.array_equal(actual,base) and not ctx["route"].any()
            else:
                assert ctx["route"].all()
                logit=float(ctx["logit"][0])
                expected=base[0,6] if logit==0 else (1 if logit>0 else -1)
                assert actual[0,6]==expected
            reports.append(dict(history=length,task=task,motion_exact=True,
                                protected_full_exact=task!=6,classifier_matches_actual=True))
    # Engineered forced flips at the real action interface, not learned-performance evidence.
    save=cpu(head.state_dict())
    forced=[]
    batch=copy.deepcopy(data)
    for sign in (-1,1):
        with torch.no_grad():head.linear.weight.zero_();head.linear.bias.fill_(sign)
        r=cpu(old.rng())
        with torch.no_grad():base=original(copy.deepcopy(batch))
        old.restore_rng(r)
        with torch.no_grad():actual=p.get_action(copy.deepcopy(batch))
        assert actual[0,6]==sign and np.array_equal(actual[:,:6],base[:,:6])
        forced.append(dict(forced_sign=sign,actual_sign=int(actual[0,6])))
    head.load_state_dict(save,strict=True)
    assert old.tensor_hash(p)==public_hash
    handle.remove()
    return dict(status="PASS",cases=reports,forced_interface_checks=forced,
                environment_constructs=0,environment_steps=0,public_tensor_unchanged=True)

def main():
    assert not O.exists(),"Never overwrite an existing run"
    O.mkdir(parents=True)
    write("RUN_STATUS.json",dict(status="ADMISSION",time=time.time()))
    old.verify();old.setup()
    bindpath=old.P/"bindings/task6.json";binding=old.js(bindpath)
    fm=old.js(old.P/"FEATURES.json")
    paths=sorted((old.P/"features/6").glob("*.pt"));assert len(paths)==1562
    files={str(old.CP):sha(old.CP),str(bindpath):sha(bindpath),str(old.W/"language.npz"):sha(old.W/"language.npz")}
    for f in W.iterdir():
        if f.is_file() and f.suffix in (".py",".md",".sh",".txt"):files[str(f)]=sha(f)
    files[binding["hdf5"]]=sha(binding["hdf5"])
    ids=[];excluded=[];train_ids=[];test_ids=[]
    with h5py.File(binding["hdf5"]) as hf:
        for f in paths:
            h=sha(f);assert h==fm["hashes"][f"6/{f.name}"]==binding["windows"][f.name]["sha256"]
            files[str(f)]=h
            row=binding["windows"][f.name]
            if row["demo_id"]=="AMBIGUOUS":
                excluded.append(dict(sample_id=f.name,reason="cross-demo identity ambiguous",sha256=h));continue
            d=row["demo_id"]
            assert d in [f"demo_{j}" for j in range(10)]
            x=torch.load(f,map_location="cpu")
            starts=row["possible_starts"];demo=hf["data"][d]["actions"][:].astype(np.float32)
            assert any(np.array_equal(demo[np.minimum(np.arange(s,s+8),len(demo)-1)],x["actions"][0].numpy()) for s in starts)
            assert float(x["actions"][0,-1,6]) in (-1.,1.)
            split="train" if int(d.split("_")[-1])<8 else "holdout"
            idx=len(ids);(train_ids if split=="train" else test_ids).append(idx)
            ids.append(dict(sample_id=f.name,demo_id=d,split=split,sha256=h))
    assert (len(train_ids),len(test_ids),len(excluded))==(1255,303,4)
    write("SPLIT.json",dict(train_demos=[f"demo_{j}" for j in range(8)],
          holdout_demos=["demo_8","demo_9"],entries=ids,excluded=excluded,
          limitation="Parent previously trained on all 10 demonstrations; holdout is only relative to new head."))
    from robosuite.models.grippers.panda_gripper import PandaGripper
    path=Path(inspect.getfile(PandaGripper));files[str(path)]=sha(path)
    # No environment: invoke pure format_action on a fresh object carrying only its accumulator.
    maps=[]
    for sign in (-1,1):
        grip=object.__new__(PandaGripper);grip.current_action=np.zeros(2)
        control=grip.format_action(np.array([sign]))
        assert np.array_equal(control,np.array([-1.,1.])*.01*sign)
        maps.append(dict(command=sign,meaning="open" if sign==-1 else "closed",actuator_delta=control.tolist()))
    write("ENVIRONMENT_SOURCE_AUDIT.json",dict(gripper_source=str(path),sha256=sha(path),mapping=maps,
          evidence="Actual PandaGripper.format_action docstring and pure function; no physical environment",
          state_restore_status="STATIC_ONLY_NOT_COUNTERFACTUAL_VERIFIED",
          required_additional_state=["gripper.current_action","controller internal targets","observable state/history","RNG","physics warm-start/control state"]))
    write("CONTRACT.json",dict(status="FROZEN_BEFORE_TRAINING",files=files,seed=41,epochs=10,steps=1260,
          budget_environment=0,heldout_locked=True,architecture="Linear512to1 independent binary gripper",
          source_parent_contract=str(old.O/"CONTRACT.json")))
    prop=[properties("cpu"),properties("cuda")]
    write("SOFTWARE_TESTS.json",dict(status="PASS",checks=prop))
    p=old.new_public();ph=old.tensor_hash(p)
    zs=[];es=[];ys=[];parents=[]
    with torch.inference_mode():
        for start in range(0,len(ids),32):
            xx=[torch.load(old.P/"features/6"/r["sample_id"],map_location="cpu") for r in ids[start:start+32]]
            h=torch.cat([x["feature"] for x in xx]).cuda()
            z=p.temporal_encode(h)[:,-1]
            b,s,l=old.raw(p.policy_head,z);ii=l.argmax(-1);rr=torch.arange(len(z),device=z.device)
            zs.append(z.cpu());es.append(h[:,-1,0].cpu())
            ys.append(torch.cat([x["actions"] for x in xx])[:,-1,6])
            parents.append(b[rr,ii,6].tanh().cpu())
    z=torch.cat(zs).cuda();e=torch.cat(es).cuda();y=torch.cat(ys).cuda();parent=torch.cat(parents).cuda()
    target=(y>0).float()
    # Three isolated calls: step1 snapshot, step2, restore and replay step2.
    pre=head_new();op,sc=opt_new(pre);call_count=0
    def call(h,op,sc,ix):
        op.zero_grad(set_to_none=True)
        loss=nn.functional.binary_cross_entropy_with_logits(h(z[ix]),target[ix])
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(h.parameters(),100)
        assert torch.isfinite(norm) and float(norm)>0
        op.step();sc.step()
        return float(loss),float(norm)
    ix=torch.tensor(train_ids[:10],device="cuda")
    call(pre,op,sc,ix);call_count+=1
    saved=dict(head=cpu(pre.state_dict()),optimizer=cpu(op.state_dict()),scheduler=cpu(sc.state_dict()),rng=cpu(old.rng()))
    call(pre,op,sc,ix);call_count+=1;expected=cpu(pre.state_dict());expected_op=cpu(op.state_dict())
    pre.load_state_dict(saved["head"]);op.load_state_dict(saved["optimizer"]);sc.load_state_dict(saved["scheduler"]);old.restore_rng(saved["rng"])
    call(pre,op,sc,ix);call_count+=1
    assert all(torch.equal(v,pre.state_dict()[k].cpu()) for k,v in expected.items())
    for k,st in expected_op["state"].items():
        for key,v in st.items():assert torch.equal(v,op.state_dict()["state"][k][key].cpu())
    write("PREFLIGHT.json",dict(status="PASS",isolated_optimizer_calls=call_count,recovery_exact=True,environment_steps=0))
    old.control_seed(41);head=head_new()
    assert head.route(e).all()
    op,sc=opt_new(head);epochs=[];steps=0
    for epoch in range(10):
        gen=torch.Generator().manual_seed(epoch)
        order=torch.tensor(train_ids,device="cuda")[torch.randperm(len(train_ids),generator=gen).cuda()]
        losses=[];norms=[];tic=time.time()
        for start in range(0,len(order),10):
            loss,norm=call(head,op,sc,order[start:start+10]);steps+=1
            losses.append(loss);norms.append(norm)
        row=dict(epoch=epoch,batches=len(losses),steps=steps,samples=len(order),
                 BCE=float(np.mean(losses)),grad_min=min(norms),grad_max=max(norms),seconds=time.time()-tic)
        epochs.append(row);write("epochs.json",epochs)
        (O/"models").mkdir(exist_ok=True)
        tmp=O/"models/latest.tmp"
        torch.save(dict(head=cpu(head.state_dict()),optimizer=cpu(op.state_dict()),scheduler=cpu(sc.state_dict()),
                        rng=cpu(old.rng()),next_epoch=epoch+1,steps=steps,epochs=epochs,
                        contract_sha256=sha(O/"CONTRACT.json"),parent_tensor_hash=ph),tmp)
        tmp.replace(O/"models/latest.pth")
    assert steps==1260 and len(epochs)==10
    final=O/"models/gripper_ep9.pth"
    (O/"models/latest.pth").replace(final)
    loaded=head_new();loaded.load_state_dict(torch.load(final,map_location="cpu")["head"],strict=True)
    assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in head.state_dict().items())
    head.eval()
    with torch.no_grad():
        logits=head(z);pred=torch.where(logits>0,torch.ones_like(y),-torch.ones_like(y))
        pred=torch.where(logits==0,torch.sign(parent),pred)
    records=[]
    for j,row in enumerate(ids):
        records.append(dict(**row,parent_correct=bool(torch.sign(parent[j])==y[j]),
                            candidate_correct=bool(pred[j]==y[j]),
                            parent_predicted_class=int(torch.sign(parent[j])),candidate_predicted_class=int(pred[j]),
                            target_class=int(y[j])))
    # Label classes only, not raw actions or latent payloads.
    write("WINDOW_OUTCOMES.json",records)
    def metric(split):
        rr=[r for r in records if r["split"]==split]
        pc=sum(r["parent_correct"] for r in rr);cc=sum(r["candidate_correct"] for r in rr)
        rescue=sum(not r["parent_correct"] and r["candidate_correct"] for r in rr)
        harm=sum(r["parent_correct"] and not r["candidate_correct"] for r in rr)
        assert cc-pc==rescue-harm
        return dict(n=len(rr),parent_correct=pc,candidate_correct=cc,parent_errors=len(rr)-pc,
                    label_repairs=rescue,label_harms=harm,net=cc-pc,
                    error_correction_rate=rescue/(len(rr)-pc) if pc<len(rr) else None,
                    parent_correct_retention=(pc-harm)/pc if pc else None)
    metrics={split:metric(split) for split in ("train","holdout")}
    write("NATIVE_INTERFACE.json",native_check(head))
    assert old.tensor_hash(p)==ph
    for f,h in files.items():assert sha(f)==h,("Input/source changed",f)
    old.verify()
    write("FINAL_AUDIT.json",dict(status="PASS",engineering_complete=True,metrics=metrics,
          formal_optimizer_steps=1260,isolated_optimizer_steps=3,exposures=12550,
          environment_constructs=0,environment_steps=0,strict_checkpoint_load=True,
          checkpoint_sha256=sha(final),parent_tensor_hash=ph,all_inputs_source_unchanged=True,
          learned_correction_observed=metrics["train"]["label_repairs"]>0,
          heldout_correction_observed=(metrics["holdout"]["label_repairs"]>0 if metrics["holdout"]["parent_errors"] else None)))
    write("RUN_STATUS.json",dict(status="STAGE_A_COMPLETED",B="DEFERRED_BY_USER",C="NOT_STARTED",time=time.time()))
    print("STAGE_A_RESULT="+json.dumps(metrics))

if __name__=="__main__":
    try:main()
    except BaseException:
        import traceback
        if O.exists():write("FAILURE.json",dict(traceback=traceback.format_exc(),time=time.time()))
        raise
