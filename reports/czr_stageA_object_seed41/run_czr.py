"""Native-image CZR feasibility. Zero optimizer calls and zero environments."""
import sys,json,time,copy,hashlib,os
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch,h5py
from torch.utils.data import DataLoader,Subset
import robomimic.utils.tensor_utils as TU
from libero.lifelong.datasets import get_dataset,SequenceVLDataset
from libero.lifelong.utils import get_task_embs
from sazr_exp import common as old
from czr_core import Memory,language_route,metric,properties

W=Path(__file__).resolve().parent
O=Path("/data/mlr_ifa/analyses/CZR_A_OBJECT_seed41_20261020")
PREV=Path("/data/mlr_ifa/experiments/GripperRepair_A_OBJECT_seed41_20261020")
def write(name,obj):
    (O/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def subset_metrics(rows,key):
    return {s:metric(np.array([r["pc"] for r in rows if r["split"]==s]),
                     np.array([r[key] for r in rows if r["split"]==s]),
                     np.ones(sum(r["split"]==s for r in rows),dtype=bool))
            for s in ("train","holdout")}
def main():
    assert not O.exists(),"Refuse duplicate execution"
    O.mkdir(parents=True);tic=time.time()
    write("RUN_STATUS.json",dict(status="ADMISSION",pid=os.getpid(),time=time.time()))
    old.verify();old.setup()
    base=old.js(PREV/"CONTRACT.json")
    files=dict(base["files"])
    for f in W.iterdir():
        if f.is_file() and f.suffix in (".py",".md",".txt",".sh"):files[str(f)]=sha(f)
    files[str(PREV/"SPLIT.json")]=sha(PREV/"SPLIT.json")
    for f,h in files.items():assert sha(f)==h,("Frozen file changed",f)
    write("CONTRACT.json",dict(files=files,protocol_sha256=sha(W/"PROTOCOL.md"),
           radius=.45,epsilon=1e-12,optimizer_budget=0,environment_budget=0,time=time.time()))
    write("SOFTWARE_TESTS.json",properties())
    c=old.cfg("eval");bench,_,shape=old.data_setup(c,False);c.shape_meta=shape
    ds,_=get_dataset(str(Path(c.folder)/bench.get_task_demonstration(6)),c.data.obs.modality,
                    initialize_obs_utils=False,seq_len=8,demo_limit=10,train=False)
    tok,mask=bench.get_task_emb(6);wrapped=SequenceVLDataset(ds,tok,mask)
    p=old.complete(c,shape).eval().requires_grad_(False);old.strict_downstream(p,old.CP)
    ph=old.tensor_hash(p);ctx={}
    spatial=p.spatial_encode
    def capture(data):
        h=spatial(data);ctx["h"]=h;return h
    p.spatial_encode=capture
    hook=p.policy_head.register_forward_pre_hook(lambda m,args:ctx.update(z=args[0].detach()))
    original=p.get_action
    canon=np.load(old.W/"language.npz")["canonical"]
    binding=old.js(old.P/"bindings/task6.json")
    split=old.js(PREV/"SPLIT.json")
    rows=[];excluded=list(split["excluded"]);zs=[];ys=[];parents=[];indices=[]
    def data_at(index):
        return TU.map_tensor(next(iter(DataLoader(Subset(wrapped,[index]),batch_size=1))),lambda x:x.cuda())
    with h5py.File(binding["hdf5"]) as hf,torch.inference_mode():
        for r in split["entries"]:
            b=binding["windows"][r["sample_id"]]
            if len(b["possible_starts"])!=1:
                excluded.append(dict(sample_id=r["sample_id"],reason="multiple starts"));continue
            start=b["possible_starts"][0];demo=hf["data"][r["demo_id"]]["actions"][:].astype(np.float32)
            old_window=torch.load(old.P/"features/6"/r["sample_id"],map_location="cpu")["actions"][0].numpy()
            assert np.array_equal(demo[np.minimum(np.arange(start,start+8),len(demo)-1)],old_window)
            index=ds._demo_id_to_start_indices[r["demo_id"]]+start;data=data_at(index)
            assert np.array_equal(data["actions"][0].cpu().numpy(),old_window)
            label=int(data["actions"][0,-1,6]);assert label in (-1,1)
            original(copy.deepcopy(data))
            z=ctx["z"];bmu,std,logits=old.raw(p.policy_head,z)
            par=int(bmu[0,logits[0].argmax(),6].tanh().sign());assert par in (-1,1)
            assert language_route(ctx["h"][:,-1,0].cpu().numpy(),canon).all()
            rows.append(dict(**r,target_step=min(start+7,len(demo)-1),start=start,
                             parent_correct=par==label))
            zs.append(z[0].cpu().numpy());ys.append(label);parents.append(par);indices.append(index)
            if len(rows)%100==0:
                write("RUN_STATUS.json",dict(status="EXTRACTING_NATIVE",windows=len(rows),time=time.time()))
                print("NATIVE_WINDOWS",len(rows),flush=True)
    z=np.array(zs);y=np.array(ys);pa=np.array(parents)
    train=np.array([r["split"]=="train" for r in rows]);test=~train
    assert (train.sum(),test.sum(),len(excluded))==(1255,303,4)
    # Server-only features, no cached-latent training or payload download.
    torch.save(dict(z=torch.from_numpy(z),label=torch.from_numpy(y),parent=torch.from_numpy(pa)),O/"native_features.pt")
    bank=Memory(z[train],pa[train],y[train])
    torch.save(dict(mu=bank.mu,centers=bank.centers,parent=bank.parent,label=bank.label,radius=bank.radius),O/"repair_memory.pt")
    co,tr,conf,n=bank.query(z,pa)
    assert not tr[train&(pa==y)].any(),"Known correct protection failed"
    events=[];case_rows=[];event_counter=0
    last={}
    for i in sorted(np.flatnonzero(train&(pa!=y)),key=lambda i:(rows[i]["demo_id"],rows[i]["target_step"],rows[i]["sample_id"])):
        r=rows[i];key=(r["demo_id"],int(pa[i]),int(y[i]))
        prev=last.get(key)
        if prev is None or r["target_step"]>prev[0]+1:
            event_counter+=1;eid=event_counter
        else:eid=prev[1]
        last[key]=(r["target_step"],eid)
        j=np.flatnonzero(np.flatnonzero(train)[bank.indices]==i)[0]
        case_rows.append(dict(sample_id=r["sample_id"],demo_id=r["demo_id"],target_step=r["target_step"],
                event_id=eid,radius=float(bank.radius[j]),nearest_correct_distance=float(bank.radius[j]/.45)))
    folds=[]
    for d in range(8):
        valid=train&np.array([r["demo_id"]==f"demo_{d}" for r in rows]);fit=train&~valid
        m=Memory(z[fit],pa[fit],y[fit]);cc,tt,cf,_=m.query(z[valid],pa[valid])
        folds.append(dict(demo_id=f"demo_{d}",fit_n=int(fit.sum()),error_centers=len(m.radius),
             metrics=metric(pa[valid],cc,y[valid]),triggers=int(tt.sum()),conflicts=int(cf.sum()),
             normalization_fit_only=True))
    write("TRAINING_CROSS_DEMO.json",folds)
    write("ERROR_CASES_LIGHT.json",case_rows)
    # Lock memory before once-only heldout reporting; no tuning or fitting follows.
    memory_hash=sha(O/"repair_memory.pt")
    write("MEMORY.json",dict(hash=memory_hash,file_bytes=(O/"repair_memory.pt").stat().st_size,
          numeric_payload_bytes=bank.mu.nbytes+bank.centers.nbytes+bank.parent.nbytes+bank.label.nbytes+bank.radius.nbytes,
          training_windows=int(train.sum()),correct_protection=int((train&(pa==y)).sum()),
          error_centers=len(bank.radius),error_events=event_counter,train_only=True,
          native_features_sha256=sha(O/"native_features.pt"),native_features_bytes=(O/"native_features.pt").stat().st_size))
    def actual_action(data):
        with torch.no_grad():
            parent=original(data);zz=ctx["z"]
            bm,_,lg=old.raw(p.policy_head,zz)
            rr=torch.arange(len(zz),device=zz.device)
            par=bm[rr,lg.argmax(-1),6].tanh().sign().cpu().numpy().astype(int)
            pred,hit,cf,_=bank.query(zz.cpu().numpy(),par)
            admitted=language_route(ctx["h"][:,-1,0].cpu().numpy(),canon)
            hit=hit&admitted
            out=parent.copy();out[hit,6]=pred[hit]
            ctx.update(trigger=hit,admitted=admitted,parent_sample=parent.copy(),prediction=pred,conflict=cf)
            return out
    p.get_action=actual_action
    # Same native-image query path on ALL windows; no environment executions.
    records=[];max_z_diff=0.
    old.control_seed(41)
    with torch.inference_mode():
        for i,(r,index) in enumerate(zip(rows,indices)):
            data=data_at(index);rng=old.cpu(old.rng())
            base_action=original(copy.deepcopy(data));after=old.cpu(old.rng())
            old.restore_rng(rng);a=p.get_action(copy.deepcopy(data))
            now=old.cpu(old.rng())
            assert torch.equal(after["cpu"],now["cpu"])
            assert all(torch.equal(v,w) for v,w in zip(after["cuda"],now["cuda"]))
            diff=float(np.abs(ctx["z"][0].cpu().numpy()-z[i]).max());max_z_diff=max(max_z_diff,diff)
            assert diff==0,"Native feature mismatch"
            assert bool(ctx["trigger"][0])==bool(tr[i])
            assert np.array_equal(base_action[:,:6],a[:,:6])
            if not tr[i]:assert np.array_equal(base_action,a)
            if tr[i]:assert a[0,6]==co[i]
            records.append(dict(**r,candidate_correct=bool(co[i]==y[i]),trigger=bool(tr[i]),
                  conflicting_support=bool(conf[i]),support_count=int(n[i]),motion_exact=True,
                  rejected_action_exact=not bool(tr[i]),native_feature_exact=True,rng_exact=True,
                  sampled_parent_correct=bool(np.sign(base_action[0,6])==y[i]),
                  sampled_candidate_correct=bool(np.sign(a[0,6])==y[i])))
            if (i+1)%200==0:
                print("NATIVE_INTERFACE",i+1,flush=True)
                write("RUN_STATUS.json",dict(status="CHECKING_INTERFACE",windows=i+1,time=time.time()))
    write("WINDOW_OUTCOMES_LIGHT.json",records)
    # Base/unknown instructions and startup history lengths must fall back exactly.
    unknown=get_task_embs(c,["move the invisible diagnostic marker to an unspecified place"])
    interface=[]
    data=data_at(indices[0])
    with torch.inference_mode():
        for length in (1,4,8):
            for task in list(range(7))+["unknown"]:
                batch=copy.deepcopy(data)
                batch["obs"]={k:v[:,:length].contiguous() for k,v in batch["obs"].items()}
                if task=="unknown":
                    batch["task_emb"]=unknown["input_ids"][0:1].cuda()
                    batch["masks"]=unknown["attention_mask"][0:1].cuda()
                else:
                    t,m=bench.get_task_emb(task);batch["task_emb"]=t[None].cuda();batch["masks"]=m[None].cuda()
                rng=old.cpu(old.rng());a=original(copy.deepcopy(batch));old.restore_rng(rng)
                b=p.get_action(copy.deepcopy(batch))
                assert np.array_equal(a[:,:6],b[:,:6])
                if task!=6:assert not ctx["trigger"].any() and np.array_equal(a,b)
                interface.append(dict(history=length,task=task,motion_exact=True,
                                      protected_full_exact=(task!=6),status="PASS"))
    write("NATIVE_INTERFACE.json",dict(status="PASS",window_checks=len(rows),extra_cases=interface,
                    max_feature_difference=max_z_diff,global_rng_consumption_equal=True))
    assert old.tensor_hash(p)==ph and sha(O/"repair_memory.pt")==memory_hash
    for f,h in files.items():assert sha(f)==h,("Frozen file changed",f)
    old.verify()
    metrics={s:metric(pa[mask],co[mask],y[mask]) for s,mask in (("train",train),("holdout",test))}
    gate=metrics["holdout"]["FS"]>=1 and metrics["holdout"]["SF"]==0
    cross=sum(f["metrics"]["FS"] for f in folds)>0
    write("FINAL_AUDIT.json",dict(status="PASS",metrics=metrics,prototype_gate=bool(gate),
        cross_demo_support=bool(cross),recommend_closure=bool(gate and cross),
        known_correct_protection=True,parent_tensor_unchanged=True,parent_tensor_hash=ph,
        frozen_inputs_unchanged=True,optimizer_calls=0,environment_constructs=0,environment_steps=0,
        policy_rollouts=0,seconds=time.time()-tic,excluded=excluded,feature_mode="native eval",
        note="No causal physical-error evidence; parent saw all demos. B budget remains unspecified."))
    write("RUN_STATUS.json",dict(status="STAGE_A_COMPLETED",prototype_gate=bool(gate),
                       B="NOT_STARTED",time=time.time()))
    print("CZR_RESULT",json.dumps(metrics),flush=True)

if __name__=="__main__":
    try:main()
    except BaseException:
        import traceback
        if O.exists():write("FAILURE.json",dict(traceback=traceback.format_exc(),time=time.time()))
        raise
