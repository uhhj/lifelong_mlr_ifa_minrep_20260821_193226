"""Inference-only historical RA diagnostic. No optimizer, environment or samples."""
import os,sys,json,time,hashlib,math,copy,argparse,traceback
from pathlib import Path
sys.dont_write_bytecode=True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
import torch
import h5py

def js(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
    return h.hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")
def thash(p,frontend=False):
    h=hashlib.sha256()
    for k,v in sorted(p.state_dict().items()):
        if frontend and k.startswith(("temporal_transformer.","policy_head.")):continue
        a=v.detach().cpu().contiguous()
        h.update(k.encode());h.update(str((a.dtype,tuple(a.shape))).encode());h.update(a.numpy().tobytes())
    return h.hexdigest()
def fail(*a,**k):raise RuntimeError("Stage A forbids optimizer construction/update and action sampling")

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--suite",choices=["GOAL42","OBJECT41"],required=True)
    ap.add_argument("--out",required=True);a=ap.parse_args()
    O=Path(a.out);assert not O.exists(),"Do not overwrite any prior attempt"
    O.mkdir(parents=True);start=time.time()
    def status(phase,**kw):write(O/"RUN_STATUS.json",dict(status=phase,pid=os.getpid(),time=time.time(),**kw))
    status("ADMISSION")
    from coreset_exp.common import verify,SEED
    from coreset_exp import common
    from coreset_exp.model import cfg,data_setup,complete,strict_downstream
    from libero.lifelong.selectors.greedy_kcenter import robust_scale
    from libero.lifelong.datasets import get_dataset,SequenceVLDataset
    from libero.lifelong.utils import control_seed
    import robomimic.utils.tensor_utils as TU
    from torch.utils.data import DataLoader,Subset
    P=common.O
    assert (a.suite,SEED) in [("GOAL42",42),("OBJECT41",41)]
    assert torch.cuda.is_available(),"No CPU fallback"
    contract=verify()
    # Runtime guard: these paths are never needed by inference.
    torch.optim.Optimizer.__init__=fail
    torch.distributions.Normal.sample=fail
    torch.distributions.Independent.sample=fail
    torch.set_num_threads(4);control_seed(SEED)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    c=cfg("eval")
    assert c.training is False and all(c.policy[k] is True for k in ("use_state","use_vision","use_language"))
    b,_,shape=data_setup(c,False);c.shape_meta=shape
    paths=set()
    for name in ("CONTRACT.json","MODAL_CONTRACT.json","FEATURES.json","PARENT_BINDING.json","resolved_eval.json"):
        paths.add(P/name)
    paths.update(Path(p) for p in contract["files"])
    paths.update(p for p in common.S.rglob("*") if p.is_file() and p.suffix in (".py",".yaml",".sh") and "__pycache__" not in p.parts)
    paths.add(Path(c.pretrain))
    cp={s:P/"ra_coreset/models"/("mlrifa_%d_model_ep99.pth"%s) for s in range(6,10)}
    paths.update(cp.values())
    taskdata={};excluded=[];selection=[]
    # Freeze sample identities BEFORE any model forward.
    for task in (6,7,8):
        status("IDENTITY",task=task)
        bp=P/"bindings"/("task%d.json"%task);sp=P/"ra_coreset/storage"/("task%d.json"%task)
        paths.update([bp,sp]);bind=js(bp);storage=js(sp)
        hp=Path(bind["hdf5"]);paths.add(hp)
        assert sha(hp)==bind["hdf5_sha256"]
        selected={Path(v["source"]).name for v in storage["selected"]}
        files=sorted((P/"features"/str(task)).glob("*.pt"),key=lambda p:p.name)
        paths.update(files);H=[];A=[];rows=[]
        with h5py.File(hp) as hf:
            demos=js(P/"FEATURES.json")["demos"][str(task)]
            pop=np.concatenate([hf["data"][d]["actions"][:] for d in demos])
            scale=robust_scale(pop)
            assert np.array_equal(scale,np.asarray(storage["selection"]["original_selection"]["action_scale"]))
            for p in files:
                rr=bind["windows"][p.name];assert sha(p)==rr["sha256"]
                x=torch.load(p,map_location="cpu")
                assert tuple(x["feature"].shape)==(1,8,4,512) and tuple(x["actions"].shape)==(1,8,7)
                H.append(x["feature"]);A.append(x["actions"][0,-1].numpy())
                unique=rr.get("identity_unique",True) and rr.get("demo_id") is not None
                r=dict(suite=a.suite,seed=SEED,task=task,sample_id=p.name,
                       demo_id=rr.get("demo_id") or "unknown",target=rr.get("target"),
                       length=rr.get("length"),selected=p.name in selected,native_sample=False,
                       identity_unique=bool(unique),index=len(rows))
                if unique and len(rr["possible_starts"])==1:
                    t=rr["possible_starts"][0];raw=hf["data"][r["demo_id"]]["actions"][:].astype(np.float32)
                    expected=raw[np.minimum(np.arange(t,t+8),len(raw)-1)]
                    assert np.array_equal(expected,x["actions"][0].numpy())
                    assert r["target"]==min(t+7,len(raw)-1)
                    r.update(start=t,quartile=min(3,int(4*r["target"]/max(1,r["length"]-1))))
                else:excluded.append(dict(suite=a.suite,task=task,sample_id=p.name,reason="unknown target or nonunique history"))
                rows.append(r)
        for demo in sorted({r["demo_id"] for r in rows if "start" in r}):
            for q in range(4):
                for member in (False,True):
                    group=[r for r in rows if r["demo_id"]==demo and r.get("quartile")==q and r["selected"]==member and "start" in r]
                    unique={}
                    for r in sorted(group,key=lambda r:(r["target"],r["sample_id"])):unique.setdefault(r["target"],r)
                    vals=list(unique.values())
                    if not vals:continue
                    for idx in sorted({min(len(vals)-1,len(vals)//3),min(len(vals)-1,2*len(vals)//3)}):
                        vals[idx]["native_sample"]=True
        native=[r for r in rows if r["native_sample"]]
        assert len(native)<=160
        selection.extend({k:v for k,v in r.items() if k!="index"} for r in native)
        taskdata[task]=dict(rows=rows,H=torch.cat(H,0),A=np.asarray(A),scale=scale,native=native)
        for x in storage["selected"]:
            dst=P/"ra_coreset/buffer"/x["destination"]
            if not dst.exists():dst=Path(c.buffer_dir)/x["destination"] if hasattr(c,"buffer_dir") else dst
            # Source membership/payload is verified; final buffer root comes from training config.
            train=js(P/"resolved_ra_coreset.json")
            dst=Path(train["buffer_dir"])/x["destination"]
            assert sha(dst)==x["sha256"];paths.add(dst)
        taskdata[task]["hdf5"]=str(hp)
    here=Path(__file__).resolve().parent
    paths.update([here/"PROTOCOL.md",Path(__file__)])
    before={str(p):sha(p) for p in sorted(paths)}
    write(O/"INPUT_MANIFEST.json",dict(files=before,parent=str(P),source=str(common.S)))
    write(O/"NATIVE_SELECTION.json",dict(rows=selection,excluded=excluded,prediction_independent=True))
    write(O/"DIAGNOSTIC_CONFIG.json",dict(suite=a.suite,seed=SEED,cache_batch=16,native_batch=4,
        teacher_error_max=.5,cross_mean_max=.25,cross_std_max=.25,harm_delta=.1,
        validation_demo_ids=list(range(5,10)),optimizer_budget=0,environment_budget=0,
        policy_action_samples=0,deterministic_algorithms=True,cudnn_benchmark=False,tf32=False,
        original_eval_config_sha256=before[str(P/"resolved_eval.json")],
        note="Diagnostic eval-mode flags do not modify original runs; all historical demos were training demos."))
    p=complete(c,shape).eval().requires_grad_(False)
    front=thash(p,True);base=thash(p);native_seconds=0;decode_seconds=0
    with torch.inference_mode():
        for task,d in taskdata.items():
            tic=time.time();status("NATIVE_ENCODING",task=task,count=len(d["native"]))
            ds,_=get_dataset(d["hdf5"],c.data.obs.modality,initialize_obs_utils=False,seq_len=8,demo_limit=10,train=False)
            tok,mask=b.get_task_emb(task);wrapped=SequenceVLDataset(ds,tok,mask)
            indices=[ds._demo_id_to_start_indices[r["demo_id"]]+r["start"] for r in d["native"]]
            chunks=[];cursor=0
            for data in DataLoader(Subset(wrapped,indices),batch_size=4,shuffle=False,num_workers=0):
                n=len(data["actions"]);rr=d["native"][cursor:cursor+n]
                assert np.array_equal(data["actions"][:,-1].numpy(),d["A"][[r["index"] for r in rr]])
                # Full padded action-window identity already verified against HDF5;
                # additionally bind native dataset to exact original cached sequence.
                for j,r in enumerate(rr):
                    raw=torch.load(P/"features"/str(task)/r["sample_id"],map_location="cpu")["actions"][0].numpy()
                    assert np.array_equal(data["actions"][j].numpy(),raw)
                data=TU.map_tensor(data,lambda x:x.cuda())
                chunks.append(p.spatial_encode(p.preprocess_input(data,train_mode=False)).cpu())
                cursor+=n
            d["HN"]=torch.cat(chunks);assert len(d["HN"])==len(indices)
            native_seconds+=time.time()-tic
    assert thash(p)==base
    outputs={};loads=[]
    with torch.inference_mode():
        for stage in range(6,10):
            strict_n=strict_downstream(p,cp[stage]);p.eval().requires_grad_(False)
            assert thash(p,True)==front
            checkpoint_data=torch.load(cp[stage],map_location="cpu")
            assert checkpoint_data["modal_sha256"]==sha(P/"MODAL_CONTRACT.json")
            assert checkpoint_data["contract_sha256"]==sha(P/"CONTRACT.json")
            del checkpoint_data
            before_model=thash(p)
            for task,d in taskdata.items():
                if task>stage:continue
                for view,h in (("cache",d["H"]),("native",d["HN"])):
                    tic=time.time();status("DECODING",stage=stage,task=task,view=view)
                    params=[[],[],[]]
                    for i in range(0,len(h),16):
                        z=p.temporal_encode(h[i:i+16].cuda())[:,-1]
                        mu,sd,lg=p.policy_head.forward_fn(z)
                        dist=p.policy_head(z,train=False)
                        ix=lg.argmax(-1);ix2=torch.arange(len(z),device=z.device)
                        assert torch.equal(dist.base_dist.loc,mu[ix2,ix])
                        assert torch.equal(dist.base_dist.scale,sd[ix2,ix])
                        for dst,v in zip(params,(mu,sd,lg)):dst.append(v.cpu().numpy())
                    outputs[(task,stage,view)]=[np.concatenate(x) for x in params]
                    decode_seconds+=time.time()-tic
            assert thash(p)==before_model and all(v.grad is None for v in p.parameters())
            loads.append(dict(stage=stage,strict_downstream_keys=strict_n,checkpoint_sha256=before[str(cp[stage])],
                              tensor_sha256=before_model,frontend_sha256=front,unchanged_during_inference=True))
    def selected(v):
        mu,sd,lg=v;ix=lg.argmax(1);bi=np.arange(len(ix))
        sl=np.sort(lg,axis=1)
        return mu[bi,ix].astype(np.float64),sd[bi,ix].astype(np.float64),sl[:,-1]-sl[:,-2],ix
    def describe(v,y,scale):
        m,s,margin,ix=selected(v)
        err=np.sqrt(np.mean(((m[:,:6]-y[:,:6])/scale[:6])**2,1))
        expected=np.mean(((m[:,:6]-y[:,:6])**2+s[:,:6]**2)/scale[:6]**2,1)
        correct=np.sign(m[:,6])==np.sign(y[:,6])
        prob=np.asarray([.5*math.erfc(float(np.sign(yy)*mm/ss)/math.sqrt(2)) for yy,mm,ss in zip(y[:,6],m[:,6],s[:,6])])
        return dict(error=err,expected_mse=expected,grip_correct=correct,grip_error_prob=prob,margin=margin)
    with (O/"pairs.jsonl").open("w") as f:
        count=0
        for task,d in taskdata.items():
            nv_idx=[r["index"] for r in d["native"]];nv_pos={idx:i for i,idx in enumerate(nv_idx)}
            tc=selected(outputs[(task,task,"cache")]);tn=selected(outputs[(task,task,"native")])
            crossmean=np.sqrt(np.mean(((tc[0][nv_idx,:6]-tn[0][:,:6])/d["scale"][:6])**2,1))
            crossstd=np.sqrt(np.mean(((tc[1][nv_idx,:6]-tn[1][:,:6])/d["scale"][:6])**2,1))
            for view,rows,ids in (("cache",d["rows"],list(range(len(d["rows"])))),("native",d["native"],nv_idx)):
                tv=outputs[(task,task,view)];teacher=describe(tv,d["A"][ids],d["scale"]);tm,ts,_,_=selected(tv)
                for stage in range(task+1,10):
                    sv=outputs[(task,stage,view)];student=describe(sv,d["A"][ids],d["scale"]);sm,ss,_,si=selected(sv)
                    dis=((sv[0]-tm[:,None,:])**2+(sv[1]-ts[:,None,:])**2).mean(-1);match=dis.argmin(1)
                    w2=np.mean((sm-tm)**2+(ss-ts)**2,1)
                    mm=np.sqrt(np.mean(((sm[:,:6]-tm[:,:6])/d["scale"][:6])**2,1))
                    sd=np.sqrt(np.mean(((ss[:,:6]-ts[:,:6])/d["scale"][:6])**2,1))
                    for i,r in enumerate(rows):
                        row={k:v for k,v in r.items() if k not in ("index","start","quartile")}
                        row.update(stage=stage,view=view,distribution_w2=float(w2[i]),mean_shift_scaled=float(mm[i]),
                                   std_shift_scaled=float(sd[i]),matched_is_winner=bool(match[i]==si[i]))
                        for pre,vals in (("teacher",teacher),("student",student)):
                            for k,v in vals.items():row[pre+"_"+k]=bool(v[i]) if k=="grip_correct" else float(v[i])
                        if r["native_sample"]:
                            j=nv_pos[r["index"]];row.update(teacher_cross_mean_scaled=float(crossmean[j]),teacher_cross_std_scaled=float(crossstd[j]))
                        f.write(json.dumps(row,allow_nan=False)+"\n");count+=1
    assert all(sha(p)==h for p,h in before.items()),"Read-only input mutation"
    verify()
    write(O/"FINAL_AUDIT.json",dict(status="PASS",suite=a.suite,checkpoints=loads,files_unchanged=len(before),
        cache_windows={str(t):len(d["rows"]) for t,d in taskdata.items()},
        native_windows={str(t):len(d["native"]) for t,d in taskdata.items()},pair_rows=count,
        optimizer_calls=0,optimizer_constructs=0,environment_constructs=0,environment_steps=0,policy_action_samples=0,
        native_seconds=native_seconds,decode_seconds=decode_seconds,total_seconds=time.time()-start,
        input_hashes_unchanged=True,parameters_unchanged_by_inference=True,
        no_latents_or_actions_or_reference_sidecar_saved=True))
    status("ZERO_UPDATE_DIAGNOSTIC_COMPLETE",pair_rows=count)
    print(json.dumps(js(O/"FINAL_AUDIT.json")),flush=True)

if __name__=="__main__":
    try:main()
    except BaseException:
        print(traceback.format_exc(),flush=True)
        raise
