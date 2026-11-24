"""Offline paired evaluation in robot-base XY with identical robot state and action anchor."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

from annotate import images
from assess import geometry
from train_contrast import MANIFEST, build_variant, image_rgb


def write(path,data):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False))
    tmp.replace(path)


@torch.inference_mode()
def evaluate(work,out,variant):
    meta,records,goals,xy=geometry(work,out)
    by_id={r["episode"]:r for r in records}
    root=work/"offline_diagnostics/b1_ode_lerobot_20261123"
    sys.path.insert(0,str(root/"code"))
    from ode_policy_adapter import build_policy
    path=root/"run/epoch_50.pt" if variant=="baseline" else out/variant/"epoch_50.pt"
    digest=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            digest.update(chunk)
    payload=torch.load(path,map_location="cpu",weights_only=True)
    assert payload["manifest_sha256"]==MANIFEST and payload["global_step"]==15250 and payload["epoch"]==50
    assert payload["config"].get("visual_intervention","baseline")==variant
    model=build_policy(payload["stats"],payload["config"]) if variant=="baseline" else build_variant(
        build_policy,payload["stats"],payload["config"],variant)
    model.load_state_dict(payload["ema"],strict=True)
    model.eval()
    torch.set_num_threads(2)
    hist={}
    natural=[]
    contrasts=[]
    counter_modes=("overhead","local_overhead","both")
    modes=("zero","field_seed42")
    for phase,offset in [("approach",18),("closing",3)]:
        for r in records:
            j=r["close"]-offset
            key=r["episode"]
            obs={"agent_pos":torch.from_numpy(r["q"][None,j-7:j+1])}
            for role in ("overhead","wrist"):
                side=256 if variant=="high" and role=="overhead" else 128
                files=images(r["folder"],role)[r["first"]+j-7:r["first"]+j+1]
                assert len(files)==8
                pixels=np.stack([image_rgb(cv2.imread(str(f)),side) for f in files])
                obs[role]=torch.from_numpy(pixels[None]).float().div_(255)
            hist[key]=(obs,torch.from_numpy(r["cmd"][None,j-1]))
        def endpoints(obs,anchor):
            results={}
            for mode in modes:
                torch.manual_seed(42)
                eps=torch.zeros_like(anchor) if mode=="zero" else None
                traj=model.predict_action(obs,a0=anchor,eps=eps,eps_center=True)["action"][0].numpy()
                results[mode]=xy(traj[-1])
            return results
        original={}
        for key,(obs,anchor) in hist.items():
            r=by_id[key]
            j=r["close"]-offset
            goal=goals[key]
            initial=xy(anchor[0].numpy())
            ori=endpoints(obs,anchor)
            original[key]=ori
            dist=np.linalg.norm(goal-initial)
            for mode,endpoint in {**ori,"expert":xy(r["cmd"][j+7]),"hold":initial}.items():
                natural.append(dict(episode=key,phase=phase,mode=mode,initial_distance_mm=float(dist),
                    end_distance_mm=float(np.linalg.norm(goal-endpoint)),
                    advance_mm=float(dist-np.linalg.norm(goal-endpoint))))
        for a,b in meta["pairs"]:
            for recipient,donor in [(a,b),(b,a)]:
                obs,anchor=hist[recipient]
                other=hist[donor][0]
                delta=goals[donor]-goals[recipient]
                direction=delta/np.linalg.norm(delta)
                actual_delta=xy(by_id[donor]["q"][by_id[donor]["contact"]])-xy(by_id[recipient]["q"][by_id[recipient]["contact"]])
                for kind in counter_modes:
                    changed=dict(obs)
                    if kind=="local_overhead":
                        # Same rectangles in original-camera coordinates across resolutions. May include a finger or shadow.
                        result=obs["overhead"].clone()
                        for key in (recipient,donor):
                            u,v=by_id[key]["uv"]
                            factor=result.shape[-1]/128
                            x=int(round(u/5)); y=int(round((v+80)/5))
                            sx=slice(int((x-6)*factor),int((x+7)*factor))
                            sy=slice(int((y-6)*factor),int((y+7)*factor))
                            result[:,:,:,sy,sx]=other["overhead"][:,:,:,sy,sx]
                        changed["overhead"]=result
                    else:
                        changed["overhead"]=other["overhead"]
                        if kind=="both":
                            changed["wrist"]=other["wrist"]
                    for mode,end in endpoints(changed,anchor).items():
                        shift=end-original[recipient][mode]
                        contrasts.append(dict(pair=[a,b],recipient=recipient,donor=donor,phase=phase,mode=mode,view=kind,
                            separation_mm=float(np.linalg.norm(delta)),signed_mm=float(shift@direction),
                            contact_direction_signed_mm=float(shift@actual_delta/max(1e-8,np.linalg.norm(actual_delta))),
                            orthogonal_mm=float(shift@np.array([-direction[1],direction[0]]))))
    return dict(variant=variant,sha256=digest.hexdigest(),geometry=meta,natural=natural,contrasts=contrasts)


def summarize(out):
    res={name:json.loads((out/(name+"_results.json")).read_bytes()) for name in ("baseline","high","shared")}
    base=res["baseline"]
    assert all(r["geometry"]==base["geometry"] for r in res.values())
    table={}
    audit_pairs={}
    def paired_scores(r,phase,mode,view):
        pairs=r["geometry"]["pairs"]
        v=[x for x in r["contrasts"] if (x["phase"],x["mode"],x["view"])==(phase,mode,view)]
        samples=[]
        for p in pairs:
            rows=[x for x in v if x["pair"]==p]
            assert len(rows)==2 and {x["recipient"] for x in rows}==set(p)
            samples.append([np.mean([x[k] for x in rows]) for k in ("signed_mm","contact_direction_signed_mm","separation_mm")])
        return np.array(samples)
    for k,r in res.items():
        result={}
        audit_pairs[k]={view:paired_scores(r,"approach","zero",view).round(8).tolist()
                        for view in ("overhead","local_overhead","both")}
        for phase in ("approach","closing"):
            for mode in ("zero","field_seed42"):
                for view in ("overhead","local_overhead","both"):
                    s,contact,sep=paired_scores(r,phase,mode,view).T
                    ref=paired_scores(base,phase,mode,view)[:,0]
                    delta=s-ref
                    boot=np.random.default_rng(20261124).integers(len(s),size=(10000,len(s)))
                    result["/".join([phase,mode,view])]=dict(pairs=len(s),positive=int(sum(s>0)),toward_gt_1mm=int(sum(s>1)),
                        away_gt_1mm=int(sum(s< -1)),median_signed_mm=float(np.median(s)),mean_signed_mm=float(np.mean(s)),
                        median_difference_vs_base_mm=float(np.median(delta)),
                        median_difference_CI95_mm=np.percentile(np.median(delta[boot],axis=1),[2.5,97.5]).tolist(),
                        contact_median_mm=float(np.median(contact)),median_gain_fraction=float(np.median(s/sep)),
                        wide_pair_count=int(sum(sep>=30)),wide_pair_median_mm=float(np.median(s[sep>=30])))
        for phase in ("approach","closing"):
            for mode in ("zero","field_seed42","expert","hold"):
                rows=[x for x in r["natural"] if (x["phase"],x["mode"])==(phase,mode)]
                v=np.array([x["advance_mm"] for x in rows])
                result["/".join([phase,mode,"natural"])]=dict(n=len(v),closer=int(sum(v>0)),
                    median_advance_mm=float(np.median(v)),median_end_error_mm=float(np.median([x["end_distance_mm"] for x in rows])))
        table[k]=result
    return dict(geometry=base["geometry"],checkpoint_ids={k:r["sha256"] for k,r in res.items()},metrics=table,
                approach_pairs_zero=audit_pairs, pair_columns=["signed_mm","signed_vs_expert_contact_mm","separation_mm"])


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("work",type=Path)
    parser.add_argument("out",type=Path)
    parser.add_argument("variant",choices=["baseline","high","shared","summary"])
    args=parser.parse_args()
    cv2.setNumThreads(1)
    if args.variant=="summary":
        write(args.out/"final.json",summarize(args.out))
    else:
        write(args.out/(args.variant+"_results.json"),evaluate(args.work,args.out,args.variant))
