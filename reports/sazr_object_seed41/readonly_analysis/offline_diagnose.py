"""Read existing weights/data; inference only, no optimizer or environment."""
import json
import hashlib
import time
from sazr_exp.common import *
from torch.utils.data import DataLoader
from libero.lifelong.algos.mlr import build_dataset_for_task

tic = time.time()
verify()
tracked = [O/"models/sazr_stage6_calibration_ep9.pth", O/"anchor_basis.pth",
           O/"successful_parent_features.pth", O/"DECISION.json"]
before = {str(f): sha(f) for f in tracked}
setup()
p = new_public()
ph = tensor_hash(p)
cp = torch.load(tracked[0], map_location="cpu")
bank = make_bank(cp["bank"]["Q"])
bank.load_state_dict(cp["bank"], strict=True)
bank.eval().requires_grad_(False)
assert ph == cp["public_hash"]
ds = build_dataset_for_task(str(P/"features"), 6, str(O/"empty_buffer"), 41, start=6)
assert len(ds) == 1562
col = {}
def add(k, v):
    col.setdefault(k, []).append(v.detach().double().cpu())
def stats(v):
    v = v.double().flatten()
    return dict(n=len(v), mean=float(v.mean()), min=float(v.min()), max=float(v.max()),
                quantiles={str(q): float(torch.quantile(v,q)) for q in (0.,.25,.5,.75,.9,.95,.99,1.)})
with torch.inference_mode():
    for samples, _ in DataLoader(ds, batch_size=32, shuffle=False, num_workers=0):
        h = samples["feature"].squeeze(1).cuda()
        a = samples["actions"].squeeze(1).cuda()[:, -1]
        out = decode(p,bank,h)
        z = out["z"].double()
        projected = z-z@bank.Q@bank.Q.T
        idx = out["selected"]
        rr = torch.arange(len(z),device=z.device)
        base = out["b"][rr,idx].tanh()
        new = out["means"][rr,idx]
        scale = out["s"][rr,idx]
        add("projection_norm_ratio",projected.norm(dim=-1)/z.norm(dim=-1))
        add("z_energy",z.square().sum(-1))
        add("projected_energy",projected.square().sum(-1))
        add("delta",out["delta"])
        add("mean_change",new-base)
        add("mean_change_over_sigma",(new-base)/scale)
        add("parent_abs_error",(base-a).abs())
        add("candidate_abs_error",(new-a).abs())
        add("gaussian_KL",.5*((new-base)/scale).square().sum(-1))
        add("tanh_slope",1-base.square())
        add("selected_sigma",scale)
        add("gripper_sign_changed",torch.sign(base[:,-1])!=torch.sign(new[:,-1]))
        add("gripper_target_sign",torch.sign(a[:,-1]))
        add("route",bank.route(h[:,-1,0]))
    col = {k:torch.cat(v) for k,v in col.items()}
    ea=col["parent_abs_error"].mean(-1);eb=col["candidate_abs_error"].mean(-1)
    gain=ea-eb
    corpus = dict(samples=len(ea), parent_L1=float(ea.mean()), candidate_L1=float(eb.mean()),
                  relative_L1_reduction=float(gain.mean()/ea.mean()),
                  improved=int((gain>1e-9).sum()),worsened=int((gain < -1e-9).sum()),
                  unchanged=int((gain.abs()<=1e-9).sum()),
                  retained_feature_energy=float(col["projected_energy"].sum()/col["z_energy"].sum()),
                  metrics={k:stats(v.abs() if k in ("delta","mean_change","mean_change_over_sigma") else v)
                           for k,v in col.items() if k not in ("z_energy","projected_energy")},
                  by_dimension={str(j):dict(parent_L1=float(col["parent_abs_error"][:,j].mean()),
                     candidate_L1=float(col["candidate_abs_error"][:,j].mean()),
                     mean_abs_shift=float(col["mean_change"][:,j].abs().mean()),
                     max_abs_shift=float(col["mean_change"][:,j].abs().max()),
                     mean_abs_shift_over_sigma=float(col["mean_change_over_sigma"][:,j].abs().mean()))
                     for j in range(7)})
    trace_results={}
    e=torch.from_numpy(np.load(W/"language.npz")["canonical"][6:7]).cuda()
    success=torch.load(O/"successful_parent_features.pth",map_location="cpu")
    for i,zz in success.items():
        z=zz.cuda();proj=z.double()-z.double()@bank.Q@bank.Q.T
        delta=bank(z,e.expand(len(z),-1))
        b,s,l=raw(p.policy_head,z);ix=l.argmax(-1);r=torch.arange(len(z),device=z.device)
        diff=(b[r,ix]+delta).tanh()-b[r,ix].tanh()
        trace_results[str(i)]=dict(
            note="existing successful parent trajectory features; no candidate rollout",
            steps=len(z),projection_norm_ratio=stats(proj.norm(dim=-1)/z.double().norm(dim=-1)),
            delta_abs=stats(delta.abs()),mean_change_abs=stats(diff.abs()),
            gaussian_KL=stats(.5*(diff/s[r,ix]).square().sum(-1)),
            nonzero_mean_rows=int((diff.abs().amax(-1)>1e-6).sum()))
verify()
assert tensor_hash(p)==ph
after={str(f):sha(f) for f in tracked}
assert before==after
print("DIAGNOSTIC_JSON="+json.dumps(dict(
    status="PASS", mode="read_only_offline_inference", optimizer_calls=0, environment_calls=0,
    source_contract_unchanged=True, artifact_hashes_before=before,artifact_hashes_after=after,
    parent_tensor_hash=ph, corpus=corpus,successful_parent_features=trace_results,
    seconds=time.time()-tic),allow_nan=False))
