from sazr_exp.common import *
from sazr_exp.native import Readout
from collections import Counter
verify();setup()
rows={};attempts={}
for phase,n in (("anchor",10),("validation_parent",10),("validation_candidate",10),("formal",20)):
    rs=[json.loads(l) for l in (O/phase/"episodes.jsonl").read_text().splitlines()]
    assert len(rs)==n
    ids=list(range(20,30)) if phase=="anchor" else list(range(30,40)) if phase.startswith("validation") else list(range(20))
    assert sorted(r["initial_index"] for r in rs)==ids
    events=Counter(json.loads(l)["event"] for l in (O/phase/"attempts.jsonl").read_text().splitlines())
    assert events==Counter(environment_construct_start=1,environment_construct_complete=1,environment_closed=1,vector_start=n//10,vector_complete=n//10)
    assert all(r["parent_sha256"]==sha(CP) and r["initial_sha256"]==js(O/"BINDING.json")["initial_hashes"][r["initial_index"]] for r in rs)
    rows[phase]=rs;attempts[phase]=dict(events)
decision=js(O/"DECISION.json");a,b=rows["validation_parent"],rows["validation_candidate"]
counts=Counter(("S" if x["success"] else "F")+("S" if y["success"] else "F") for x,y in zip(a,b))
assert decision["enabled"]==(counts["FS"]>=2 and counts["SF"]==0)
for x,y in zip(a,b):
    for k in ("initial_index","noise_sha256","noise_seed","env_seed"):assert x[k]==y[k]
for f,h in decision["files"].items():assert sha(f)==h
cp=torch.load(O/"models/sazr_stage6_calibration_ep9.pth",map_location="cpu")
assert cp["steps"]==1570 and cp["next_epoch"]==10
bank=make_bank(cp["bank"]["Q"]);bank.load_state_dict(cp["bank"],strict=True)
ep=js(O/"epochs.json");assert len(ep)==10 and sum(r["samples"] for r in ep)==15620
st=[json.loads(l) for l in (O/"STRUCTURE.jsonl").read_text().splitlines()];assert len(st)==11 and all(r["passed"] for r in st)
p=new_public();assert tensor_hash(p)==cp["public_hash"]
for f,h in js(O/"ANCHOR_CONTRACT.json")["files"].items():assert sha(f)==h
assert sha(O/"models/sazr_stage6_calibration_ep9.pth")==decision["checkpoint_sha256"]
# Actual head bypass for canonical base tasks, unknown tokens and disabled current.
gen=torch.Generator().manual_seed(123);z=torch.randn(9,512,generator=gen).cuda()
e=torch.from_numpy(np.load(W/"language.npz")["canonical"]).cuda()
language=torch.cat([e,torch.ones_like(e[:1]),torch.zeros_like(e[:1])])
ctx={"h":language[:,None,None,:]};head=Readout(p.policy_head,bank,False,ctx).eval()
with torch.no_grad():
    original=p.policy_head(z,train=False);fallback=head(z)
    assert torch.equal(original.base_dist.loc,fallback.base_dist.loc) and torch.equal(original.base_dist.scale,fallback.base_dist.scale)
    rr=cpu(rng());act=original.sample();restore_rng(rr);assert torch.equal(act,fallback.sample())
    head.enabled=True
    ids=[0,1,2,3,4,5,7,8];ctx["h"]=language[ids,None,None,:]
    bypass=head(z[ids]);base=p.policy_head(z[ids],train=False)
    assert torch.equal(bypass.base_dist.loc,base.base_dist.loc) and torch.equal(bypass.base_dist.scale,base.base_dist.scale)
write(O/"AUDIT.json",dict(status="PASS",epochs=10,steps=1570,checkpoint_load="strict PASS",structure_checks=11,
    episodes=50,vector_attempts=5,attempts=attempts,enabled=decision["enabled"],validation_counts=dict(counts),
    successes={n:sum(r["success"] for r in rs) for n,rs in rows.items()},formal_rate=sum(r["success"] for r in rows["formal"])/20,
    historical_parent_rate=.6,formal_delta_pp=(sum(r["success"] for r in rows["formal"])/20-.6)*100,
    fallback_distribution_and_sample_exact=True,base_and_unknown_bypass_exact=True,
    input_hashes_pass=True,anchor_max_response=max(r["anchor_response_max"] for r in st),
    max_candidate_delta=max(r["dataset_delta_max"] for r in st)))
print("AUDIT PASS",decision["enabled"])
