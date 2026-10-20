"""Post-training inference only on existing demonstrations; no environment."""
import json,copy,time
from pathlib import Path
from sazr_exp import common as old
from gripper_core import GripperHead,install
import torch,numpy as np
import robomimic.utils.tensor_utils as TU
from torch.utils.data import DataLoader,Subset
from libero.lifelong.datasets import get_dataset,SequenceVLDataset

O=Path("/data/mlr_ifa/experiments/GripperRepair_A_OBJECT_seed41_20261020")
old.verify();old.setup()
contract=old.js(O/"CONTRACT.json")
for f,h in contract["files"].items():assert old.sha(f)==h
cp=O/"models/gripper_ep9.pth";before=old.sha(cp)
canon=torch.from_numpy(np.load(old.W/"language.npz")["canonical"])
head=GripperHead(canon).cuda()
head.load_state_dict(torch.load(cp,map_location="cpu")["head"],strict=True)
head.eval().requires_grad_(False)
c=old.cfg("eval");bench,_,shape=old.data_setup(c,False);c.shape_meta=shape
ds,_=get_dataset(str(Path(c.folder)/bench.get_task_demonstration(6)),c.data.obs.modality,
                initialize_obs_utils=False,seq_len=8,demo_limit=10,train=False)
tok,mask=bench.get_task_emb(6)
wrapped=SequenceVLDataset(ds,tok,mask)
p=old.complete(c,shape).eval().requires_grad_(False);old.strict_downstream(p,old.CP)
ph=old.tensor_hash(p);ctx,original,handle=install(p,head)
binding=old.js(old.P/"bindings/task6.json")["windows"]
rr=old.js(O/"WINDOW_OUTCOMES.json")
chosen=[r for r in rr if not r["parent_correct"] or
        (r["split"]=="holdout" and r["parent_correct"] and not r["candidate_correct"])]
out=[]
for r in chosen:
    br=binding[r["sample_id"]];starts=br["possible_starts"]
    if len(starts)!=1:
        out.append(dict(sample_id=r["sample_id"],status="SKIPPED_AMBIGUOUS_START",possible_start_count=len(starts)))
        continue
    index=ds._demo_id_to_start_indices[r["demo_id"]]+starts[0]
    data=TU.map_tensor(next(iter(DataLoader(Subset(wrapped,[index]),batch_size=1))),lambda x:x.cuda())
    label=float(data["actions"][0,-1,6])
    assert label==r["target_class"]
    rnd=old.cpu(old.rng())
    with torch.no_grad():
        parent_action=original(copy.deepcopy(data))
        z=ctx["z"];b,s,l=old.raw(p.policy_head,z)
        parentmean=b[0,l[0].argmax(),6].tanh()
    old.restore_rng(rnd)
    with torch.no_grad():candidate=p.get_action(copy.deepcopy(data))
    assert np.array_equal(parent_action[:,:6],candidate[:,:6])
    assert bool(ctx["route"][0])
    logit=float(ctx["logit"][0])
    expected=parent_action[0,6] if logit==0 else (1 if logit>0 else -1)
    assert candidate[0,6]==expected
    out.append(dict(sample_id=r["sample_id"],demo_id=r["demo_id"],split=r["split"],status="PASS",
                    cache_parent_correct=r["parent_correct"],cache_candidate_correct=r["candidate_correct"],
                    raw_image_parent_mean_correct=bool(float(parentmean.sign())==label),
                    raw_image_parent_sample_correct=bool(np.sign(parent_action[0,6])==label),
                    raw_image_candidate_correct=bool(candidate[0,6]==label),motion_exact=True,
                    route_current=True,command_matches_classifier=True))
assert old.tensor_hash(p)==ph and old.sha(cp)==before
for f,h in contract["files"].items():assert old.sha(f)==h
old.verify()
print("DIAGNOSTIC_JSON="+json.dumps(dict(status="PASS",optimizer_calls=0,environment_calls=0,
    checkpoint_unchanged=True,source_inputs_unchanged=True,rows=out,
    note="Native get_action on real stored images, not simulator execution; matching actions do not prove native image features equal cached augmented features.")))
