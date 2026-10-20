"""Independent outcome/attempt/loaded-checkpoint/input/buffer audit. No rollout."""
from collections import Counter
from fractions import Fraction as F
import json
import math
from pathlib import Path
import torch
from libero.lifelong.models.brief import BriefPolicy
from ra_exp.selector_reference import anchor_ids
from .common import O,SEED,verify,sha,write
from .data import cfg
from .metrics import metrics

def lines(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def main():
    verify();root=O/"ra_coreset"
    features=json.loads((O/"FEATURES.json").read_text())["tasks"]
    assert set(features)=={str(t) for t in range(25)}
    history=json.loads((root/"epochs.json").read_text())
    assert len(history)==500
    memory={};expected_steps=0;storage=[]
    for stage in range(1,6):
        current=list(range((stage-1)*5,stage*5))
        n=sum(features[str(t)]["count"] for t in current)+len(memory)
        batches=math.ceil(n/10)
        rows=[r for r in history if r["stage"]==stage]
        assert [r["epoch"] for r in rows]==list(range(100))
        for row in rows:
            expected_steps+=batches
            assert row["batches"]==batches and row["steps"]==expected_steps
        for t in current:
            feature=features[str(t)]
            paths=sorted((O/"features"/str(t)).glob("*.pt"));ids=[p.name for p in paths]
            assert len(paths)==feature["count"] and {p.name:sha(p) for p in paths}==feature["hashes"]
            m=json.loads((root/"storage"/("task%d.json"%t)).read_text())
            assert m["before"]==memory
            assert m["K"]==math.ceil(len(ids)/2)==len(m["selected"])
            selected=[Path(r["source"]).name for r in m["selected"]]
            assert len(set(selected))==len(selected)
            assert set(anchor_ids(ids,m["K"],SEED))<=set(selected)
            assert selected==m["selection"]["selected_ids"]
            assert m["selection"]["independent_all_deletion_choices"] is True
            for r in m["selected"]:
                assert sha(r["source"])==r["sha256"]==sha(root/"buffer"/r["destination"])
                memory[r["destination"]]=r["sha256"]
            storage.append(dict(task=t,windows=len(ids),retained=m["K"],
                                manifest_sha256=sha(root/"storage"/("task%d.json"%t))))
    assert {p.name:sha(p) for p in (root/"buffer").glob("*.pt")}==memory
    c=cfg("train");checkpoints={}
    for stage in range(1,6):
        p=root/"models"/("mlrifa_stage%d_ep99.pth"%stage)
        raw=torch.load(p,map_location="cpu");model=BriefPolicy(c,c.shape_meta)
        model.load_state_dict(raw["state_dict"],strict=True)
        assert all(torch.equal(v.cpu(),raw["state_dict"][k]) and torch.isfinite(v).all()
                   for k,v in model.state_dict().items())
        checkpoints[stage]=sha(p);del model,raw
    rates={};count=0;attempts=Counter();record_hashes={}
    expected={"trials_stage%d_task%d.jsonl"%(s,t) for s in range(1,6) for t in range(5*s)}
    assert {p.name for p in (root/"evaluation").glob("trials*.jsonl")}==expected
    for stage in range(1,6):
        for task in range(stage*5):
            p=root/"evaluation"/("trials_stage%d_task%d.jsonl"%(stage,task))
            rows=lines(p);assert len(rows)==20 and {r["trial"] for r in rows}==set(range(20))
            assert all(r["seed"]==SEED and r["stage"]==stage and r["task"]==task and
                       r["initial_state_index"]==r["trial"] and type(r["success"]) is bool and
                       r["checkpoint_sha256"]==checkpoints[stage] for r in rows)
            rates[stage,task]=F(sum(r["success"] for r in rows),20);count+=20;record_hashes[p.name]=sha(p)
            a=lines(root/"evaluation"/("attempts_stage%d_task%d.jsonl"%(stage,task)))
            events=Counter(r["event"] for r in a);attempts.update(events)
            assert events==dict(environment_construct_start=1,environment_construct_complete=1,
                               vector_attempt_start=2,vector_attempt_complete=2,environment_closed=1)
            starts=[r for r in a if r["event"]=="vector_attempt_start"]
            done=[r for r in a if r["event"]=="vector_attempt_complete"]
            assert [r["env_seed"] for r in starts]==[SEED,SEED+1]
            for i,row in enumerate(done):
                assert row["successes"]==[next(r["success"] for r in rows if r["trial"]==i*10+j) for j in range(10)]
    assert count==1500 and attempts["vector_attempt_complete"]==150
    result=metrics(rates)
    write(O/"AUDIT.json",dict(status="PASS",epochs=500,optimizer_steps=expected_steps,
        strict_loaded_checkpoints=checkpoints,formal_records=count,vector_attempts=150,
        metrics=result,matrix={str(s):{str(t):float(rates[s,t]*100) for t in range(5*s)} for s in range(1,6)},
        storage=storage,attempt_events=attempts,episode_file_hashes=record_hashes,
        inputs_unchanged=True,contract_sha256=sha(O/"CONTRACT.json")))
if __name__=="__main__":main()
