"""Independent local metadata-only recomputation; no torch, data or weights."""
import argparse
import csv
import hashlib
import json
import statistics
from fractions import Fraction as F
from pathlib import Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit(root,seed):
    manifest=json.loads((root/"PACKAGE_MANIFEST.json").read_text())
    assert manifest["status"]=="REVIEWED_METADATA_ONLY"
    for rel,proof in manifest["files"].items():
        p=(root/rel).resolve()
        assert root.resolve() in p.parents and p.suffix in (".json",".jsonl",".exit")
        assert p.stat().st_size==proof["bytes"] and sha(p)==proof["sha256"]
    for name in ("pipeline.exit","supervisor.exit"):
        assert (root/name).read_text().strip()=="0"
    acceptance=json.loads((root/"PROCESS_EXIT_ACCEPTANCE.json").read_text())
    remote=json.loads((root/"AUDIT.json").read_text())
    assert acceptance["status"]==remote["status"]=="PASS"
    assert acceptance["audit_sha256"]==sha(root/"AUDIT.json")
    epochs=json.loads((root/"ra_coreset/epochs.json").read_text())
    assert len(epochs)==500 and sum(x["batches"] for x in epochs)==remote["optimizer_steps"]
    assert {(x["stage"],x["epoch"]) for x in epochs}=={(s,e) for s in range(1,6) for e in range(100)}
    rates={};rows=[];total=0
    for s in range(1,6):
        for t in range(s*5):
            name="trials_stage%d_task%d.jsonl"%(s,t);p=root/"ra_coreset/evaluation"/name
            trials=[json.loads(x) for x in p.read_text().splitlines() if x]
            assert len(trials)==20 and {x["trial"] for x in trials}==set(range(20))
            assert sha(p)==remote["episode_file_hashes"][name]
            assert all(x["seed"]==seed and x["stage"]==s and x["task"]==t and
                       x["initial_state_index"]==x["trial"] and type(x["success"]) is bool
                       for x in trials)
            assert all(x["checkpoint_sha256"]==remote["strict_loaded_checkpoints"][str(s)] for x in trials)
            attempts=[json.loads(x) for x in (p.parent/("attempts_stage%d_task%d.jsonl"%(s,t))).read_text().splitlines() if x]
            assert len(attempts)==7
            assert [x["event"] for x in attempts]==[
                "environment_construct_start","environment_construct_complete",
                "vector_attempt_start","vector_attempt_complete",
                "vector_attempt_start","vector_attempt_complete","environment_closed"]
            for j in range(2):
                start,done=attempts[2+2*j:4+2*j]
                assert start["env_seed"]==done["env_seed"]==seed+j
                assert start["initial_indices"]==list(range(10*j,10*(j+1)))
                assert done["successes"]==[next(x["success"] for x in trials if x["trial"]==10*j+i) for i in range(10)]
            successes=sum(x["success"] for x in trials);rates[s,t]=F(successes,20);total+=20
            rows.append(dict(seed=seed,stage=s,task=t,successes=successes,trials=20,percent=5*successes))
    assert total==1500 and len(rows)==75
    auc=sum((sum((rates[s,t] for s in range(t//5+1,6)),F(0))/(5-t//5) for t in range(25)),F(0))/25
    fwt=sum((rates[t//5+1,t] for t in range(25)),F(0))/25
    nbt=sum((sum((rates[t//5+1,t]-rates[s,t] for s in range(t//5+2,6)),F(0))/(4-t//5) for t in range(20)),F(0))/20
    metrics=dict(AUC=auc,FWT=fwt,NBT=nbt,
        OldFinal=sum((rates[5,t] for t in range(20)),F(0))/20,
        FinalAll=sum((rates[5,t] for t in range(25)),F(0))/25)
    for k,v in metrics.items():assert F(remote["metrics"][k]["fraction"])==v
    return dict(seed=seed,metrics={k:dict(fraction=str(v),percent=float(v*100)) for k,v in metrics.items()},
                records=total,manifest_sha256=sha(root/"PACKAGE_MANIFEST.json")),rows
def main():
    ap=argparse.ArgumentParser()
    for s in (41,42,43):ap.add_argument("--seed%d"%s,required=True,type=Path)
    ap.add_argument("--out",type=Path,required=True);a=ap.parse_args()
    assert not a.out.exists();a.out.mkdir(parents=True)
    results=[];rows=[]
    for s in (41,42,43):
        result,matrix=audit(getattr(a,"seed%d"%s),s);results.append(result);rows+=matrix
    summary={}
    for k in results[0]["metrics"]:
        values=[x["metrics"][k]["percent"] for x in results]
        summary[k]=dict(mean=statistics.mean(values),sample_sd=statistics.stdev(values),ddof=1)
    output=dict(status="LOCAL_INDEPENDENT_PASS",records=4500,vector_attempts=450,
        seeds=results,summary=summary,no_matched_random_or_coreset_baseline=True)
    (a.out/"AUDIT.json").write_text(json.dumps(output,indent=2)+"\n")
    with (a.out/"ALL_TASKS.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(output))
if __name__=="__main__":main()
