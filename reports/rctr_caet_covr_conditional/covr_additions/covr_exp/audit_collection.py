"""Recompute every COVR decision from immutable branch outcomes and payload hashes."""
import json
from pathlib import Path
import torch
from coreset_exp.common import O, sha


def counts(before, after):
    pairs = [(bool(a[0]["success"]), bool(b[0]["success"]))
             for a, b in zip(before, after)]
    assert len(before) == len(after) == 3
    return sum(not a and b for a, b in pairs), sum(a and not b for a, b in pairs)


def qualifies(before, after):
    fs, sf = counts(before, after)
    return fs >= 2 and sf == 0


def js(p):
    return json.loads(p.read_text())


def audit():
    reports=[]
    for stage in (7,8,9):
        root=O/"covr"/("stage%d"%stage)
        a=js(root/"ACCEPTANCE.json")
        assert a["status"]=="PASS" and a["optimizer_calls"]==0
        binding=js(O/"ra_coreset/covr_bindings"/("stage%d.json"%stage))
        assert sha(root/"sidecar.pth")==a["sidecar_sha256"]==binding["sidecar_sha256"]
        assert sha(root/"ACCEPTANCE.json")==binding["acceptance_sha256"]
        decisions=js(root/"DECISIONS.json")
        branches={p.stem:js(p) for p in (root/"committed").glob("*.json")}
        assert len(decisions)==6*(stage-6) and len(branches)==a["unique_branches"]
        assert sum(r["phase"]=="exploration" for r in branches.values())<=54*(stage-6)
        assert sum(r["phase"]=="confirmation" for r in branches.values())<=27*(stage-6)
        assert all(r["steps"]<=600-r["snapshot_step"] for r in branches.values())
        for task in range(6,stage):
            expected=[]
            for initial in range(20,26):
                key="t%d_i%d"%(task,initial)
                n=js(root/"normal"/(key+".json"))
                assert n==decisions[key]["normal"] and n["steps"]<=600
                if not n["success"] and n["snapshot_available"]:
                    expected.append(key)
            actual=[key for key in expected if decisions[key]["selected"]]
            assert actual==expected[:3]
        used=0
        for key,decision in decisions.items():
            if not decision["selected"]:
                continue
            task=decision["normal"]["task"];initial=decision["normal"]["initial"]
            meta=decision["certificate"]
            def get(phase,mode):
                rows=[]
                for rep in range(3):
                    k="s%d_t%d_i%d_%s_r%d_%s"%(stage,task,initial,phase,rep,str(mode))
                    rows.append((branches[k],None))
                return rows
            normal=get("exploration","normal")
            winner=normal[0][0]["winner"]
            means={m:get("exploration",m) for m in range(5)}
            for rep in range(3):
                assert all(means[m][rep][0]["noise_sha256"] ==
                           normal[rep][0]["noise_sha256"] for m in range(5))
            eligible=[]
            for m in range(5):
                if m==winner:
                    continue
                if qualifies(normal,means[m]) and qualifies(means[winner],means[m]):
                    if all(means[m][r][0]["first_action_sha256"] not in (
                        normal[r][0]["first_action_sha256"],means[winner][r][0]["first_action_sha256"])
                        for r in range(3)):
                        eligible.append((-(counts(normal,means[m])[0]+counts(means[winner],means[m])[0]),m))
            if not eligible:
                assert meta["status"]=="NO_EXPLORATION_CANDIDATE"
            else:
                chosen=min(eligible)[1]
                assert chosen==meta["chosen"]
                nc,mc,cc=get("confirmation","normal"),get("confirmation",winner),get("confirmation",chosen)
                for rep in range(3):
                    assert nc[rep][0]["noise_sha256"]==mc[rep][0]["noise_sha256"]==cc[rep][0]["noise_sha256"]
                passes=qualifies(nc,cc) and qualifies(mc,cc)
                distinct=all(cc[r][0]["first_action_sha256"] not in (
                    nc[r][0]["first_action_sha256"],mc[r][0]["first_action_sha256"]) for r in range(3))
                assert (meta["status"]=="CERTIFIED")==(passes and distinct)
            payload=root/"selected"/(key+".pth")
            assert sha(payload)==decision["payload_sha256"]
            records=torch.load(payload,map_location="cpu")
            if meta["status"]!="CERTIFIED":
                assert not records
            else:
                assert 1<=len(records)<=8
                assert all(r["weight"]==(1. if i==0 else .25) and r["continuation_step"]==i
                           for i,r in enumerate(records))
                assert all(r["branch_key"]==meta["training_branch_key"] for r in records)
            used+=len(records)
        assert used==a["training_windows"]
        allrecords=torch.load(root/"sidecar.pth",map_location="cpu")
        assert len(allrecords)==used
        if used:
            assert a["real_gradient"]["status"]=="PASS"
        reports.append(a)
    return reports
