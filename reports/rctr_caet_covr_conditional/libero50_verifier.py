"""External independent record/count/exits verifier for the LIBERO50 queue."""
import hashlib
import json
import os
from fractions import Fraction as F
from pathlib import Path
from queue_verifier import sha

def verify_libero50(e):
    out=Path(e["output"]);source=Path(e["source"])
    for name in ("pipeline.exit","supervisor.exit"):
        assert (out/name).read_text().strip()=="0"
    accept=json.loads((out/"PROCESS_EXIT_ACCEPTANCE.json").read_text())
    audit=json.loads((out/"AUDIT.json").read_text())
    assert accept["status"]==audit["status"]=="PASS"
    assert accept["audit_sha256"]==sha(out/"AUDIT.json")
    contract=json.loads((out/"CONTRACT.json").read_text())
    assert accept["contract_sha256"]==sha(out/"CONTRACT.json")
    actual={str(p.relative_to(source)):sha(p) for p in sorted(source.rglob("*"))
            if p.is_file() and p.suffix in (".py",".yaml",".yml",".sh") and "__pycache__" not in p.parts}
    assert actual==contract["source_hashes"]
    for p,h in contract["files"].items():assert sha(p)==h,p
    for p in Path("/proc").glob("[0-9]*"):
        try:
            if int(p.name)==os.getpid():continue
            env=(p/"environ").read_bytes().split(b"\0")
            cmd=(p/"cmdline").read_bytes()
            if ("LIBERO50_OUT="+str(out)).encode() in env:
                assert not any(x in cmd for x in (b"l50_exp.train",b"l50_exp.evaluate",
                    b"l50_exp.export",b"l50_exp.pipeline",b"torch.distributed.run")),(p.name,cmd)
        except (OSError,ProcessLookupError):pass
    rates={};hashes={};count=0
    for stage in range(1,6):
        for t in range(5*stage):
            p=out/"ra_coreset/evaluation"/("trials_stage%d_task%d.jsonl"%(stage,t))
            rows=[json.loads(x) for x in p.read_text().splitlines() if x]
            assert len(rows)==20 and {r["trial"] for r in rows}==set(range(20))
            assert all(r["seed"]==e["seed"] and r["stage"]==stage and r["task"]==t and
                       type(r["success"]) is bool for r in rows)
            rates[stage,t]=F(sum(r["success"] for r in rows),20)
            hashes[p.name]=sha(p);count+=len(rows)
    # Direct task-level formula, independent of the implementation's group table.
    auc=sum((sum((rates[s,t] for s in range(t//5+1,6)),F(0))/(5-t//5)
             for t in range(25)),F(0))/25
    fwt=sum((rates[t//5+1,t] for t in range(25)),F(0))/25
    nbt=sum((sum((rates[t//5+1,t]-rates[s,t] for s in range(t//5+2,6)),F(0))/(4-t//5)
             for t in range(20)),F(0))/20
    for name,v in dict(AUC=auc,FWT=fwt,NBT=nbt).items():
        assert F(audit["metrics"][name]["fraction"])==v
    assert count==audit["formal_records"]==1500
    assert audit["epochs"]==500 and audit["vector_attempts"]==150
    assert len(audit["strict_loaded_checkpoints"])==5
    for stage,h in audit["strict_loaded_checkpoints"].items():
        assert sha(out/"ra_coreset/models"/("mlrifa_stage%s_ep99.pth"%stage))==h
    assert hashes==audit["episode_file_hashes"]
    return dict(engineering="PASS",acceptance_sha256=sha(out/"PROCESS_EXIT_ACCEPTANCE.json"),
        episodes_sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
        auc="%d/%d"%(auc.numerator,auc.denominator),metrics=audit["metrics"],
        formal_records=count,optimizer_steps=audit["optimizer_steps"],
        origin="shared newly reproduced pretraining; no matched released LIBERO50 baseline")
