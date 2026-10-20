"""Build reviewed text deployments and OBJECT entries; no experiment execution."""
from pathlib import Path
import hashlib
import json
import shlex

R=Path(__file__).resolve().parent
Q="/data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    bundle={}
    for folder in ("source","caet_additions","covr_additions"):
        for p in sorted((R/folder).rglob("*")):
            if not p.is_file() or p.suffix not in (".py",".yaml",".yml",".sh"):
                continue
            if "__pycache__" in p.parts:
                continue
            text=p.read_text(encoding="utf8")
            bundle[p.relative_to(R/folder).as_posix()]=dict(text=text,sha256=digest(text.encode()))
    text=(R/"test_auxiliary.py").read_text()
    bundle["test_auxiliary.py"]=dict(text=text,sha256=digest(text.encode()))
    raw=(json.dumps(bundle,ensure_ascii=True)+"\n").encode()
    (R/"OBJECT_WORKER_BUNDLE.json").write_bytes(raw)
    output=R/"object_registry";output.mkdir(exist_ok=True)
    registry={}
    shared={Q+"/"+p:digest((R/p).read_bytes()) for p in (
        "worker_entrypoint.py","queue_controller.py","queue_verifier.py","branch_gate.py")}
    for method in ("RCTR","CAET","COVR"):
        for seed in (42,41,43):
            key="%s_OBJECT_%d"%(method,seed)
            if key=="RCTR_OBJECT_42":
                registry[key]=dict(method=method,suite="OBJECT",seed=seed,
                    source="/data/mlr_ifa/experiment_workspaces/RCTR_OBJECT_seed42_full_20261021/source",
                    output="/data/mlr_ifa/experiments/RCTR_OBJECT_seed42_full_20261021",
                    baseline="/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed42_20261019",
                    baseline_acceptance="/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed42_20261019/AUTONOMOUS_EXIT_ACCEPTANCE.json",
                    implementation_reviewed=True,existing_run=True)
                continue
            workspace=Q+"/workers/"+key
            e=dict(method=method,suite="OBJECT",seed=seed,source=workspace+"/source",
                output="/data/mlr_ifa/experiments/%s_OBJECT_seed%d_full_20261021"%(method,seed),
                baseline="/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed%d_20261019"%seed,
                baseline_acceptance="/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed%d_20261019/AUTONOMOUS_EXIT_ACCEPTANCE.json"%seed,
                implementation_reviewed=True,bundle=Q+"/OBJECT_WORKER_BUNDLE.json",bundle_sha256=digest(raw))
            entryraw=(json.dumps(e,indent=2)+"\n").encode()
            (output/(key+".json")).write_bytes(entryraw)
            command="#!/usr/bin/env bash\nset -euo pipefail\nsource /data/mlr_ifa/activate.sh\nexec /data/mlr_ifa/env/bin/python "+shlex.quote(Q+"/worker_entrypoint.py")+" "+shlex.quote(Q+"/entries/"+key+".json")+"\n"
            (output/(key+".sh")).write_bytes(command.encode())
            e.update(launcher=Q+"/entries/"+key+".sh",launcher_sha256=digest(command.encode()),
                deployment_hashes=dict(shared,**{Q+"/OBJECT_WORKER_BUNDLE.json":digest(raw),
                    Q+"/entries/"+key+".json":digest(entryraw)}))
            registry[key]=e
    (output/"REGISTRY.json").write_text(json.dumps(registry,indent=2)+"\n")
    print(json.dumps(dict(status="PREPARED_NOT_LAUNCHED",entries=len(registry),
        source_files=len(bundle),bundle_sha256=digest(raw))))


if __name__=="__main__":
    main()
