"""Post-exit read-only process and lightweight output attestation."""
from pathlib import Path
import os,json,hashlib,subprocess,time
W=Path(__file__).resolve().parent;A=Path("/data/mlr_ifa/analyses/EPR_stageA_20261020")
assert json.loads((W/"RUN_EXIT.json").read_text())["exit"]==0
workers=[]
for p in Path("/proc").iterdir():
    if not p.name.isdigit():continue
    try:
        cmd=(p/"cmdline").read_bytes().split(b"\0")
        if any(x.endswith(b"/run_diagnostic.py") or x.endswith(b"/run_all.py") for x in cmd):
            workers.append(int(p.name))
    except OSError:pass
assert not workers,workers
files={}
for s in ("GOAL42","OBJECT41"):
    assert (W/(s+".exit")).read_text().strip()=="0"
    f=json.loads((A/s/"FINAL_AUDIT.json").read_text())
    assert f["status"]=="PASS" and f["optimizer_calls"]==f["environment_steps"]==f["policy_action_samples"]==0
    for n in ("pairs.jsonl","FINAL_AUDIT.json","INPUT_MANIFEST.json","NATIVE_SELECTION.json","DIAGNOSTIC_CONFIG.json","RUN_STATUS.json"):
        p=A/s/n;raw=p.read_bytes();files[str(p.relative_to(A))]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    for n in (s+".log",s+".exit",s+"_COMMAND.json"):
        p=W/n;raw=p.read_bytes();files["execution/"+n]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
for n in ("RUN_STARTED.json","RUN_EXIT.json"):
    p=W/n;raw=p.read_bytes();files["execution/"+n]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
p=A/"GRIPPER_DOMAIN_AUDIT.json";raw=p.read_bytes();files[p.name]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
result=dict(status="PASS",time=time.time(),workers=workers,files=files,
            gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.used,utilization.gpu","--format=csv,noheader"],text=True).strip(),
            scripts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in W.glob("*.py")},
            protocol_sha256=hashlib.sha256((W/"PROTOCOL.md").read_bytes()).hexdigest())
(A/"FINAL_PROCESS_AND_PACKAGE.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items() if k!="files"}))
