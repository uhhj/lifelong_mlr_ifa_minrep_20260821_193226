"""Download explicit lightweight metadata only; never checkpoints, noise or raw traces."""
import getpass,paramiko,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
W="/data/mlr_ifa/experiment_workspaces/OVR_OBJECT_RA_seed42_screen_20261020"
O="/data/mlr_ifa/experiments/OVR_OBJECT_RA_seed42_screen_20261020"
A="/data/mlr_ifa/analyses/OVR_OBJECT_seed42_20261020_admission_v3/RESOURCE_AUDIT.json"
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
c.connect("jq1.9gpu.com",port=11920,username="root",password=getpass.getpass("SSH password (not saved): "),look_for_keys=False,allow_agent=False,timeout=25)
try:
    with c.open_sftp() as s:
        files={A:"RESOURCE_AUDIT.json"}
        for n in ("CONTRACT.json","PREFLIGHT.json","PREFLIGHT_REPAIR.json","RUN_STATUS.json","SCREEN_AUDIT.json","PROCESS_EXIT_ACCEPTANCE.json"):
            files[O+"/"+n]=n
        for n in ("freeze","preflight","preflight_repair","parent9","teacher7","audit"):
            for ext in ("log","exit"):files[W+"/"+n+"."+ext]="commands/"+n+"."+ext
        for n in ("pipeline.started","pipeline.pid","pipeline.exit","commands.jsonl","final_process_audit.exit"):files[W+"/"+n]="commands/"+n
        for phase in ("parent9","teacher7"):
            for n in ("episodes.jsonl","attempts.jsonl","step_hashes.jsonl","resolved_actual.json","RNG_METADATA.json","COMPLETE.json"):
                files[O+"/"+phase+"/"+n]=phase+"/"+n
        receipt={}
        for remote,local in files.items():
            info=s.stat(remote);assert info.st_size<8*1024**2
            with s.open(remote,"rb") as f:raw=f.read()
            raw.decode("utf-8")
            p=R/"evidence"/local;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
            receipt[local]=dict(source=remote,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        (R/"DOWNLOAD_MANIFEST.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("DOWNLOADED",len(files),"light files")
finally:c.close()
