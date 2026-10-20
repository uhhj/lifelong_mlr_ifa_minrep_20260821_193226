"""Bounded read-only SSH check; credentials exist only in getpass memory."""
import datetime
import getpass
import hashlib
import json
from pathlib import Path
import paramiko
import argparse
import shlex

R=Path(__file__).resolve().parent
Q="/data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020"
COMMAND=f"/data/mlr_ifa/env/bin/python {Q}/monitor_snapshot_v2.py"
W="/data/mlr_ifa/experiment_workspaces/RCTR_OBJECT_seed42_full_20261021"
DETAIL_CODE=r'''
import collections,json,time
from pathlib import Path
from coreset_exp.common import O,verify,sha
c=verify()
root=O/"rctr/task6"
rows=[json.loads(p.read_text()) for p in sorted((root/"committed").glob("*.json"))]
def read(p):
    return json.loads(p.read_text()) if p.exists() else None
result=dict(unix=time.time(),status="FROZEN_CONTRACT_INPUTS_PASS",
    contract_sha256=sha(O/"CONTRACT.json"),source_files=len(c["source_sha256"]),
    committed=rows,collection_status=read(root/"RUN_STATUS.json"),
    live_step=read(root/"LIVE_STEP.json"),commands=read(O/"pipeline.pid.json"),
    logs={p.name:dict(bytes=p.stat().st_size,mtime=p.stat().st_mtime,
         tail=p.read_text(errors="replace")[-2000:])
         for p in (O/"collect6.log",O/"train6.log") if p.exists()},
    checkpoint_files=[dict(name=p.name,bytes=p.stat().st_size,mtime=p.stat().st_mtime)
                      for p in (O/"ra_coreset/models").glob("*.pth")])
print(json.dumps(result,indent=2))
'''

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--details",type=Path)
    args=ap.parse_args()
    stamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out=args.details or R/("progress_"+stamp)
    if args.details:
        assert out.resolve().parent==R and (out/"SNAPSHOT.json").is_file()
    client=paramiko.SSHClient()
    client.load_system_host_keys()
    # The destination key is already pinned by the user's previous native SSH connections.
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    password=getpass.getpass("SSH password (memory only): ")
    try:
        client.connect("jq1.9gpu.com",port=11920,username="root",password=password,
                       allow_agent=False,look_for_keys=False,timeout=20)
        password=None
        if args.details:
            command=f"source {W}/env.sh && /data/mlr_ifa/env/bin/python -c {shlex.quote(DETAIL_CODE)}"
            _,stdout,stderr=client.exec_command(command,timeout=60)
            raw=stdout.read();err=stderr.read();rc=stdout.channel.recv_exit_status()
            assert rc==0,(rc,err.decode(errors="replace"))
            details=json.loads(raw)
            assert details["status"]=="FROZEN_CONTRACT_INPUTS_PASS"
            (out/"DETAILS.json").write_bytes(raw)
            (out/"DETAILS_RECEIPT.json").write_text(json.dumps(dict(command=command,exit=rc,
                sha256=hashlib.sha256(raw).hexdigest(),server_writes=0),indent=2)+"\n")
            print(json.dumps(dict(status=details["status"],committed=len(details["committed"]),
                                  directory=str(out))))
            return
        _,stdout,stderr=client.exec_command(COMMAND,timeout=60)
        raw=stdout.read();err=stderr.read();rc=stdout.channel.recv_exit_status()
        assert rc==0,(rc,err.decode(errors="replace"))
        snapshot=json.loads(raw)
        out.mkdir()
        (out/"SNAPSHOT.json").write_bytes(raw)
        with client.open_sftp() as sftp:
            for name in ("REGISTRY.json","controller_v2.pid.json","CONTROLLER_V2_HANDOFF.json",
                         "LIBERO50_REGISTRY_ACCEPTANCE.json"):
                with sftp.open(Q+"/"+name,"rb") as f:data=f.read()
                json.loads(data)
                (out/name).write_bytes(data)
        receipt=dict(status="READ_ONLY_SNAPSHOT",command=COMMAND,exit=rc,
                     snapshot_sha256=hashlib.sha256(raw).hexdigest(),
                     unix=snapshot["unix"],writes_on_server=0,optimizer_calls=0,environment_steps=0)
        (out/"CHECK_RECEIPT.json").write_text(json.dumps(receipt,indent=2)+"\n")
        print(json.dumps(dict(directory=str(out),snapshot=snapshot),ensure_ascii=False,indent=2))
    finally:
        password=None
        client.close()

if __name__=="__main__":main()
