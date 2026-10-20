"""Deploy once; no credential persistence or archive transfer."""
from pathlib import Path
import json,getpass,hashlib,datetime
import paramiko
R=Path(__file__).resolve().parent
W="/data/mlr_ifa/experiment_workspaces/GripperRepair_A_OBJECT_seed41_20261020"
O="/data/mlr_ifa/experiments/GripperRepair_A_OBJECT_seed41_20261020"
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
c.connect("jq1.9gpu.com",port=11920,username="root",password=getpass.getpass("SSH password (not saved): "),
          allow_agent=False,look_for_keys=False,timeout=25)
try:
    _,o,e=c.exec_command("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader",timeout=30)
    assert not o.read().decode().strip(),"GPU busy; do not start"
    s=c.open_sftp()
    for path in (W,O):
        try:s.stat(path)
        except FileNotFoundError:pass
        else:raise RuntimeError("Existing directory; inspect instead of duplicate launch: "+path)
    s.mkdir(W)
    files={}
    for name in ("gripper_core.py","run_stage_a.py","PROTOCOL.md","USER_PROTOCOL.txt"):
        raw=(R/name).read_bytes().replace(b"\r\n",b"\n")
        with s.open(W+"/"+name,"wb") as f:f.write(raw)
        files[name]=hashlib.sha256(raw).hexdigest()
    env="source /data/mlr_ifa/experiment_workspaces/SAZR_OBJECT_RA_seed41_task6_20261020/env.sh\nexport PYTHONDONTWRITEBYTECODE=1\ncd "+W+"\n"
    with s.open(W+"/env.sh","w") as f:f.write(env)
    s.close()
    cmd="bash -lc 'source "+W+"/env.sh; python run_stage_a.py > "+W+"/run.log 2>&1; rc=$?; printf \"%s\\n\" \"$rc\" > "+W+"/run.exit; exit \"$rc\"'"
    (R/"DEPLOYMENT.json").write_text(json.dumps(dict(workspace=W,output=O,files=files,command=cmd,
         time=datetime.datetime.now().astimezone().isoformat()),indent=2)+"\n")
    _,o,e=c.exec_command(cmd,timeout=300)
    stdout=o.read().decode();stderr=e.read().decode();rc=o.channel.recv_exit_status()
    s=c.open_sftp()
    for folder,dirs,files_remote in [(O,[],s.listdir(O))]:
        for n in files_remote:
            if n.endswith(".json"):
                s.get(O+"/"+n,str(R/n))
    s.get(W+"/run.log",str(R/"run.log"));s.get(W+"/run.exit",str(R/"run.exit"))
    s.close()
    (R/"COMMAND_EXIT.json").write_text(json.dumps(dict(exit=rc,stderr=stderr,stdout=stdout,
         time=datetime.datetime.now().astimezone().isoformat()),indent=2)+"\n")
    print("REMOTE_EXIT",rc)
    print((R/"run.log").read_text(encoding="utf-8")[-3500:])
finally:c.close()
