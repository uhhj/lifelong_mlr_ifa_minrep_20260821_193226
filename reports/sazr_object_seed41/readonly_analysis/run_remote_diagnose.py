"""Execute read-only inference through stdin; download JSON summaries only."""
from pathlib import Path
import getpass, json, hashlib, datetime, sys
import paramiko

R=Path(__file__).resolve().parent
script=R/(sys.argv[1] if len(sys.argv)>1 else "offline_diagnose.py")
D=R/(sys.argv[2] if len(sys.argv)>2 else ".")
D.mkdir(exist_ok=True)
c=paramiko.SSHClient()
c.load_system_host_keys()
c.set_missing_host_key_policy(paramiko.RejectPolicy())
c.connect("jq1.9gpu.com",port=11920,username="root",password=getpass.getpass("SSH password (not saved): "),
          look_for_keys=False,allow_agent=False,timeout=25)
cmd="bash -lc 'source /data/mlr_ifa/experiment_workspaces/SAZR_OBJECT_RA_seed41_task6_20261020/env.sh && PYTHONDONTWRITEBYTECODE=1 python -'"
try:
    _,o,e=c.exec_command("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader",timeout=30)
    busy=o.read().decode().strip();assert not busy,busy
    i,o,e=c.exec_command(cmd,timeout=300)
    i.write(script.read_text());i.flush();i.channel.shutdown_write()
    stdout=o.read().decode();stderr=e.read().decode();rc=o.channel.recv_exit_status()
    (D/"remote_stdout.log").write_text(stdout,encoding="utf-8")
    (D/"remote_stderr.log").write_text(stderr,encoding="utf-8")
    (D/"EXECUTION.json").write_text(json.dumps(dict(command=cmd,exit=rc,script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        time=datetime.datetime.now().astimezone().isoformat()),indent=2)+"\n")
    assert rc==0,stderr
    result=json.loads(next(x.split("=",1)[1] for x in stdout.splitlines() if x.startswith("DIAGNOSTIC_JSON=")))
    (D/"OFFLINE_DIAGNOSTIC.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("status","optimizer_calls","environment_calls","seconds")},indent=2))
finally:c.close()
