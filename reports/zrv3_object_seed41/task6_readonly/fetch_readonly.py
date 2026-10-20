"""Execute reviewed read-only audit in memory; retrieve metadata only."""
import getpass, json, hashlib, datetime
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
c=paramiko.SSHClient()
c.load_system_host_keys()
c.set_missing_host_key_policy(paramiko.RejectPolicy())
c.connect("jq1.9gpu.com",port=11920,username="root",
          password=getpass.getpass("SSH password (not saved): "),
          look_for_keys=False,allow_agent=False,timeout=25)
source=(R/"remote_audit.py").read_bytes()
cmd="/data/mlr_ifa/env/bin/python -"
try:
    stdin,stdout,stderr=c.exec_command(cmd,timeout=240)
    stdin.write(source.decode());stdin.flush();stdin.channel.shutdown_write()
    out=stdout.read();err=stderr.read().decode();rc=stdout.channel.recv_exit_status()
    (R/"REMOTE_COMMAND.json").write_text(json.dumps(dict(command=cmd,source_sha256=hashlib.sha256(source).hexdigest(),
        exit_code=rc,stderr=err,time=datetime.datetime.now().astimezone().isoformat()),indent=2)+"\n")
    assert rc==0,err
    result=json.loads(out);assert result["status"]=="PASS_READONLY"
    (R/"REMOTE_AUDIT.json").write_text(json.dumps(result,indent=2)+"\n")
    print("PASS_READONLY",len(result["input_hashes"]),"files; no CUDA/models/environment")
finally:c.close()
