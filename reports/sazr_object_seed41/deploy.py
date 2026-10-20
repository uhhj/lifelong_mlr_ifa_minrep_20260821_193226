"""Deploy only reviewed new prototype code; never modifies existing experiments."""
from pathlib import Path
import paramiko,getpass,hashlib,json,sys,shlex
R=Path(__file__).resolve().parent
W="/data/mlr_ifa/experiment_workspaces/SAZR_OBJECT_RA_seed41_task6_20261020"
O="/data/mlr_ifa/experiments/SAZR_OBJECT_RA_seed41_task6_20261020"
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
c.connect("jq1.9gpu.com",port=11920,username="root",password=getpass.getpass("SSH password (not saved): "),look_for_keys=False,allow_agent=False,timeout=25)
def run(cmd):
    _,a,b=c.exec_command(cmd,timeout=30);out=a.read().decode();err=b.read().decode();rc=a.channel.recv_exit_status();assert rc==0,(cmd,err);return out
try:
    sync="--sync" in sys.argv
    if sync:run("test -d "+W+" && test ! -f "+O+"/CONTRACT.json")
    else:run("test ! -e "+W+" && test ! -e "+O+" && mkdir -p "+W+" "+O)
    files=list((R/"source").rglob("*.py"))+[R/"PROTOCOL.md",R/"USER_PROTOCOL.txt",R/"pipeline.py",R/"supervisor.py",R/"freeze.py",R/"audit.py"]
    with c.open_sftp() as s:
        hashes={}
        for p in files:
            assert p.exists(),p
            rel=p.relative_to(R).as_posix();remote=W+"/"+rel
            run("mkdir -p "+shlex.quote(str(Path(remote).parent).replace("\\","/")))
            raw=p.read_bytes();assert len(raw)<1000000
            with s.open(remote,"wb") as f:f.write(raw)
            hashes[rel]=hashlib.sha256(raw).hexdigest()
        env='#!/usr/bin/env bash\nsource /data/mlr_ifa/experiment_workspaces/RA_OBJECT_conditional_20261019/seed41/env.sh\nexport PYTHONPATH="'+W+'/source:$PYTHONPATH"\ncd '+W+'\n'
        with s.open(W+"/env.sh","w") as f:f.write(env)
    (R/"DEPLOYMENT.json").write_text(json.dumps(dict(workspace=W,output=O,files=hashes),indent=2)+"\n")
    print("DEPLOYED",len(files))
finally:c.close()
