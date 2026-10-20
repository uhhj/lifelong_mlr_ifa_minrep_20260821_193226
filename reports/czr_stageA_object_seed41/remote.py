"""One-time native inference deployment, or read-only collection; no secrets persisted."""
import sys,getpass,json,hashlib,datetime,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
W="/data/mlr_ifa/experiment_workspaces/CZR_A_OBJECT_seed41_20261020"
O="/data/mlr_ifa/analyses/CZR_A_OBJECT_seed41_20261020"
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
c.connect("jq1.9gpu.com",port=11920,username="root",password=getpass.getpass("SSH password (not saved): "),
          allow_agent=False,look_for_keys=False,timeout=25)
def command(cmd):
    _,o,e=c.exec_command(cmd,timeout=1800)
    result=o.read().decode();err=e.read().decode();rc=o.channel.recv_exit_status()
    return result,err,rc
try:
    s=c.open_sftp()
    if sys.argv[1]=="launch":
        result,err,rc=command("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader")
        assert rc==0 and not result.strip(),(result,err)
        for p in (W,O):
            try:s.stat(p)
            except FileNotFoundError:pass
            else:raise RuntimeError("Refuse existing workspace "+p)
        s.mkdir(W);files={}
        for n in ("run_czr.py","czr_core.py","PROTOCOL.md","USER_PROTOCOL.txt"):
            raw=(R/n).read_bytes().replace(b"\r\n",b"\n")
            with s.open(W+"/"+n,"wb") as f:f.write(raw)
            files[n]=hashlib.sha256(raw).hexdigest()
        env="source /data/mlr_ifa/experiment_workspaces/SAZR_OBJECT_RA_seed41_task6_20261020/env.sh\nexport PYTHONDONTWRITEBYTECODE=1\ncd "+W+"\n"
        with s.open(W+"/env.sh","w") as f:f.write(env)
        cmd="bash -lc "+shlex.quote("source "+W+"/env.sh; python run_czr.py > "+W+"/run.log 2>&1; rc=$?; printf '%s\\n' \"$rc\" > "+W+"/run.exit; exit \"$rc\"")
        (R/"DEPLOYMENT.json").write_text(json.dumps(dict(command=cmd,files=files,time=datetime.datetime.now().astimezone().isoformat()),indent=2)+"\n")
        result,err,rc=command(cmd)
        (R/"COMMAND_EXIT.json").write_text(json.dumps(dict(command=cmd,exit=rc,stderr=err,stdout=result),indent=2)+"\n")
        print("REMOTE_EXIT",rc)
    # Fetch reviewed metadata; never feature or repair-bank tensors.
    manifest={}
    for n in s.listdir(O):
        if not n.endswith(".json"):continue
        with s.open(O+"/"+n,"rb") as f:raw=f.read()
        (R/n).write_bytes(raw);manifest[n]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    for n in ("run.log","run.exit"):s.get(W+"/"+n,str(R/n))
    script="""
import json,hashlib,os
from pathlib import Path
O=Path(%r);W=Path(%r)
c=json.loads((O/'CONTRACT.json').read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in c['files'].items())
workers=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=(p/'cmdline').read_bytes().split(b'\\0')
  if any(a==b'run_czr.py' for a in args):workers.append(int(p.name))
 except OSError:pass
print(json.dumps(dict(exit=int((W/'run.exit').read_text()),workers=workers,frozen_hash_pass=True)))
"""%(O,W)
    i,o,e=c.exec_command("/data/mlr_ifa/env/bin/python -",timeout=180)
    i.write(script);i.flush();i.channel.shutdown_write()
    snap=o.read().decode();err=e.read().decode();rc=o.channel.recv_exit_status()
    assert rc==0,err
    observation=json.loads(snap)
    observation.update(time=datetime.datetime.now().astimezone().isoformat(),downloaded=manifest)
    (R/"FINAL_OBSERVATION.json").write_text(json.dumps(observation,indent=2)+"\n")
    print(json.dumps({k:v for k,v in observation.items() if k!="downloaded"},indent=2))
    print((R/"run.log").read_text(encoding="utf-8")[-2000:])
finally:c.close()
