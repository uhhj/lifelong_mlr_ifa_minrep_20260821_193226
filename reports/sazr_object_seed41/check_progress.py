"""Read-only monitoring and allowlisted metadata download."""
import json,getpass,datetime,hashlib
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
W="/data/mlr_ifa/experiment_workspaces/SAZR_OBJECT_RA_seed41_task6_20261020"
O="/data/mlr_ifa/experiments/SAZR_OBJECT_RA_seed41_task6_20261020"
D=R/("progress_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S"));D.mkdir()
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
c.connect("jq1.9gpu.com",port=11920,username="root",password=getpass.getpass("SSH password (not saved): "),look_for_keys=False,allow_agent=False,timeout=25)
script=r'''
import json,os,time,hashlib,subprocess,shutil
from pathlib import Path
W=Path("/data/mlr_ifa/experiment_workspaces/SAZR_OBJECT_RA_seed41_task6_20261020")
O=Path("/data/mlr_ifa/experiments/SAZR_OBJECT_RA_seed41_task6_20261020")
files={}
for p in O.rglob("*"):
 if p.is_file() and p.suffix in (".json",".jsonl",".log",".exit",".started") and p.stat().st_size<6*1024**2:
  if p.name=="batches.jsonl":continue
  files[p.relative_to(O).as_posix()]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
workers=[]
for p in Path("/proc").iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  cmd=(p/"cmdline").read_bytes().replace(b"\0",b" ").decode()
  if "sazr_exp" in cmd or str(W) in cmd:workers.append(dict(pid=int(p.name),command=cmd))
 except (OSError,UnicodeError):pass
hashes=True
if (O/"CONTRACT.json").exists():
 for f,h in json.loads((O/"CONTRACT.json").read_text())["files"].items():
  if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h:hashes=False
print(json.dumps(dict(time=time.time(),files=files,workers=workers,hashes_pass=hashes,
 gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.used,utilization.gpu","--format=csv,noheader"]).decode(),
 free_GiB=shutil.disk_usage("/data").free/1024**3)))
'''
try:
    i,o,e=c.exec_command("/data/mlr_ifa/env/bin/python -",timeout=180);i.write(script);i.flush();i.channel.shutdown_write()
    raw=o.read();err=e.read().decode();rc=o.channel.recv_exit_status();assert rc==0,err
    snap=json.loads(raw);assert snap["hashes_pass"]
    (D/"SNAPSHOT.json").write_text(json.dumps(snap,indent=2)+"\n")
    manifest={}
    with c.open_sftp() as s:
        for n,meta in snap["files"].items():
            p=D/"evidence"/n;p.parent.mkdir(parents=True,exist_ok=True)
            with s.open(O+"/"+n,"rb") as f:b=f.read()
            # Active logs may grow; record the actual read hash and preserve snapshot hash separately.
            if hashlib.sha256(b).hexdigest()!=meta["sha256"]:
                assert n.endswith((".log",".jsonl")) or n in ("RUN_STATUS.json","epochs.json")
            p.write_bytes(b);manifest[n]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
    (D/"DOWNLOAD_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
    counts={}
    for name in ("anchor","validation_parent","validation_candidate","formal"):
        f=D/"evidence"/name/"episodes.jsonl"
        counts[name]=len(f.read_text().splitlines()) if f.exists() else 0
    final=(D/"evidence/PROCESS_EXIT_ACCEPTANCE.json").exists()
    blocked=(D/"evidence/BLOCKED.json").exists()
    st=dict(status="AWAITING_LOCAL_FINAL_AUDIT" if final else "BLOCKED" if blocked else "PARTIAL",
        time=datetime.datetime.now().astimezone().isoformat(),evidence=D.name,episodes=counts)
    (R/"RUN_STATUS.json").write_text(json.dumps(st,indent=2)+"\n")
    print(json.dumps(st,indent=2));print(snap["gpu"]);print("WORKERS",json.dumps(snap["workers"]))
finally:c.close()
