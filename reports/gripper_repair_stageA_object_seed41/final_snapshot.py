"""Read-only remote exit/hash observation and metadata download."""
import getpass,json,hashlib,datetime
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
W="/data/mlr_ifa/experiment_workspaces/GripperRepair_A_OBJECT_seed41_20261020"
O="/data/mlr_ifa/experiments/GripperRepair_A_OBJECT_seed41_20261020"
A="/data/mlr_ifa/analyses/GripperRepair_A_OBJECT_seed41_20261020"
script=r'''
import os,json,hashlib,subprocess,time
from pathlib import Path
W=Path("/data/mlr_ifa/experiment_workspaces/GripperRepair_A_OBJECT_seed41_20261020")
O=Path("/data/mlr_ifa/experiments/GripperRepair_A_OBJECT_seed41_20261020")
A=Path("/data/mlr_ifa/analyses/GripperRepair_A_OBJECT_seed41_20261020")
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
contract=json.loads((O/"CONTRACT.json").read_text())
assert all(sha(f)==h for f,h in contract["files"].items())
workers=[]
for p in Path("/proc").iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  cmd=(p/"cmdline").read_bytes().replace(b"\0",b" ").decode()
  if "run_stage_a.py" in cmd or "native_error_audit.py" in cmd:workers.append(dict(pid=int(p.name),command=cmd))
 except (OSError,UnicodeError):pass
fm=json.loads(Path("/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed41_20261019/FEATURES.json").read_text())
source=Path("/data/mlr_ifa/experiment_workspaces/RA_OBJECT_conditional_20261019/seed41/source/libero/lifelong/models/modules/data_augmentation.py")
txt=source.read_text();start=txt.index("class DataAugGroup");end=txt.index("class RandomErasingAug",start)
print(json.dumps(dict(time=time.time(),workers=workers,source_input_hash_pass=True,
 run_exit=int((W/"run.exit").read_text()),native_audit_exit=int((A/"native.exit").read_text()),
 gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.used,utilization.gpu","--format=csv,noheader"]).decode(),
 feature_export_mode=fm["export_mode"],augmentation_source=str(source),augmentation_sha256=sha(source),
 augmentation_mode_source_excerpt=txt[start:end],
 files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in O.glob("*.json")})))
'''
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
c.connect("jq1.9gpu.com",port=11920,username="root",password=getpass.getpass("SSH password (not saved): "),
          look_for_keys=False,allow_agent=False,timeout=25)
try:
    i,o,e=c.exec_command("/data/mlr_ifa/env/bin/python -",timeout=120)
    i.write(script);i.flush();i.channel.shutdown_write()
    result=o.read().decode();err=e.read().decode();rc=o.channel.recv_exit_status();assert rc==0,err
    snap=json.loads(result)
    assert not snap["workers"] and snap["run_exit"]==snap["native_audit_exit"]==0
    (R/"FINAL_OBSERVATION.json").write_text(json.dumps(snap,indent=2)+"\n")
    for n,m in snap["files"].items():
        raw=(R/n).read_bytes();assert hashlib.sha256(raw).hexdigest()==m["sha256"]
    with c.open_sftp() as s:s.get(A+"/native.exit",str(R/"native.exit"))
    (R/"SNAPSHOT_EXIT.json").write_text(json.dumps(dict(command="/data/mlr_ifa/env/bin/python -",exit=rc,
         script_sha256=hashlib.sha256(script.encode()).hexdigest(),time=datetime.datetime.now().astimezone().isoformat()),indent=2)+"\n")
    print("PASS: no workers, both exits zero, all frozen files and downloaded JSON hashes match.")
finally:c.close()
