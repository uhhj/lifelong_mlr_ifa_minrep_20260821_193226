"""Read-only remote snapshot and reviewed light evidence retrieval; credentials in memory."""
import getpass,json,time,hashlib,shlex,datetime
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
O="/data/mlr_ifa/experiments/RCTR_GOAL_seed42_stageA2_20261020"
REC=O+"/engineering_recovery/dedicated_env_v1"
REMOTE=r'''
import json,subprocess,time,hashlib
from pathlib import Path
O=Path("/data/mlr_ifa/experiments/RCTR_GOAL_seed42_stageA2_20261020")
R=O/"engineering_recovery/dedicated_env_v1"
def read(p):
 if not p.exists():return None
 try:return json.loads(p.read_text())
 except ValueError:return p.read_text()
rows=[json.loads(x) for x in (R/"episodes.jsonl").read_text().splitlines()] if (R/"episodes.jsonl").exists() else []
processes=subprocess.check_output(["ps","-eo","pid,ppid,etimes,args"],text=True).splitlines()
workers=[x for x in processes if x.strip().endswith(("stageA2_20261020/runner.py","stageA2_20261020/recovery_runner.py","stageA2_20261020/run_recovery.sh","stageA2_20261020/run_a2.sh"))]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
input_checks={}
for label,base in (("original",O),("recovery",R)):
 if (base/"CONTRACT.json").exists():
  files=read(base/"CONTRACT.json")["files"]
  changed=[p for p,h in files.items() if sha(p)!=h]
  input_checks[label]=dict(status="PASS" if not changed else "FAIL",count=len(files),changed=changed)
report=dict(unix=time.time(),original_exit=read(O/"runner.exit"),original_status=read(O/"RUN_STATUS.json"),
 status=read(R/"RUN_STATUS.json"),recovery_exit=read(R/"runner.exit"),supervisor_exit=read(R/"supervisor.exit"),
 committed=len(rows),gate=read(R/"ENGINEERING_GATE.json"),inventory=read(R/"SNAPSHOT_INVENTORY.json"),
 analysis=read(R/"ANALYSIS.json"),workers=workers,input_checks=input_checks,
 gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.used,utilization.gpu","--format=csv,noheader"],text=True),
 disk=subprocess.check_output(["df","-h","/data","/root"],text=True))
if rows:
 recent=rows[-min(30,len(rows)):]
 report["recent_seconds_per_branch"]=sum(x["elapsed_seconds"] for x in recent)/len(recent)
 report["eta_seconds"]=(333-len(rows))*report["recent_seconds_per_branch"]
names={"CONTRACT.json","ENGINEERING_GATE.json","SNAPSHOT_INVENTORY.json","MODEL_ADMISSION.json","FAILURE.json",
 "RUN_STATUS.json","episodes.jsonl","attempts.jsonl","runner.exit","supervisor.exit","analyze.exit","runner.log",
 "EXECUTION_COMPLETE.json","INPUT_PROTECTION.json","ANALYSIS.json","REPORT.md","candidate_summary.csv","paired_summary.csv","resolved_eval.json"}
files={}
for base in (O,R):
 if base.exists():
  for p in base.iterdir():
   if p.is_file() and p.name in names:
    raw=p.read_bytes();assert len(raw)<8*1024**2
    files[str(p.relative_to(O))]=dict(text=raw.decode(),sha256=hashlib.sha256(raw).hexdigest())
print(json.dumps(dict(report=report,files=files)))
'''
def main():
 c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
 c.connect("jq1.9gpu.com",port=11920,username="root",password=getpass.getpass("SSH password (memory only): "),
           look_for_keys=False,allow_agent=False,timeout=25)
 try:
  cmd="/data/mlr_ifa/env/bin/python -c "+shlex.quote(REMOTE)
  _,o,e=c.exec_command(cmd,timeout=90);raw=o.read();err=e.read().decode();rc=o.channel.recv_exit_status()
  assert rc==0,(rc,err);bundle=json.loads(raw)
 finally:c.close()
 dest=R/("progress_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S"));dest.mkdir()
 manifest={}
 for name,v in bundle["files"].items():
  p=dest/name;assert p.resolve().is_relative_to(dest.resolve())
  p.parent.mkdir(parents=True,exist_ok=True);data=v["text"].encode()
  assert hashlib.sha256(data).hexdigest()==v["sha256"];p.write_bytes(data);manifest[name]=v["sha256"]
 (dest/"MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
 (dest/"SNAPSHOT.json").write_text(json.dumps(bundle["report"],indent=2)+"\n")
 (R/"LATEST_PROGRESS.json").write_text(json.dumps(dict(directory=str(dest),**bundle["report"]),indent=2)+"\n")
 print(json.dumps(dict(directory=str(dest),committed=bundle["report"]["committed"],
  status=bundle["report"]["status"],eta_seconds=bundle["report"].get("eta_seconds"),
  workers=bundle["report"]["workers"]),indent=2))
if __name__=="__main__":main()
