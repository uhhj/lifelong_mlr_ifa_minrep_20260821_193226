"""One SSH progress snapshot and lightweight admission evidence; never retries work."""
from pathlib import Path
import json,getpass,datetime,paramiko
R=Path(__file__).resolve().parent;Q='/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed41_three_arm_20261019'
BO='/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed41_20261019';RO='/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed41_20261019'
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
c.connect('jq1.9gpu.com',port=11920,username='root',password=getpass.getpass('SSH password: '),look_for_keys=False,allow_agent=False,timeout=25)
D=R/('progress_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));D.mkdir()
try:
 cmd='/data/mlr_ifa/env/bin/python '+Q+'/monitor_snapshot.py';_,a,b=c.exec_command(cmd,timeout=90);raw=a.read();err=b.read().decode();rc=a.channel.recv_exit_status()
 (D/'COMMAND.json').write_text(json.dumps(dict(command=cmd,exit_code=rc,stderr=err),indent=2)+'\n');assert rc==0,err
 data=json.loads(raw);(D/'SNAPSHOT.json').write_bytes(raw);(R/'CURRENT_SNAPSHOT.json').write_bytes(raw)
 f=c.open_sftp()
 groups={Q:['commands.jsonl','queue.exit','wrapper.exit','QUEUE_FAILURE.json','AGGREGATE.json','FINAL_QUEUE_ACCEPTANCE.json','baseline_software.exit','baseline_software.log','baseline_configure.exit','baseline_configure.log','baseline_suite_audit.exit','baseline_suite_audit.log','baseline_preflight.exit','baseline_preflight.log','baseline_supervise.exit','ra_admission.exit','ra_supervise.exit'],BO:['RUN_STATUS.json','CONTRACT.json','MODAL_CONTRACT.json','PRETRAIN_METADATA.json','GPU_PREFLIGHT.json','SUITE_RESOURCE_AUDIT.json','SUMMARY.json','INDEPENDENT_ACCEPTANCE.json','pipeline.exit','supervisor.exit','resolved_export.json','resolved_random.json','resolved_coreset.json','resolved_eval.json'],RO:['RUN_STATUS.json','CONTRACT.json','GPU_PREFLIGHT.json','OFFLINE_ACCEPTANCE.json','AUDIT.json','PROCESS_EXIT_ACCEPTANCE.json','pipeline.exit','supervisor.exit']}
 for base,names in groups.items():
  dest=D/('queue' if base==Q else 'baseline' if base==BO else 'ra');dest.mkdir()
  for n in names:
   try:
    if f.stat(base+'/'+n).st_size>5*1024**2:continue
    (dest/n).write_bytes(f.open(base+'/'+n).read())
   except FileNotFoundError:pass
   except OSError as e:
    if e.errno==2:continue
    raise
 print(json.dumps(dict(directory=str(D),snapshot=data),indent=2))
finally:c.close()
