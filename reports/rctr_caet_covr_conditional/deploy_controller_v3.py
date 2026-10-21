"""Fixed-file, pinned-host scheduler handoff; never uploads experiment payloads."""
import getpass
import hashlib
import json
from pathlib import Path
import paramiko

R=Path(__file__).resolve().parent
Q='/data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020'
FILES={name:R/name for name in ('branch_gate_v3.py','queue_controller_v3.py','handoff_controller_v3.py',
    'monitor_snapshot_v3.py','CAET_GOAL_DEFERRED.json')}
FILES.update(CAET_GOAL_FEASIBILITY_EVIDENCE=R/'goal_preparation/GOAL_EFFECT_FEASIBILITY.json')
FILES['CAET_GOAL_FEASIBILITY_EVIDENCE.json']=FILES.pop('CAET_GOAL_FEASIBILITY_EVIDENCE')
FILES['CAET_OBJECT_AGGREGATE.json']=R/'caet_object_three_seed_final_20261021/SUMMARY.json'
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
password=getpass.getpass('SSH password (memory only): ')
try:
    c.connect('jq1.9gpu.com',port=11920,username='root',password=password,
        allow_agent=False,look_for_keys=False,timeout=20)
    password=None
    manifest={}
    with c.open_sftp() as sftp:
        # All files are new versioned orchestration or lightweight evidence.
        for name,p in FILES.items():
            raw=p.read_bytes()
            assert len(raw)<2*1024**2
            with sftp.open(Q+'/'+name,'wx') as f:f.write(raw)
            with sftp.open(Q+'/'+name,'rb') as f:assert f.read()==raw
            manifest[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    (R/'CONTROLLER_V3_DEPLOY_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    command='/data/mlr_ifa/env/bin/python '+Q+'/handoff_controller_v3.py'
    _,out,err=c.exec_command(command,timeout=60)
    raw=out.read();error=err.read();rc=out.channel.recv_exit_status()
    (R/'CONTROLLER_V3_DEPLOY_RESULT.json').write_text(json.dumps(dict(exit=rc,stdout=raw.decode(),stderr=error.decode(),
        command=command,scientific_source_changes=0),indent=2)+'\n')
    assert rc==0,error.decode()
    with c.open_sftp() as sftp:
        for name in ('CONTROLLER_V3_HANDOFF_STARTED.json','CONTROLLER_V3_HANDOFF.json','controller_v3.pid.json'):
            with sftp.open(Q+'/'+name,'rb') as f:(R/name).write_bytes(f.read())
    print(raw.decode())
finally:
    password=None;c.close()
