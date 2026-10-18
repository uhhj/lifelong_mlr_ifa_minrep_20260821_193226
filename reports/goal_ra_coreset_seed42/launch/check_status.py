"""Read-only status/hash snapshot, without restarting any process."""
from pathlib import Path
import getpass,json
import paramiko
R=Path(__file__).resolve().parent;W='/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed42_20261018';O='/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed42_20261018'
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());c.connect('jq1.9gpu.com',port=11920,username='root',password=getpass.getpass('SSH password: '),look_for_keys=False,allow_agent=False,timeout=25)
def run(cmd):
    _,a,b=c.exec_command(cmd,timeout=300);out=a.read().decode();err=b.read().decode();assert a.channel.recv_exit_status()==0,err;return out
try:
    run('source '+W+'/env.sh && /data/mlr_ifa/env/bin/python -c "from coreset_exp.common import verify; verify()"')
    snap=json.loads(run('source '+W+'/env.sh && /data/mlr_ifa/env/bin/python '+W+'/monitor_snapshot.py'));snap['frozen_hash_check']='PASS'
    (R/'current_snapshot.json').write_text(json.dumps(snap,indent=2)+'\n',encoding='utf-8')
    f=c.open_sftp();E=R/'progress_evidence';E.mkdir(exist_ok=True)
    for name in ['ra_coreset/epochs.json','ra_coreset/INITIAL_STATE.json','TASK6_TENSOR_PARITY.json','RUN_STATUS.json']:
        try:raw=f.open(O+'/'+name).read()
        except FileNotFoundError:continue
        dest=E/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    print(json.dumps(snap,indent=2))
finally:c.close()
