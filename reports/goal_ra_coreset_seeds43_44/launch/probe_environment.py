"""Read-only admission probe; credentials only in getpass memory."""
from pathlib import Path
import getpass,json,datetime
import paramiko
R=Path(__file__).resolve().parent
code=r'''
from pathlib import Path
import json,shutil,subprocess,time
out={'time':time.time(),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv'],text=True),'processes':subprocess.check_output(['ps','-eo','pid,etime,args'],text=True),'disk':{p:shutil.disk_usage(p)._asdict() for p in ['/root','/data']},'parents':{}}
for s in [43,44]:
 p=Path('/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed%d_20261017'%s)
 out['parents'][s]={n:json.loads((p/n).read_text()) for n in ['INDEPENDENT_ACCEPTANCE.json','CONTRACT.json','FEATURES.json','resolved_coreset.json']}
 out['parents'][s]['checkpoint_paths']=[str(x) for x in sorted((p/'coreset/models').glob('*ep99.pth'))]
 out['parents'][s]['new_output_exists']=Path('/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed%d_20261019'%s).exists()
out['queue_exists']=Path('/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_extension_20261019').exists()
print(json.dumps(out))
'''
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
c.connect('jq1.9gpu.com',port=11920,username='root',password=getpass.getpass('SSH password: '),look_for_keys=False,allow_agent=False,timeout=25)
try:
 i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -',timeout=120);i.write(code);i.channel.shutdown_write();raw=o.read();err=e.read().decode();rc=o.channel.recv_exit_status();assert rc==0,err
 d=json.loads(raw);(R/'ENVIRONMENT_ADMISSION.json').write_bytes(raw)
 print(json.dumps(dict(time=d['time'],gpu=d['gpu'],disk=d['disk'],queue_exists=d['queue_exists'],parents={s:dict(acceptance=v['INDEPENDENT_ACCEPTANCE.json']['status'],seed=v['resolved_coreset.json']['seed'],checkpoints=len(v['checkpoint_paths']),new_output_exists=v['new_output_exists']) for s,v in d['parents'].items()},processes=[x for x in d['processes'].splitlines() if any(t in x for t in ['torchrun','coreset_exp.train','coreset_exp.evaluate','pipeline.py'])]),indent=2))
finally:c.close()
