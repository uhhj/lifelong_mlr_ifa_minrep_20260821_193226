"""Read-only snapshot/optional safe metadata collection; never restarts a run."""
from pathlib import Path,PurePosixPath
import getpass,json,datetime,hashlib,argparse
import paramiko
R=Path(__file__).resolve().parent;Q='/data/mlr_ifa/experiment_workspaces/RA_OBJECT_conditional_20261019'
ap=argparse.ArgumentParser();ap.add_argument('--package',action='store_true');args=ap.parse_args()
D=R/('progress_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));D.mkdir()
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
c.connect('jq1.9gpu.com',port=11920,username='root',password=getpass.getpass('SSH password: '),look_for_keys=False,allow_agent=False,timeout=25)
commands=[]
def run(cmd):
 _,a,b=c.exec_command(cmd,timeout=600);out=a.read().decode();err=b.read().decode();rc=a.channel.recv_exit_status();commands.append(dict(command=cmd,exit_code=rc,stderr=err));assert rc==0,err
 return out
try:
 snap=json.loads(run('/data/mlr_ifa/env/bin/python '+Q+'/monitor_snapshot.py'));f=c.open_sftp()
 def fetch(src,dst):
  try:raw=f.open(src).read()
  except FileNotFoundError:return
  dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(raw)
 for n in ['queue.started','queue.pid','queue.exit','wrapper.exit','commands.jsonl','queue.log','QUEUE_FAILURE.json','STAGE1_RESULTS.json','FINAL_QUEUE_ACCEPTANCE.json','aggregate.exit','aggregate.log']:
  fetch(Q+'/'+n,D/'queue'/n)
 for seed in (41,42,43):
  W=Q+'/seed'+str(seed);O='/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed%d_20261019'%seed
  for n in ['admission_commands.log','software.log','software.exit','prepare.log','prepare.exit','offline.log','offline.exit','freeze.log','freeze.exit','preflight.log','preflight.exit']:
   fetch(W+'/'+n,D/('seed%d'%seed)/'admission'/n)
  for n in ['RUN_STATUS.json','GPU_PREFLIGHT.json','OFFLINE_ACCEPTANCE.json','TASK6_TENSOR_PARITY.json','ra_coreset/epochs.json','ra_coreset/INITIAL_STATE.json','pipeline.exit','supervisor.exit','PIPELINE_FAILURE.json']:
   fetch(O+'/'+n,D/('seed%d'%seed)/n)
  if args.package and snap['seeds'][str(seed)]['gpu_preflight']:
   prefix='source '+W+'/env.sh && /data/mlr_ifa/env/bin/python '
   run(prefix+'-c "from coreset_exp.common import verify; verify()"');snap['seeds'][str(seed)]['frozen_hash_check']='PASS'
   remote=run(prefix+W+'/package_light.py').strip();assert remote.startswith(W+'/light_') and '\n' not in remote
   raw=f.open(remote+'/PACKAGE_MANIFEST.json').read();pkg=json.loads(raw);E=D/('seed%d'%seed)/'evidence'
   for n,m in pkg['files'].items():
    p=PurePosixPath(n);assert not p.is_absolute() and '..' not in p.parts
    assert p.suffix in {'.json','.jsonl','.log','.exit','.pid','.started'} and not {'features','buffer','models','recovery','smoke','bindings'}&set(p.parts)
    rawfile=f.open(remote+'/'+n).read();assert len(rawfile)==m['bytes'] and hashlib.sha256(rawfile).hexdigest()==m['sha256']
    dest=E/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(rawfile)
   (E/'PACKAGE_MANIFEST.json').write_bytes(raw)
 (D/'SNAPSHOT.json').write_text(json.dumps(snap,indent=2)+'\n');(R/'CURRENT_SNAPSHOT.json').write_text(json.dumps(snap,indent=2)+'\n')
 (D/'COMMANDS.json').write_text(json.dumps(commands,indent=2)+'\n')
 print(json.dumps(snap,indent=2));print('LOCAL_EVIDENCE',D)
finally:c.close()
