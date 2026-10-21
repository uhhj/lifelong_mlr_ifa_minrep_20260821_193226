import getpass,json,shlex,datetime
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
SCRIPT=r'''
import json,subprocess,time,shutil,hashlib,os
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/CAET_v2_OBJECT_extension_LIBERO50_20261021')
out={'observed_unix':time.time(),'files':{},'processes':[],'disk':{p:shutil.disk_usage(p).free for p in ('/','/data')}}
def read(p):
 if p.exists():
  try:out['files'][str(p)]=json.loads(p.read_text()) if p.suffix=='.json' else p.read_text()[-5000:]
  except json.JSONDecodeError:out['files'][str(p)]={'inflight_read':True}
for n in ('RUN_STATUS.json','controller.pid.json','controller.exit','FAILURE.json','controller.log','QUEUE.json','ORCHESTRATION_CONTRACT.json'):read(Q/n)
for p in Q.glob('*.receipt.json'):read(p)
D=Q/'capacity_recovery_v1'
for n in ('RUN_STATUS.json','CONTRACT.json','watcher.pid.json','watcher.exit','FAILURE.json','HANDOFF.json','watcher.log'):read(D/n)
for p in D.glob('*.receipt.json'):read(p)
recovery=Q/'libero50_taskmap01'
for n in ('RUN_STATUS.json','CONTRACT.json','MAPPING_AUDIT.json','controller.pid.json','controller.exit','FAILURE.json','controller.log','QUEUE.json'):read(recovery/n)
for p in recovery.glob('*.receipt.json'):read(p)
for w in (recovery/'shared_pretraining',recovery/'seed41',recovery/'seed42',recovery/'seed43'):
 for p in (w/'admission').glob('*'):read(p)
for e in [root/('LIBERO50_seed%d.json'%seed) for root in (Q,recovery) for seed in (41,42,43)]:
 if not e.is_file():continue
 entry=json.loads(e.read_text())
 for output in (Path(entry['output']),Path(entry['shared_pretrain_output'])):
  for n in ('RUN_STATUS.json','GPU_PREFLIGHT.json','PRETRAIN_GPU_PREFLIGHT.json','PRETRAIN_INITIALIZATION.json','LAUNCH_ACCEPTANCE.json','PROCESS_EXIT_ACCEPTANCE.json','pipeline.exit','supervisor.exit','PRETRAIN_COMPLETE.json','pretraining/epochs.json','pretrain.log','pretrain.exit'):
   read(output/n)

for seed in (41,43):
 w=Q/('object_seed%d'%seed);o=Path('/data/mlr_ifa/experiments/CAET_v2_OBJECT_seed%d_20261021'%seed)
 for n in ('supervisor.exit','FAILURE.json'):read(w/n)
 for p in (w/'admission').glob('*'):
  if p.suffix in ('.exit','.log','.json'):read(p)
 for n in ('RUN_STATUS.json','GPU_PREFLIGHT.json','LAUNCH_ACCEPTANCE.json','pipeline.exit','PROCESS_EXIT_ACCEPTANCE.json','ra_coreset/epochs.json','TASK6_TENSOR_PARITY.json'):read(o/n)
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if 'python -c' in cmd:continue
  if any(x in cmd for x in ('CAET_v2_OBJECT_extension_LIBERO50','caet_v2_exp.','coreset_exp.evaluate','l50_exp.','worker_entrypoint_libero50.py')):out['processes'].append(dict(pid=int(p.name),command=cmd))
 except OSError:pass
out['filesystem_capacity']={}
for mount in ('/','/data'):
 st=os.statvfs(mount);out['filesystem_capacity'][mount]=dict(block_size=st.f_frsize,available_unprivileged=st.f_bavail*st.f_frsize,free_all_blocks=st.f_bfree*st.f_frsize,reserved_difference=(st.f_bfree-st.f_bavail)*st.f_frsize,euid=os.geteuid())
out['gpu']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
print(json.dumps(out))
'''
dest=R/('queue_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));dest.mkdir()
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
try:
 c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
 i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(SCRIPT),timeout=120);raw=o.read();err=e.read();rc=o.channel.recv_exit_status();(dest/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err.decode()),indent=2),encoding='utf8');assert rc==0,err.decode();(dest/'SNAPSHOT.json').write_bytes(raw)
 print(str(dest))
 print('INSPECT_OK')
finally:pw=None;c.close()
