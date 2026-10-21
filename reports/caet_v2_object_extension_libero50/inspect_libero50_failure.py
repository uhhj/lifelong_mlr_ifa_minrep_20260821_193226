"""Fixed read-only diagnosis of the first LIBERO50 admission failure."""
import datetime,getpass,json,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
SCRIPT=r'''
import json,time,hashlib,subprocess
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/CAET_v2_OBJECT_extension_LIBERO50_20261021')
W=Q/'libero50/shared_pretraining';O=Path('/data/mlr_ifa/experiments/LIBERO50_reproduced_pretraining_seed42_20261021')
d={'unix':time.time(),'files':{},'hashes':{},'processes':[]}
def read(p):
 if p.is_file() and p.suffix in ('.json','.exit','.log','.sh','.py'):
  raw=p.read_bytes();assert len(raw)<8*1024**2
  d['files'][str(p)]=json.loads(raw) if p.suffix=='.json' else raw.decode()[-40000:]
for n in ('RA_IFA_LIBERO50_41.log','RA_IFA_LIBERO50_41.exit','LIBERO50_seed41.json','LIBERO50_seed41.sh','FAILURE.json','controller.exit','ORCHESTRATION_CONTRACT.json'):read(Q/n)
for p in (W/'admission').glob('*'):read(p)
for p in O.glob('*.json'):read(p)
for p in O.glob('*.exit'):read(p)
c=json.loads((O/'CONTRACT.json').read_text())
for p,h in c.get('files',{}).items():d['hashes'][p]=Path(p).is_file() and hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
for n,h in c.get('source_hashes',{}).items():d['hashes']['source/'+n]=hashlib.sha256((W/'source'/n).read_bytes()).hexdigest()==h
import sys,os,io,contextlib
assert os.environ.get('LIBERO_CONFIG_PATH'), 'Use the established native environment'
sys.path.insert(0,str(W/'source'))
from libero.libero.benchmark import task_maps,get_benchmark
partition=json.loads((W/'PARTITION.json').read_text());rows=partition['pretraining']+sum(partition['lifelong_stages'],[])
d['task_mapping']=[]
cfg=json.loads((O/'resolved_pretrain.json').read_text())
for row in rows:
 t=task_maps[row['source_suite']][row['name']]
 try:old=[x for x in get_benchmark(row['source_suite'])(0).tasks if x.name==row['name']];err=None
 except Exception as ex:old=[];err=repr(ex)
 assert t.problem_folder+'/'+t.bddl_file==row['bddl_relative_path']
 assert t.problem_folder+'/'+t.init_states_file==row['init_relative_path']
 assert t.problem_folder+'/'+t.name+'_demo.hdf5'==row['dataset_relative_path']
 d['task_mapping'].append(dict(name=t.name,suite=t.problem_folder,language=t.language,old_matches=len(old),old_error=err,old_identical=old==[t] if old else None,bddl_sha256=hashlib.sha256((Path(cfg['bddl_folder'])/row['bddl_relative_path']).read_bytes()).hexdigest()))
assert len(d['task_mapping'])==50
d['output_files']=[str(p.relative_to(O)) for p in O.rglob('*') if p.is_file()]
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if 'python -c' not in cmd and any(x in cmd for x in ('l50_exp.','worker_entrypoint_libero50','queue_controller.py','capacity_recovery_v1.py')):d['processes'].append({'pid':int(p.name),'command':cmd})
 except OSError:pass
d['gpu']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
print(json.dumps(d))
'''
dest=R/('libero50_failure_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));dest.mkdir()
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
try:
 c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
 i,o,e=c.exec_command('source /data/mlr_ifa/activate.sh && /data/mlr_ifa/env/bin/python -c '+shlex.quote(SCRIPT),timeout=180);i.channel.shutdown_write();raw=o.read();err=e.read();rc=o.channel.recv_exit_status()
 (dest/'ATTEMPT.json').write_text(json.dumps({'exit':rc,'stderr':err.decode()},indent=2),encoding='utf8');assert rc==0,err.decode()
 # Native benchmark construction can print informational text before JSON.
 lines=raw.decode().splitlines();payload=lines[-1];json.loads(payload)
 (dest/'NATIVE_IMPORT_STDOUT.md').write_text('\n'.join(lines[:-1]),encoding='utf8')
 (dest/'SNAPSHOT.json').write_text(payload+'\n',encoding='utf8');print(dest)
finally:pw=None;c.close()
