"""Deploy a one-file versioned resolver repair, leaving failed contracts untouched."""
import datetime,getpass,hashlib,json,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
Q='/data/mlr_ifa/experiment_workspaces/CAET_v2_OBJECT_extension_LIBERO50_20261021'
D=Q+'/libero50_taskmap01'
SCRIPT=r'''
import json,hashlib,os,sys,subprocess,time,fcntl,signal
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/CAET_v2_OBJECT_extension_LIBERO50_20261021');D=Q/'libero50_taskmap01'
oldq=Path('/data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
lock=(Q/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert (Q/'controller.exit').read_text().strip()=='1' and (Q/'capacity_recovery_v1/watcher.exit').read_text().strip()=='1'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
# Stop only the obsolete read-only inspector awaiting an absent native config.
helpers=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if int(p.name)!=os.getpid() and "d['task_mapping']=[]" in cmd and 'python -c' in cmd and "assert os.environ.get('LIBERO_CONFIG_PATH')" not in cmd:
   os.kill(int(p.name),signal.SIGTERM);helpers.append(int(p.name))
 except (OSError,ProcessLookupError):pass
old=Path('/data/mlr_ifa/experiments/LIBERO50_reproduced_pretraining_seed42_20261021')
assert sorted(p.name for p in old.iterdir())==['CONTRACT.json','RUN_STATUS.json','resolved_pretrain.json']
assert (Q/'libero50/shared_pretraining/admission/gpu_preflight.exit').read_text().strip()=='1'
original=json.loads((oldq/'LIBERO50_SOURCE_BUNDLE_final.json').read_text());new=json.loads((D/'TASKMAP01_BUNDLE.json').read_text())
assert sha(oldq/'LIBERO50_SOURCE_BUNDLE_final.json')=='370dd4645bb319f0f1739fa13b5a84fa82d86c7c0e0078be0da28d8caad718ae'
assert [k for k in original if original[k]!=new[k]]==['l50_exp/data.py'] and len(new)==196
sys.path.insert(0,str(oldq));from worker_entrypoint import install
audit_source=D/'mapping_source';install(D/'TASKMAP01_BUNDLE.json',audit_source,sha(D/'TASKMAP01_BUNDLE.json'))
env=dict(os.environ,PYTHONPATH=str(audit_source),PYTHONDONTWRITEBYTECODE='1',LIBERO50_WORKSPACE=str(Q/'libero50/shared_pretraining'),LIBERO50_OUT=str(old),LIBERO50_SEED='42',CUDA_VISIBLE_DEVICES='0',MUJOCO_GL='egl',EGL_PLATFORM='surfaceless')
code="""
import json,torch
from l50_exp.data import native_task,partition
from libero.libero.benchmark import task_maps
rows=partition()['pretraining']+sum(partition()['lifelong_stages'],[])
identities=[]
for row in rows:
 t=native_task(row);assert t==task_maps[row['source_suite']][row['name']]
 identities.append(dict(name=t.name,language=t.language,bddl=t.problem_folder+'/'+t.bddl_file,init=t.problem_folder+'/'+t.init_states_file))
assert len(identities)==50 and len(set(x['name'] for x in identities))==50
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS_50_NATIVE_IDENTITIES',identities=identities,optimizer_updates=0,model_forwards=0,environment_steps=0,cuda_initialized=False)))
"""
r=subprocess.run([sys.executable,'-c',code],cwd=audit_source,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=60)
write(D/'mapping_attempt.json',dict(exit=r.returncode,stderr=r.stderr,stdout=r.stdout));assert r.returncode==0,r.stderr
proof=json.loads(r.stdout.splitlines()[-1]);write(D/'MAPPING_AUDIT.json',proof)
entries=[];files={}
shared_w=D/'shared_pretraining';shared_o='/data/mlr_ifa/experiments/LIBERO50_reproduced_pretraining_seed42_20261021_taskmap01'
assert not Path(shared_o).exists()
for seed in (41,42,43):
 oldentry=json.loads((Q/('LIBERO50_seed%d.json'%seed)).read_text());e=dict(oldentry)
 e.update(source=str(D/('seed%d'%seed)/'source'),shared_pretrain_workspace=str(shared_w),shared_pretrain_output=shared_o,bundle=str(D/'TASKMAP01_BUNDLE.json'),bundle_sha256=sha(D/'TASKMAP01_BUNDLE.json'),output=oldentry['output']+'_taskmap01',engineering_version='taskmap01')
 assert not Path(e['output']).exists()
 ep=D/('LIBERO50_seed%d.json'%seed);write(ep,e)
 shell=D/('LIBERO50_seed%d.sh'%seed);shell.write_text('#!/usr/bin/env bash\nset -euo pipefail\nsource /data/mlr_ifa/activate.sh\nexec /data/mlr_ifa/env/bin/python '+str(oldq/'worker_entrypoint_libero50.py')+' '+str(ep)+'\n')
 entries.append(dict(key='RA_IFA_LIBERO50_%d'%seed,seed=seed,output=e['output'],storage_mount='/' if seed==41 else '/data',argv=['bash',str(shell)]))
 for f in (ep,shell):files[str(f)]=sha(f)
write(D/'QUEUE.json',entries)
for f in (D/'QUEUE.json',D/'libero50_taskmap01_controller.py',D/'TASKMAP01_BUNDLE.json',D/'MAPPING_AUDIT.json',Q/'ORCHESTRATION_CONTRACT.json',Q/'FAILURE.json',old/'CONTRACT.json',oldq/'worker_entrypoint_libero50.py'):
 files[str(f)]=sha(f)
for p,h in json.loads((Q/'ORCHESTRATION_CONTRACT.json').read_text())['files'].items():assert sha(p)==h;files[p]=h
write(D/'CONTRACT.json',dict(version='taskmap01',files=files,changed_scientific_hyperparameters=False,changed_source_files=['l50_exp/data.py'],original_failure_retained=True,original_updates=0,original_environments=0,source_fix='exact full task_maps lookup preserving partition identity',helper_cleanup=helpers,unix=time.time()))
lock.close()
with (D/'controller.log').open('x') as f:p=subprocess.Popen([sys.executable,'-u',str(D/'libero50_taskmap01_controller.py')],cwd=D,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(status='TASKMAP01_DISPATCHED',pid=p.pid,contract_sha256=sha(D/'CONTRACT.json'),bundle_sha256=sha(D/'TASKMAP01_BUNDLE.json'),mapping=proof['status'],entries=entries,original_outputs_untouched=True,helper_cleanup=helpers)))
'''
dest=R/('taskmap01_deploy_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));dest.mkdir()
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
try:
 c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
 s=c.open_sftp();s.mkdir(D)
 for local,name in [('LIBERO50_TASKMAP01_BUNDLE.json','TASKMAP01_BUNDLE.json'),('libero50_taskmap01_controller.py','libero50_taskmap01_controller.py')]:
  with s.open(D+'/'+name,'wx') as f:f.write((R/local).read_bytes())
 i,o,e=c.exec_command('source /data/mlr_ifa/activate.sh && /data/mlr_ifa/env/bin/python -c '+shlex.quote(SCRIPT),timeout=180);i.channel.shutdown_write();raw=o.read();err=e.read();rc=o.channel.recv_exit_status()
 (dest/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stdout=raw.decode(),stderr=err.decode()),indent=2),encoding='utf8');assert rc==0,err.decode()
 (dest/'RECEIPT.json').write_text(raw.decode().splitlines()[-1]+'\n',encoding='utf8')
 for n in ('CONTRACT.json','MAPPING_AUDIT.json','mapping_attempt.json','QUEUE.json'):
  with s.open(D+'/'+n,'rb') as f:(dest/n).write_bytes(f.read())
 print(dest);print(raw.decode().splitlines()[-1])
finally:pw=None;c.close()
