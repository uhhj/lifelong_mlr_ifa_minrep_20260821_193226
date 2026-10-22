"""Install a versioned finite queue; credentials only via getpass in memory."""
import datetime,getpass,hashlib,json,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
Q='/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022'
SCRIPT=r'''
import sys,os,json,time,hashlib,subprocess,shutil,fcntl
from pathlib import Path
d=json.load(sys.stdin);Q=Path(d['q']);assert not Q.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace');cwd=str((p/'cwd').resolve())
  if int(p.name)!=os.getpid() and 'python -c' not in cmd:assert '/experiment_workspaces/' not in cmd and '/experiment_workspaces/' not in cwd,('existing worker',p.name)
 except (OSError,ValueError):pass
B=Path(d['base']);assert all(sha(B/n)==h for n,h in d['base_sha256'].items())
prior=Path('/data/mlr_ifa/experiment_workspaces/CAETv3_TAC_EFTR_ALIGN_OBJECT41_20261022_collate01/align_after_eftr_stop01')
assert (prior/'controller.exit').read_text().strip()=='0'
assert all(sha(p)==h for p,h in json.loads((prior/'CONTRACT.json').read_text())['files'].items())
prefix=d['prefix'];old=Path(prefix['output']);assert (old/'pipeline.exit').read_text().strip()=='0'
assert sha(old/'PROCESS_EXIT_ACCEPTANCE.json')==prefix['acceptance_sha256']
# The five-run bound includes copied RA buffers, four models, recovery, admission recovery, and light logs.
v=os.statvfs('/data');rootfree=v.f_bfree*v.f_frsize;future=sum(e['storage_bound_bytes'] for e in d['entries'])
assert rootfree>future+2*1024**3,('RESOURCE_CAPACITY_FULL_QUEUE',rootfree,future)
lock=(prior.parent/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
Q.mkdir();files={};diff={}
def wr(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
for n,s in d['top'].items():
 p=Q/n;p.write_text(s);files[str(p)]=sha(p)
wr(Q/'ACCEPTED_OBJECT41.json',prefix);files[str(Q/'ACCEPTED_OBJECT41.json')]=sha(Q/'ACCEPTED_OBJECT41.json')
for e in d['entries']:
 W=Q/e['key'];W.mkdir();S=W/'source';assert not Path(e['output']).exists()
 parent=Path(e['parent']);assert sha(parent/'CONTRACT.json')==e['parent_contract_sha256']
 assert json.loads((parent/e['parent_acceptance']).read_text())['status']=='PASS'
 shutil.copytree(B,S,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for n,s in d['source'].items():
  compile(s,n,'exec');p=S/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
 actual={str(p.relative_to(S)):sha(p) for p in S.rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.sh') and '__pycache__' not in p.parts}
 changes={n:dict(old=d['base_sha256'].get(n),new=h) for n,h in actual.items() if h!=d['base_sha256'].get(n)}
 assert set(changes)<=set(d['source']),set(changes)-set(d['source'])
 assert all(n in actual for n in d['base_sha256'])
 diff[e['key']]=changes;files.update({str(S/n):h for n,h in actual.items()})
 wr(W/'ENTRY.json',e);files[str(W/'ENTRY.json')]=sha(W/'ENTRY.json')
 for n in ['PROTOCOL.md','GATE1_EVIDENCE.json','GATE1_LOCAL_AUDIT.json']:
  raw=d['top']['PROTOCOL.md'] if n=='PROTOCOL.md' else d['gates'][e['key']][n]
  (W/n).write_text(raw);files[str(W/n)]=sha(W/n)
wr(Q/'QUEUE.json',d['entries']);files[str(Q/'QUEUE.json')]=sha(Q/'QUEUE.json')
wr(Q/'SOURCE_DIFF.json',diff);files[str(Q/'SOURCE_DIFF.json')]=sha(Q/'SOURCE_DIFF.json')
wr(Q/'CAPACITY_PLAN.json',dict(status='FULL_FIVE_ENTRY_START_BOUND_PASS_NOT_FINAL_CAPACITY_ACCEPTANCE',observed_unix=time.time(),data_root_free=rootfree,reserve_bytes=2*1024**3,all_pending_bound_bytes=future,per_entry_bound_bytes={e['key']:e['storage_bound_bytes'] for e in d['entries']},new_feature_bytes=0,old_assets_deleted=0));files[str(Q/'CAPACITY_PLAN.json')]=sha(Q/'CAPACITY_PLAN.json')
wr(Q/'ORCHESTRATION_CONTRACT.json',dict(files=files,user_scope='ALIGN OBJECT42/43; mean AUC above same-seed RA then GOAL41/42/43',strict_greater_only=True,base_source=str(B),base_source_sha256=d['base_sha256'],created_unix=time.time()))
fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
with (Q/'controller.log').open('x') as f:p=subprocess.Popen(['/data/mlr_ifa/env/bin/python','-u',str(Q/'controller.py')],cwd=Q,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(status='DISPATCHED_AWAITING_OBJECT42_OWN_GPU_ADMISSION',controller_pid=p.pid,queue=str(Q),entries=[e['key'] for e in d['entries']],prefix='OBJECT_41_ACCEPTED_READ_ONLY',frozen_files=len(files),contract_sha256=sha(Q/'ORCHESTRATION_CONTRACT.json'),source_diff=diff,capacity= json.loads((Q/'CAPACITY_PLAN.json').read_text()),unix=time.time())
wr(Q/'DISPATCH_RECEIPT.json',r);print(json.dumps(r))
'''
def main():
 inv=json.loads((R/'inventory_readonly_20261022_223503/EVIDENCE.json').read_text(encoding='utf8'))
 gateidx=json.loads((R/'GATE1_INDEX.json').read_text(encoding='utf8'));assert json.loads((R/gateidx['directory']/'LOCAL_AUDIT.json').read_text())['status']=='PASS'
 entries=[];gates={}
 for suite,seeds in [('OBJECT',(42,43)),('GOAL',(41,42,43))]:
  for seed in seeds:
   key=suite+'_'+str(seed);v=inv['parents'][key];cfg=v['resolved_ra_coreset.json'];n=[1562,2261,3081,3620] if suite=='OBJECT' else [997,1388,1848,3365]
   entry=dict(key=key,suite=suite,seed=seed,variant='ALIGN',parent=v['path'],parent_contract_sha256=json.loads((R/gateidx['directory']/(key+'.json')).read_text())['protected'][v['path']+'/CONTRACT.json'],parent_acceptance='AUTONOMOUS_EXIT_ACCEPTANCE.json' if suite=='OBJECT' else 'PROCESS_EXIT_ACCEPTANCE.json',output='/data/mlr_ifa/experiments/ALIGN_%s_seed%d_20261022'% (suite,seed),storage_bound_bytes=2*1024**3,
    spec=dict(benchmark_name=cfg['benchmark_name'],gamma=cfg['gamma'],pretrain=cfg['pretrain'],optimizer_steps=105500 if suite=='OBJECT' else 76100,dataset_sizes={str(t):n[t-6] for t in range(6,10)}))
   entries.append(entry);gates[key]={'GATE1_EVIDENCE.json':(R/gateidx['directory']/(key+'.json')).read_text(), 'GATE1_LOCAL_AUDIT.json':(R/gateidx['directory']/(key+'_LOCAL_AUDIT.json')).read_text()}
 old=R.parent/'20261022_caet_v3_tac_eftr_object41';final=old/'final_ALIGN_seed41_20261022_220252'
 audit=json.loads((final/'LOCAL_FINAL_AUDIT.json').read_text());qcost=json.loads((final/'QUEUE_TASK_COST_AUDIT.json').read_text());assert audit['status']=='COMPLETED_LOCAL_INDEPENDENT_PASS'
 prefix=dict(key='OBJECT_41',seed=41,suite='OBJECT',output='/data/mlr_ifa/experiments/CAETv3_ALIGN_OBJECT_seed41_20261022_collate01',parent=inv['parents']['OBJECT_41']['path'],local_final_audit=audit,local_final_audit_sha256=hashlib.sha256((final/'LOCAL_FINAL_AUDIT.json').read_bytes()).hexdigest(),acceptance_sha256=qcost['receipt']['acceptance_sha256'])
 top={n:(R/n).read_text(encoding='utf8') for n in ['PROTOCOL.md','controller.py','seed_supervisor.py','independent_audit.py','formal_audit.py']}
 source={p.relative_to(R/'additions').as_posix():p.read_text(encoding='utf8') for p in (R/'additions').rglob('*.py')}
 for n,s in [*top.items(),*source.items()]:
  if n.endswith('.py'):compile(s,n,'exec')
 data=dict(q=Q,base=inv['source_root'],base_sha256=inv['source_sha256'],top=top,source=source,entries=entries,gates=gates,prefix=prefix)
 out=R/('launch_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir()
 c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
 try:
  c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
  i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(SCRIPT),timeout=240);i.write(json.dumps(data));i.flush();i.channel.shutdown_write();raw=o.read();err=e.read().decode();rc=o.channel.recv_exit_status()
  (out/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err),indent=2)+'\n');assert rc==0,err
  (out/'DISPATCH_RECEIPT.json').write_bytes(raw);print(out);print(raw.decode()[:2000])
 finally:pw=None;c.close()
if __name__=='__main__':main()
