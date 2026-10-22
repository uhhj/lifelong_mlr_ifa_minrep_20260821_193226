"""Install six bounded deferred entries; never touch the active prior worker."""
import datetime,getpass,hashlib,json,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
Q='/data/mlr_ifa/experiment_workspaces/TACR_TACST_OBJECT_conditional_20261023'
SCRIPT=r'''
import sys,os,json,time,hashlib,subprocess,shutil
from pathlib import Path
d=json.load(sys.stdin);Q=Path(d['q']);assert not Q.exists();B=Path(d['base']);old=Path(d['prior']['queue'])
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(old/'ORCHESTRATION_CONTRACT.json')==d['prior']['contract_sha256']
assert all(sha(B/n)==h for n,h in d['base_sha256'].items())
state=json.loads((old/'RUN_STATUS.json').read_text());pending=sum(not Path(e['output']).exists() for e in json.loads((old/'QUEUE.json').read_text()))
if (old/'controller.exit').exists():assert (old/'controller.exit').read_text().strip()=='0'
cpu=d['cpu'];assert cpu['status']=='ACTUAL_TEMPORAL_GMM_CPU_GRADIENT_PASS' and not cpu['cuda_initialized'] and cpu['optimizer_calls']==cpu['environment_steps']==cpu['server_writes']==0
files={};diff={};Q.mkdir()
def wr(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
for n,s in d['top'].items():
 if n.endswith('.py'):compile(s,n,'exec')
 p=Q/n;p.write_text(s);files[str(p)]=sha(p)
wr(Q/'PRIOR_QUEUE.json',d['prior']);files[str(Q/'PRIOR_QUEUE.json')]=sha(Q/'PRIOR_QUEUE.json')
for e in d['entries']:
 W=Q/e['key'];W.mkdir();S=W/'source';assert not Path(e['output']).exists();parent=Path(e['parent'])
 assert sha(parent/'CONTRACT.json')==e['parent_contract_sha256'] and json.loads((parent/'AUTONOMOUS_EXIT_ACCEPTANCE.json').read_text())['status']=='PASS'
 shutil.copytree(B,S,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for n,s in d['source'].items():
  compile(s,n,'exec');p=S/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
 actual={str(p.relative_to(S)):sha(p) for p in S.rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.sh') and '__pycache__' not in p.parts}
 changes={n:dict(old=d['base_sha256'].get(n),new=h) for n,h in actual.items() if h!=d['base_sha256'].get(n)}
 assert set(changes)<=set(d['source']) and all(n in actual for n in d['base_sha256'])
 diff[e['key']]=changes;files.update({str(S/n):h for n,h in actual.items()})
 wr(W/'ENTRY.json',e);files[str(W/'ENTRY.json')]=sha(W/'ENTRY.json')
 for n,raw in [('PROTOCOL.md',d['top']['PROTOCOL.md']),('GATE1_EVIDENCE.json',d['gates'][str(e['seed'])]['evidence']),('GATE1_LOCAL_AUDIT.json',d['gates'][str(e['seed'])]['audit'])]:
  (W/n).write_text(raw);files[str(W/n)]=sha(W/n)
wr(Q/'QUEUE.json',d['entries']);files[str(Q/'QUEUE.json')]=sha(Q/'QUEUE.json')
wr(Q/'SOURCE_DIFF.json',diff);files[str(Q/'SOURCE_DIFF.json')]=sha(Q/'SOURCE_DIFF.json')
# Full worst-case storage: old pending entries plus all six new runs. No old payload is moved/deleted.
need={'/':0,'/data':pending*int(1.6*1024**3)}
for e in d['entries']:need[e['volume']]+=e['storage_bound_bytes']
capacity={}
for m in ('/','/data'):
 v=os.statvfs(m);free=(v.f_bfree if m=='/data' else v.f_bavail)*v.f_frsize;reserve=(2 if m=='/data' else 1)*1024**3
 assert free>need[m]+reserve,('RESOURCE_CAPACITY_FULL_DEFERRED_QUEUE',m,free,need[m],reserve)
 capacity[m]=dict(free=free,reserve=reserve,pending_bound=need[m],free_basis='root f_bfree' if m=='/data' else 'ordinary f_bavail')
wr(Q/'CAPACITY_PLAN.json',dict(status='WORST_CASE_START_BOUND_PASS_NOT_FINAL_CAPACITY_ACCEPTANCE',measurements=d['storage_measurements'],old_pending_runs=pending,volumes=capacity,old_assets_deleted=0,unix=time.time()));files[str(Q/'CAPACITY_PLAN.json')]=sha(Q/'CAPACITY_PLAN.json')
wr(Q/'CPU_VALIDATION.json',cpu);files[str(Q/'CPU_VALIDATION.json')]=sha(Q/'CPU_VALIDATION.json')
wr(Q/'ORCHESTRATION_CONTRACT.json',dict(files=files,user_scope='OBJECT TACR / independent TACRST seed41; each > same-seed RA41 AUC then its42/43',gate_baseline_exact='701/960',prior_queue=d['prior'],base_source=str(B),base_source_sha256=d['base_sha256'],created_unix=time.time()))
with (Q/'controller.log').open('x') as f:p=subprocess.Popen(['/data/mlr_ifa/env/bin/python','-u',str(Q/'controller.py')],cwd=Q,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='DISPATCHED_WAITING_PRIOR_QUEUE_NO_GPU_ADMISSION',controller_pid=p.pid,queue=str(Q),entries=[e['key'] for e in d['entries']],frozen_files=len(files),contract_sha256=sha(Q/'ORCHESTRATION_CONTRACT.json'),prior_state=state,capacity=capacity,unix=time.time())
wr(Q/'DISPATCH_RECEIPT.json',receipt);print(json.dumps(receipt))
'''
def main():
 A=R.parent/'20261022_align_object_extension_goal_conditional';inv=json.loads((A/'inventory_readonly_20261022_223503/EVIDENCE.json').read_text(encoding='utf8'));gidx=json.loads((A/'GATE1_INDEX.json').read_text(encoding='utf8'))
 entries=[]
 for variant,volumes in [('TACR',['/','/data','/']),('TACRST',['/data','/','/data'])]:
  for seed,vol in zip((41,42,43),volumes):
   parent=inv['parents']['OBJECT_'+str(seed)]['path'];proof=json.loads((A/gidx['directory']/('OBJECT_'+str(seed)+'.json')).read_text(encoding='utf8'))
   entries.append(dict(key=variant+'_'+str(seed),variant=variant,seed=seed,suite='OBJECT',parent=parent,parent_contract_sha256=proof['protected'][parent+'/CONTRACT.json'],output=('/root/mlr_ifa/experiments/' if vol=='/' else '/data/mlr_ifa/experiments/')+variant+'_OBJECT_seed'+str(seed)+'_20261023',volume=vol,storage_bound_bytes=int(1.6*1024**3),spec=dict(optimizer_steps=105500,dataset_sizes={str(t):n for t,n in zip(range(6,10),(1562,2261,3081,3620))})))
 gates={str(s):dict(evidence=(R/('GATE1_seed%d_EVIDENCE.json'%s)).read_text(encoding='utf8'),audit=(R/('GATE1_seed%d_LOCAL_AUDIT.json'%s)).read_text(encoding='utf8')) for s in (41,42,43)}
 assert all(json.loads(g['audit'])['status']=='PASS' for g in gates.values())
 top={n:(R/n).read_text(encoding='utf8') for n in ['PROTOCOL.md','controller.py','seed_supervisor.py','independent_audit.py','formal_audit.py']}
 source={p.relative_to(R/'additions').as_posix():p.read_text(encoding='utf8') for p in (R/'additions').rglob('*.py')}
 cpu=json.loads((sorted(R.glob('actual_cpu_*/EVIDENCE.json'))[-1]).read_text(encoding='utf8'));storage=json.loads((R/'cpu_checks_20261023_000439/EVIDENCE.json').read_text(encoding='utf8'))['original_complete_run_bytes']
 prior=dict(queue='/data/mlr_ifa/experiment_workspaces/ALIGN_OBJECT42_43_GOAL_conditional_20261022',contract_sha256='8c35a51e3bf0d4c3c7a5b55b642a88d6b2dca861cd3c7099fd31fbc94e8e21ca')
 data=dict(q=Q,base=inv['source_root'],base_sha256=inv['source_sha256'],entries=entries,gates=gates,top=top,source=source,cpu=cpu,storage_measurements=storage,prior=prior)
 out=R/('launch_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir();c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
 try:
  c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
  i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(SCRIPT),timeout=240);i.write(json.dumps(data));i.flush();i.channel.shutdown_write();raw=o.read();err=e.read().decode();rc=o.channel.recv_exit_status()
  (out/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err),indent=2)+'\n');assert rc==0,err
  (out/'DISPATCH_RECEIPT.json').write_bytes(raw);print(out);print(raw.decode())
 finally:pw=None;c.close()
if __name__=='__main__':main()
