"""Stop only the verified old dispatcher and TAC subtree for the user's AUX8 revision."""
import datetime,getpass,json,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
SCRIPT=r'''
import os,json,signal,time,hashlib,subprocess
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/CAETv2_TAC_RSET_EFTR_OBJECT41_20261022')
def procs():
 out={}
 for p in Path('/proc').glob('[0-9]*'):
  try:
   cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode();cwd=str((p/'cwd').resolve());st=(p/'stat').read_text().split(') ',1)[1].split()
   if cmd:out[int(p.name)]={'cmd':cmd,'cwd':cwd,'ppid':int(st[1]),'state':st[0]}
  except (OSError,ValueError):pass
 return out
ps=procs();cp=json.loads((Q/'controller.pid.json').read_text())['pid'];before={str(k):v for k,v in ps.items() if (str(Q) in v['cmd'] or v['cwd'].startswith(str(Q))) and k!=os.getpid() and 'python -c' not in v['cmd']}
contract=json.loads((Q/'ORCHESTRATION_CONTRACT.json').read_text());bad=[p for p,h in contract['files'].items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];assert not bad,bad
if cp in ps:
 assert str(Q) in ps[cp]['cmd'] and 'queue_controller.py' in ps[cp]['cmd'],ps[cp]
 os.kill(cp,signal.SIGSTOP)
ps=procs();roots={k for k,v in ps.items() if k!=os.getpid() and 'python -c' not in v['cmd'] and (v['cwd'].startswith(str(Q/'TAC')+'/') or v['cwd']==str(Q/'TAC') or str(Q/'TAC'/'ENTRY.json') in v['cmd'] or str(Q/'TAC'/'seed_supervisor.py') in v['cmd'])}
targets=set(roots)
while True:
 more={k for k,v in ps.items() if v['ppid'] in targets};new=targets|more
 if new==targets:break
 targets=new
targets.discard(os.getpid());assert cp not in targets
if cp in ps:os.kill(cp,signal.SIGTERM);os.kill(cp,signal.SIGCONT)
for k in sorted(targets,reverse=True):
 try:os.kill(k,signal.SIGTERM)
 except ProcessLookupError:pass
time.sleep(3)
after=procs();forced=[]
for k in targets|{cp}:
 if k in after and after[k]['state']!='Z':
  assert k in ps and after[k]['cmd']==ps[k]['cmd'];os.kill(k,signal.SIGKILL);forced.append(k)
time.sleep(1);after=procs();remaining={str(k):after[k] for k in targets|{cp} if k in after and after[k]['state']!='Z'}
o=Path('/data/mlr_ifa/experiments/CAETv2_TAC_OBJECT_seed41_20261022');ep=json.loads((o/'ra_coreset/epochs.json').read_text()) if (o/'ra_coreset/epochs.json').exists() else []
r={'status':'STOPPED_BY_USER_FOR_AUX8_REVISION','unix':time.time(),'controller_pid':cp,'before':before,'target_pids':sorted(targets),'forced_pids':forced,'remaining':remaining,'completed_epochs':len(ep),'last_epoch':ep[-1] if ep else None,'formal_records':sum(len(p.read_text().splitlines()) for p in (o/'ra_coreset/evaluation').glob('trials*.jsonl')),'hash_count':len(contract['files']),'hash_failures':bad,'old_run_not_complete':True,'pending_RSET_EFTR_not_cancelled':True,'gpu_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)}
assert not remaining,remaining
p=Q/'TAC_STOPPED_FOR_AUX8_USER_REVISION.json';assert not p.exists();p.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
'''
def main():
 p=R/('tac_user_stop_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));p.mkdir()
 c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
 try:
  c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,allow_agent=False,look_for_keys=False,timeout=20);pw=None
  i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(SCRIPT),timeout=180);raw=o.read();err=e.read().decode();rc=o.channel.recv_exit_status();(p/'ATTEMPT.json').write_text(json.dumps({'exit':rc,'stderr':err},indent=2));assert rc==0,err
  (p/'STOP_RECEIPT.json').write_bytes(raw);d=json.loads(raw);print(json.dumps({'directory':str(p),'status':d['status'],'epochs':d['completed_epochs'],'formal':d['formal_records'],'remaining':d['remaining'],'gpu':d['gpu_processes']},ensure_ascii=False))
 finally:pw=None;c.close()
if __name__=='__main__':main()
