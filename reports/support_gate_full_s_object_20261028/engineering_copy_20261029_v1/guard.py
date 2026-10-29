"""Guard only the waiting coordinator; never signal a training process/group."""
import os,json,time,signal,hashlib,subprocess,sys
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/SUPPORT_GATE_FULL_S_OBJECT41_42_43_20261028_v1')
E=Q/'engineering_copy_20261029_v1'
def proc(pid):
 p=Path('/proc')/str(pid);s=(p/'stat').read_text().split(') ',1)[1].split()
 return dict(pid=pid,state=s[0],start=s[19],exit_code=int(s[49]),cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not E.exists(),'Guard is single use'
 c=json.loads((Q/'CONTRACT.json').read_text());assert all(sha(p)==h for p,h in c['files'].items())
 before=proc(3013);assert before['state']=='S' and before['cmd'].strip().endswith(str(Q/'controller.py'))
 status=json.loads((Q/'STATUS.json').read_text());assert status['phase']=='train' and status['seed']==42,status
 child=proc(status['pid']);assert '/seed42/train.py S' in child['cmd'] and 'torch.distributed.run' in child['cmd']
 assert not (Q/'seed43/train_S.log').exists()
 E.mkdir();receipt=dict(status='BOUNDARY_GUARD_ONLY',time=time.time(),controller=before,child=child,deadline=time.time()+5400,contract_sha256=sha(Q/'CONTRACT.json'),training_signals=0)
 (E/'GUARD.json').write_text(json.dumps(receipt,indent=2))
 # A separate watchdog releases the coordinator on validation timeout/failure.
 watchdog='''import json,time,os,signal\nfrom pathlib import Path\np=Path(%r);r=json.loads((p/'GUARD.json').read_text())\nwhile time.time()<r['deadline']:\n if (p/'HANDOFF_COMMITTED.json').exists():raise SystemExit\n time.sleep(5)\ns=Path('/proc')/str(r['controller']['pid'])/'stat'\nif s.exists():\n a=s.read_text().split(') ',1)[1].split()\n if a[19]==r['controller']['start'] and a[0]=='T':\n  os.kill(r['controller']['pid'],signal.SIGCONT)\n  (p/'GUARD_TIMEOUT.json').write_text(json.dumps({'time':time.time(),'action':'SIGCONT original coordinator only'}))\n''' % str(E)
 (E/'watchdog.py').write_text(watchdog)
 with (E/'watchdog.log').open('x') as f:
  w=subprocess.Popen([sys.executable,'-B',str(E/'watchdog.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 os.kill(3013,signal.SIGSTOP)
 try:
  time.sleep(.2);after=proc(3013);assert after['state']=='T' and after['start']==before['start']
  assert not (Q/'seed43/train_S.log').exists(),'Boundary raced; release original coordinator'
  assert json.loads((Q/'STATUS.json').read_text())==status
  assert proc(child['pid'])['start']==child['start']
  receipt.update(controller_after=after,watchdog_pid=w.pid,child_after=proc(child['pid']))
  (E/'GUARD.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
 except BaseException:
  os.kill(3013,signal.SIGCONT);raise
if __name__=='__main__':main()
