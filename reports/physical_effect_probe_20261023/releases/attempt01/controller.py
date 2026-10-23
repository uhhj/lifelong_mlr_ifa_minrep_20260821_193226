"""Finite gate-A controller. Never starts an unaudited B implementation."""
import os,sys,fcntl,json,subprocess,time,traceback
from pathlib import Path
from physical_effect_probe.common import ROOT,OUT,write,sha,verify,capacity
def main():
 lock=open('/data/mlr_ifa/physical_effect_probe.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True);assert not gpu.strip(),('occupied GPU',gpu)
 assert not (OUT/'EVENTS.jsonl').exists(),'No implicit restart';verify();capacity(1024**3)
 write(OUT/'RUN_STATUS.json',dict(status='PARTIAL',phase='A_START',pid=os.getpid(),unix=time.time()))
 for phase in ('binding_audit','export_native_view','visibility_test','train_probe','audit'):
  verify();write(OUT/'RUN_STATUS.json',dict(status='PARTIAL',phase=phase,pid=os.getpid(),unix=time.time()))
  with (ROOT/(phase+'.log')).open('wb') as f:
   p=subprocess.Popen([sys.executable,'-m','physical_effect_probe.'+phase],stdout=f,stderr=subprocess.STDOUT,cwd=str(ROOT));write(ROOT/(phase+'.process.json'),dict(pid=p.pid,parent=os.getpid(),start_unix=time.time()));rc=p.wait()
  (ROOT/(phase+'.exit')).write_text(str(rc)+'\n');assert rc==0,(phase,rc)
 gate=json.loads((OUT/'A_GATE.json').read_text())
 status=gate['status'] if not gate['selected'] else 'A_PASSED_B_ADMISSION_PENDING'
 write(OUT/'RUN_STATUS.json',dict(status=status,phase='A_FINISHED',gate_sha256=sha(OUT/'A_GATE.json'),pid=os.getpid(),unix=time.time()))
 (ROOT/'controller.exit').write_text('0\n')
if __name__=='__main__':
 try:main()
 except BaseException:
  write(OUT/'FAILURE.json',dict(status='ENGINEERING_FAILURE',traceback=traceback.format_exc(),unix=time.time()));(ROOT/'controller.exit').write_text('1\n');raise
