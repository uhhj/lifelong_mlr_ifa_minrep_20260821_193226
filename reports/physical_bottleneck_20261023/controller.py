import os,sys,json,time,fcntl,subprocess,traceback
from pbd.common import ROOT,OUT,write,verify,capacity,sha
def main():
 lock=open('/data/mlr_ifa/physical_effect_probe.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied'
 assert not (OUT/'EVENTS.jsonl').exists(),'No implicit restart';verify();capacity(int(3.3*1024**3))
 phases=[('readout_admission',['pbd.admit','readout']),('readout',['pbd.train','readout']),('patch_export',['pbd.export_patch']),('spatial_admission',['pbd.admit','spatial']),('spatial',['pbd.train','spatial']),('audit',['pbd.audit'])]
 for name,args in phases:
  verify();write(OUT/'RUN_STATUS.json',dict(status='PARTIAL',phase=name,controller=os.getpid(),unix=time.time()))
  with (ROOT/(name+'.log')).open('wb') as f:
   p=subprocess.Popen([sys.executable,'-m',*args],cwd=str(ROOT),stdout=f,stderr=subprocess.STDOUT);write(ROOT/(name+'.process.json'),dict(pid=p.pid,controller=os.getpid(),unix=time.time()));rc=p.wait()
  (ROOT/(name+'.exit')).write_text(str(rc)+'\n');assert rc==0,(name,rc)
 gate=json.loads((OUT/'A_GATE.json').read_text());write(OUT/'RUN_STATUS.json',dict(status='A_PASSED_B_IMPLEMENTATION_ADMISSION_PENDING' if gate['selected'] else 'STOP_STATE_GATE',phase='A_FINISHED',gate_sha256=sha(OUT/'A_GATE.json'),unix=time.time(),controller=os.getpid()))
 (ROOT/'controller.exit').write_text('0\n')
if __name__=='__main__':
 try:main()
 except BaseException:
  write(OUT/'FAILURE.json',dict(status='ENGINEERING_FAILURE',traceback=traceback.format_exc(),unix=time.time()));(ROOT/'controller.exit').write_text('1\n');raise
