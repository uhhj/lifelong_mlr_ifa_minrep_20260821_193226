import os,sys,json,time,fcntl,subprocess,traceback
from pbd.common import ROOT,OUT,write,verify,capacity,sha,js
def main():
 lock=open('/data/mlr_ifa/physical_effect_probe.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 assert not (OUT/'EVENTS.jsonl').exists(),'No implicit restart';verify();capacity(int(3.3*1024**3))
 def run(name,args):
  verify();write(OUT/'RUN_STATUS.json',dict(status='PARTIAL',phase=name,controller=os.getpid(),unix=time.time()))
  with (ROOT/(name+'.log')).open('wb') as f:
   p=subprocess.Popen([sys.executable,'-m',*args],cwd=str(ROOT),stdout=f,stderr=subprocess.STDOUT);write(ROOT/(name+'.process.json'),dict(pid=p.pid,controller=os.getpid(),unix=time.time()));rc=p.wait()
  (ROOT/(name+'.exit')).write_text(str(rc)+'\n');assert rc==0,(name,rc)
 run('readout_admission',['pbd.admit','readout']);run('readout',['pbd.train','readout']);run('readout_audit',['pbd.audit','readout'])
 gate=js(OUT/'READOUT_GATE.json')
 if not gate['selected']:
  run('patch_export',['pbd.export_patch']);run('spatial_admission',['pbd.admit','spatial']);run('spatial',['pbd.train','spatial']);run('spatial_audit',['pbd.audit','spatial']);gate=js(OUT/'SPATIAL_GATE.json')
 write(OUT/'A_GATE.json',gate)
 write(OUT/'RUN_STATUS.json',dict(status='A_PASSED_PENDING_LOCAL_AUDIT_AND_B_IMPLEMENTATION' if gate['selected'] else 'A_FAILED_PENDING_LOCAL_AUDIT_GENERALIZATION_PROTOCOL',phase='A_FINISHED',gate_sha256=sha(OUT/'A_GATE.json'),unix=time.time(),controller=os.getpid()))
 (ROOT/'controller.exit').write_text('0\n')
if __name__=='__main__':
 try:main()
 except BaseException:
  write(OUT/'FAILURE.json',dict(status='ENGINEERING_FAILURE',traceback=traceback.format_exc(),unix=time.time()));(ROOT/'controller.exit').write_text('1\n');raise
