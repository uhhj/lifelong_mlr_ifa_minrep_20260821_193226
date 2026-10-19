"""Detached parent waits for finite controller then performs actual exit audit."""
import os,subprocess,time,traceback
from flow_support import N,PY,write,verify_controller
def main():
 assert not (N/'wrapper.started').exists();verify_controller();write(N/'wrapper.started',dict(pid=os.getpid(),time=time.time()))
 with (N/'controller.log').open('x') as log:p=subprocess.Popen([PY,'-u',str(N/'controller.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 code=p.wait();(N/'wrapper.exit').write_text(str(code)+'\n')
 if code:return code
 for name in ['final_audit','package_light']:
  with (N/(name+'.log')).open('x') as log:r=subprocess.run([PY,str(N/(name+'.py'))],stdout=log,stderr=subprocess.STDOUT)
  (N/(name+'.exit')).write_text(str(r.returncode)+'\n')
  if r.returncode:
   write(N/'STATUS.json',dict(status='BLOCKED_READONLY_FINALIZATION',phase=name,exit=r.returncode,time=time.time()));return r.returncode
 return 0
if __name__=='__main__':
 rc=1
 try:rc=main()
 finally:(N/'supervisor.exit').write_text(str(rc)+'\n')
 raise SystemExit(rc)
