"""Finite 41 then 42 then 43 queue. Never retries or starts old baseline drivers."""
from pathlib import Path
import subprocess,sys,json,time,traceback,os
Q=Path(__file__).resolve().parent
def main():
 assert not (Q/'queue.started').exists();(Q/'queue.started').write_text(str(time.time()));(Q/'queue.pid').write_text(str(os.getpid()));code=1
 try:
  for seed in (41,42,43):
   for phase,cmd in [('admission',['bash',str(Q/('seed%d'%seed)/'admission.sh')]),('supervise',['bash',str(Q/('seed%d'%seed)/'supervise.sh')])]:
    tag='seed%d_%s'%(seed,phase);assert not (Q/(tag+'.exit')).exists();t=time.time()
    with (Q/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(seed=seed,phase=phase,command=cmd,time=t))+'\n')
    with (Q/(tag+'.log')).open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    (Q/(tag+'.exit')).write_text(str(r.returncode)+'\n');assert r.returncode==0,(tag,r.returncode)
  with (Q/'aggregate.log').open('x') as log:r=subprocess.run([sys.executable,str(Q/'aggregate_stage1.py')],stdout=log,stderr=subprocess.STDOUT)
  (Q/'aggregate.exit').write_text(str(r.returncode)+'\n');assert r.returncode==0
  code=0
 except BaseException:(Q/'QUEUE_FAILURE.json').write_text(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()),indent=2))
 finally:(Q/'queue.exit').write_text(str(code)+'\n')
 return code
if __name__=='__main__':raise SystemExit(main())
