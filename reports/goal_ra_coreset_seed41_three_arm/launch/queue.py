"""One finite baseline-pair then RA queue. No automatic retry."""
from pathlib import Path
import subprocess,sys,json,time,traceback,os,shlex
Q=Path(__file__).resolve().parent
PY='/data/mlr_ifa/env/bin/python'
def main():
 assert not (Q/'queue.started').exists()
 (Q/'queue.started').write_text(str(time.time()));(Q/'queue.pid').write_text(str(os.getpid()));code=1
 env=Q/'baseline/source/coreset_exp/env.sh'
 def baseline(command):return ['bash','-c','source '+shlex.quote(str(env))+' && '+command]
 phases=[('baseline_software',baseline(PY+' -m unittest coreset_exp.test_protocol coreset_exp.test_selector -v')),
 ('baseline_configure',baseline(PY+' -m coreset_exp.configure')),
 ('baseline_suite_audit',baseline(PY+' '+str(Q/'audit_suite.py'))),
 ('baseline_preflight',baseline(PY+' -m torch.distributed.run --standalone --nproc_per_node=1 -m coreset_exp.preflight')),
 ('baseline_supervise',['bash',str(Q/'baseline/source/coreset_exp/supervise.sh')]),
 ('ra_admission',['bash',str(Q/'ra/admission.sh')]),('ra_supervise',['bash',str(Q/'ra/supervise.sh')]),
 ('aggregate',[PY,str(Q/'aggregate_seed41.py')])]
 try:
  for phase,cmd in phases:
   assert not (Q/(phase+'.exit')).exists()
   with (Q/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(phase=phase,command=cmd,start=time.time()))+'\n')
   with (Q/(phase+'.log')).open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
   (Q/(phase+'.exit')).write_text(str(r.returncode)+'\n')
   with (Q/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(phase=phase,exit_code=r.returncode,end=time.time()))+'\n')
   assert r.returncode==0,(phase,r.returncode)
  code=0
 except BaseException:(Q/'QUEUE_FAILURE.json').write_text(json.dumps(dict(time=time.time(),traceback=traceback.format_exc()),indent=2))
 finally:(Q/'queue.exit').write_text(str(code)+'\n')
 return code
if __name__=='__main__':raise SystemExit(main())
