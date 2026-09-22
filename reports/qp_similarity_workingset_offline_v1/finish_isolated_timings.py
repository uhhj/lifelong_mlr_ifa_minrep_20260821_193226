"""Fixed administrative timing repair; preserves all original measurements."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import json,time,subprocess,sys,shutil,traceback
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
def read(p):return json.loads(p.read_text())
def write(p,z):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(z,indent=2,allow_nan=False));os.replace(t,p)
try:
 write(P/'COMPLETION_PIPELINE.json',dict(status='WAITING_PRIMARY_AND_SENSITIVITY',pid=os.getpid(),training_active=False))
 pid=int((P/'RUN.pid').read_text())
 while True:
  try:os.kill(pid,0)
  except ProcessLookupError:break
  time.sleep(5)
 assert read(P/'STATUS.json')['status']=='COMPLETE'
 note=read(P/'TIMING_ISOLATION_NOTE.json');archive=P/'archive_preserved_timing';archive.mkdir(exist_ok=False)
 shutil.copy2(P/'SUMMARY.json',archive/'SUMMARY_BEFORE_TIMING_RECHECK.json')
 for rel in note['cells']:
  source=(P/rel).resolve();target=archive/rel
  assert P in source.parents and (source/'COMPLETE.json').exists() and not target.exists()
  target.parent.mkdir(parents=True,exist_ok=True);source.rename(target)
 write(P/'COMPLETION_PIPELINE.json',dict(status='REPEATING_ALL_ARMS_IN_FIXED_EXCLUDED_CELLS',cells=note['cells'],training_active=False))
 with (P/'TIMING_RECHECK.log').open('w') as log:
  child=subprocess.run(['taskset','-c','0',sys.executable,'-u','-m','qp_similarity_workingset_offline_v1.run'],cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT)
 assert child.returncode==0,'Timing recheck failed'
 import numpy as np
 compared=[]
 for rel in note['cells']:
  for old in (archive/rel).glob('*/candidate.npz'):
   new=P/old.relative_to(archive)
   with np.load(old) as a,np.load(new) as b:
    ok=all(np.array_equal(a[k],b[k]) for k in a.files)
   compared.append(dict(file=str(new),identical=ok))
 assert all(z['identical'] for z in compared)
 write(P/'TIMING_RECHECK_AUDIT.json',dict(status='PASS',cells=len(note['cells']),candidates=compared,all_identical=True,
  policy='All named cells, all methods, all10 repetitions replaced by fresh isolated measurement, irrespective of speed. Originals preserved.',training_active=False))
 with (P/'INDEPENDENT_AUDIT.log').open('w') as log:
  audit=subprocess.run([sys.executable,str(P/'independent_audit.py')],cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT)
 assert audit.returncode==0
 assert read(P/'INDEPENDENT_AUDIT.json')['status']=='PASS','Independent audit failed'
 subprocess.run([sys.executable,str(P/'make_readable_report.py')],check=True)
 write(P/'COMPLETION_PIPELINE.json',dict(status='COMPLETE',time=time.time(),training_active=False,formal_solver_modified=False,model_writeback=False,GPU_required=False))
except BaseException:
 write(P/'COMPLETION_PIPELINE.json',dict(status='STOPPED',error=traceback.format_exc(),training_active=False))
 raise
