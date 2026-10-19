"""Finite amended stage1 execution. All old exits/logs preserved, no blind retries."""
import os,sys,json,time,subprocess,shutil,hashlib,traceback
from pathlib import Path
N=Path(__file__).resolve().parent;Q=Path('/data/mlr_ifa/experiment_workspaces/RA_OBJECT_conditional_20261019')
PY='/data/mlr_ifa/env/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 seed=int(sys.argv[1]);assert seed in (41,42,43);W=Q/('seed%d'%seed);E=N/'stage1'/str(seed);E.mkdir(parents=True,exist_ok=True)
 O=Path('/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed%d_20261019'%seed)
 assert not (E/'driver.started').exists();write(E/'driver.started',dict(pid=os.getpid(),time=time.time()))
 def phase(name,cmd):
  assert not (E/(name+'.exit')).exists()
  with (E/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(phase=name,command=cmd,time=time.time()))+'\n')
  with (E/(name+'.log')).open('x') as f:r=subprocess.run(['bash','-c','source '+str(W/'env.sh')+' && '+cmd],stdout=f,stderr=subprocess.STDOUT)
  (E/(name+'.exit')).write_text(str(r.returncode)+'\n');assert r.returncode==0,(seed,name,r.returncode)
 rc=1
 try:
  if seed==41:
   assert (O/'pipeline.exit').read_text().strip()=='1'
   assert not (Path('/proc')/(O/'pipeline.pid').read_text().strip()).exists()
   assert json.loads((E/'PROTOCOL_EQUIVALENCE.json').read_text())['status']=='PASS_NO_PROTOCOL_DEVIATION_FOUND'
   paths=['train.log','train.exit','pipeline.log','pipeline.exit','supervisor.exit','PIPELINE_FAILURE.json','TASK6_TENSOR_PARITY.json','RUN_STATUS.json','ra_coreset/epochs.json','ra_coreset/models/mlrifa_6_model_ep99.pth','ra_coreset/recovery/latest.pth']
   preserved={}
   for name in paths:
    p=O/name
    if not p.exists():continue
    dst=E/'original_failure'/name;dst.parent.mkdir(parents=True,exist_ok=True);assert not dst.exists();shutil.copy2(p,dst);assert sha(p)==sha(dst)
    preserved[name]=sha(dst)
   write(E/'PRESERVED_FAILURE.json',dict(files=preserved,original_optimizer_calls=15700,committed_steps=15543,replay_uncommitted_steps=157,formal_target_steps=105500,expected_physical_total_steps=105657))
  else:
   phase('admission','bash '+str(W/'admission.sh'))
   phase('protocol_audit',PY+' '+str(N/'protocol_audit.py'))
  phase('train',PY+' -m torch.distributed.run --standalone --nproc_per_node=1 '+str(N/'ra_adapter.py')+' train ra_coreset'+(' --resume' if seed==41 else ''))
  phase('evaluate',PY+' -m coreset_exp.evaluate ra_coreset')
  phase('audit',PY+' '+str(N/'ra_audit.py'))
  a=json.loads((O/'AUDIT.json').read_text());assert a['status']=='PASS' and a['epochs']==400 and a['optimizer_steps']==105500 and a['formal_records']==200
  assert len(a['checkpoints'])==4 and all(x['loaded_keys']==100 for x in a['checkpoints'])
  assert json.loads((E/'TASK6_POLICY_ACCEPTANCE.json').read_text())['status'] in ['PASS_EQUAL','DISCLOSED_CROSS_HARDWARE_DIFFERENCE']
  # All subprocess.run children have exited. Original failures are deliberately not rewritten.
  write(E/'ENGINEERING_ACCEPTANCE.json',dict(status='PASS',seed=seed,formal_steps=105500,physical_steps=105657 if seed==41 else 105500,formal_episodes=200,vector_attempts=20,original_pipeline_exit=1 if seed==41 else None,amended_phases_exit=0,task6_parity_policy=json.loads((E/'TASK6_POLICY_ACCEPTANCE.json').read_text()),time=time.time()))
  write(O/'AUTONOMOUS_EXIT_ACCEPTANCE.json',json.loads((E/'ENGINEERING_ACCEPTANCE.json').read_text()));rc=0
 except BaseException:write(E/'FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
 finally:(E/'driver.exit').write_text(str(rc)+'\n')
if __name__=='__main__':main()
