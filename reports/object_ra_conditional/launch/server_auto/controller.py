"""Server-side finite conditional study; engineering errors STOP, never choose fallback."""
import os,sys,time,json,traceback,fcntl
from pathlib import Path
from collections import Counter
from fractions import Fraction as F
from flow_support import N,PY,js,write,phase,verify_controller,no_workers,alive,sha
from flow_metrics import metrics,gate,pack,stats
from prepare_zr import prepare
Q=Path('/data/mlr_ifa/experiment_workspaces/RA_OBJECT_conditional_20261019')
def finish_ra(seed,readonly_recovery=False):
 E=N/'stage1'/str(seed);O=Path('/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed%d_20261019'%seed)
 for p in ['train','evaluate']:assert (E/(p+'.exit')).read_text().strip()=='0'
 if readonly_recovery:assert (E/'audit_python38_fix.exit').read_text().strip()=='0'
 else:assert (E/'audit.exit').read_text().strip()=='0'
 a=js(O/'AUDIT.json');assert a['status']=='PASS' and a['epochs']==400 and a['optimizer_steps']==105500 and a['formal_records']==200
 assert len(a['checkpoints'])==4 and all(v['loaded_keys']==100 for v in a['checkpoints'])
 no_workers();parity=js(E/'TASK6_POLICY_ACCEPTANCE.json');assert parity['status'] in ['PASS_EQUAL','DISCLOSED_CROSS_HARDWARE_DIFFERENCE']
 if readonly_recovery:
  proof=dict(status='PASS',seed=seed,formal_steps=105500,physical_steps=105657,formal_episodes=200,vector_attempts=20,original_pipeline_exit=1,amended_driver_exit=1,read_only_audit_recovery_exit=0,no_scientific_retry=True,task6_parity_policy=parity,time=time.time())
  write(E/'ENGINEERING_ACCEPTANCE.json',proof);write(O/'AUTONOMOUS_EXIT_ACCEPTANCE.json',proof)
 assert js(E/'ENGINEERING_ACCEPTANCE.json')['status']=='PASS'
def run_stage1():
 # Seed41 was already explicitly resumed and detached before this controller was installed.
 E=N/'stage1/41';launch=js(N/'recovery_launch.json')
 while not (E/'driver.exit').exists():
  assert alive(launch['pid']),'Recovery driver vanished without exit evidence'
  time.sleep(30)
 if (E/'driver.exit').read_text().strip()!='0':
  # This exact Python3.8 read-only adapter portability defect was identified during deployment.
  # No training/evaluation is retried. Other failures must stop.
  assert (E/'train.exit').read_text().strip()=='0' and (E/'evaluate.exit').read_text().strip()=='0'
  assert (E/'audit.exit').read_text().strip()=='1'
  log=(E/'audit.log').read_text();assert "module 'ast' has no attribute 'unparse'" in log
  assert not alive(launch['pid']);no_workers()
  write(E/'READONLY_FIX_ADDENDUM.json',dict(reason='Python3.8 lacks ast.unparse; only assertion AST matching changes to ast.dump equality',original_exit=1,training_and_evaluation_retry=False,corrected_adapter_sha256=sha(N/'ra_audit.py'),time=time.time()))
  phase(E,'audit_python38_fix',[PY,str(N/'ra_audit.py')],Q/'seed41/env.sh')
  finish_ra(41,True)
 else:finish_ra(41)
 for seed in [42,43]:
  phase(N/'stage1'/str(seed),'driver_v2',[PY,str(N/'ra_driver_v2.py'),str(seed)])
  finish_ra(seed)
 return aggregate_ra()
def aggregate_ra():
 results={};new=[];old=[];co=[];deltas=[];parents=js(Q/'PARENTS.json')
 for seed in [41,42,43]:
  O=Path('/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed%d_20261019'%seed);P=Path(parents[str(seed)]['root']);E=N/'stage1'/str(seed)
  assert js(E/'ENGINEERING_ACCEPTANCE.json')['status']=='PASS';audit=js(O/'AUDIT.json')
  epochs=js(O/'ra_coreset/epochs.json');assert len(epochs)==400 and sum(x['batches'] for x in epochs)==105500 and {(x['task'],x['epoch']) for x in epochs}=={(t,e) for t in range(6,10) for e in range(100)}
  events=Counter(json.loads(l)['event'] for f in (O/'ra_coreset/evaluation').glob('attempts*.jsonl') for l in f.read_text().splitlines())
  assert events==Counter(vector_attempt_start=20,vector_attempt_complete=20,environment_construct_start=10,environment_construct_complete=10,environment_closed=10)
  data={}
  for arm,root,target in [('random',P,old),('coreset',P,co),('ra_coreset',O,new)]:
   m,r,rows,hs=metrics(root/arm/'evaluation',seed,arm);target.append(m)
   assert pack(m)==audit['metrics'][arm]
   for x in rows.values():
    cp=root/arm/'models'/('mlrifa_%d_model_ep99.pth'%x['stage'])
    # Same phase hash for all trials; verify each real file once below.
    assert x['checkpoint_sha256']==rows[x['stage'],x['stage'],0]['checkpoint_sha256']
   for t in range(6,10):assert sha(root/arm/'models'/('mlrifa_%d_model_ep99.pth'%t))==rows[t,t,0]['checkpoint_sha256']
   data[arm]=dict(metrics=pack(m),matrix=[[float(r[s,t]*100) if t<=s else None for t in range(6,10)] for s in range(6,10)],episode_hashes=hs)
  delta=new[-1]['AUC']-old[-1]['AUC'];deltas.append(delta);results[str(seed)]=data
 out=dict(status='PASS',stage=1,seed_results=results,statistics=dict(random=stats(old),coreset=stats(co),ra_coreset=stats(new)),gate=gate(deltas,True),comparator='Random_IFA',epochs=1200,logical_optimizer_steps=316500,physical_optimizer_steps=316657,new_records=600,vector_attempts=60,strict_checkpoints=12,time=time.time())
 write(N/'STAGE1_ACCEPTANCE.json',out);return out
def run_zr(branch,seed):
 verify_controller();no_workers();W,O,spec=prepare(branch,seed)
 E=N/'zr_ledger'/('%s_seed%d'%(branch,seed));env=W/'env.sh'
 write(N/'STATUS.json',dict(status='RUNNING_ZRV2',branch=branch,seed=seed,output=str(O),time=time.time()))
 phase(E,'software',[PY,'-m','unittest','discover','-s',str(W/'attachment'),'-p','test_reference.py','-v'],env)
 phase(E,'admission',[PY,str(W/'admission.py')],env)
 phase(E,'integration',[PY,'-m','zr2_exp.integration'],env)
 phase(E,'cost_isolation',[PY,'-m','zr2_exp.cost_and_isolation'],env)
 assert js(O/'INTEGRATION.json')['optimizer_calls']==4 and js(O/'INTEGRATION.json')['environment_steps']==0
 phase(E,'freeze',[PY,'-m','zr2_exp.freeze'],env)
 phase(E,'pipeline',[PY,'-u',str(W/'pipeline.py')],env)
 phase(E,'independent_exit_audit',[PY,str(N/'zr_finish.py'),str(W)])
 return spec
def group_zr(branch,seeds):
 specs=[run_zr(branch,s) for s in seeds];ns=[];ps=[];results={}
 for spec in specs:
  O=Path(spec['output']);a=js(O/'PROCESS_EXIT_ACCEPTANCE.json');assert a['status']=='PASS'
  m,r,rows,h=metrics(O/'evaluation',spec['seed'],spec['arm']);b,br,_,_=metrics(Path(spec['parent'])/spec['parent_arm']/'evaluation',spec['seed'],spec['parent_arm'])
  assert pack(m)==a['new'] and pack(b)==a['baseline'];ns.append(m);ps.append(b)
  results[str(spec['seed'])]=dict(parent=pack(b),new=pack(m),delta=pack({k:m[k]-b[k] for k in m}),matrix=[[float(r[s,t]*100) if t<=s else None for t in range(6,10)] for s in range(6,10)],parent_matrix=[[float(br[s,t]*100) if t<=s else None for t in range(6,10)] for s in range(6,10)],episodes=h,output=str(O),cost=js(O/'AUDIT.json')['cost'])
 out=dict(status='PASS',branch=branch,seed_results=results,statistics=dict(parent=stats(ps),new=stats(ns)),formal_records=200*len(specs),calibration_epochs=40*len(specs),optimizer_steps=sum(x['optimizer_steps'] for x in specs),strict_checkpoints=4*len(specs),vector_attempts=20*len(specs),time=time.time())
 if branch=='OBJECT_RA':out['gate']=gate([n['AUC']-p['AUC'] for n,p in zip(ns,ps)],True)
 write(N/(branch+'_ACCEPTANCE.json'),out);return out
def main():
 lock=(N/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (N/'controller.started').exists();verify_controller()
 write(N/'controller.started',dict(pid=os.getpid(),time=time.time()));rc=1
 try:
  write(N/'STATUS.json',dict(status='WAITING_OR_RUNNING_STAGE1',time=time.time()))
  stage1=run_stage1();executed=['OBJECT_RA_IFA'];results={'stage1':stage1}
  if stage1['gate']['passed']:
   stage2=group_zr('OBJECT_RA',[41,42,43]);results['stage2']=stage2;executed.append('OBJECT_RA_ZRv2_IFA')
   if stage2['gate']['passed']:results['stage3']=group_zr('GOAL_RA',[41,42,43]);executed.append('GOAL_RA_ZRv2_IFA')
   else:results['stop_reason']='STAGE2_METRIC_GATE_NOT_MET'
  else:results['fallback']=group_zr('GOAL_Coreset',[41,42,43,44]);executed.append('GOAL_Coreset_ZRv2_IFA')
  verify_controller();no_workers()
  write(N/'EXPERIMENT_RESULTS.json',dict(status='SERVER_EVIDENCE_READY',executed=executed,results=results,time=time.time()))
  write(N/'STATUS.json',dict(status='AWAITING_SERVER_EXIT_AUDIT',executed=executed,time=time.time()));rc=0
 except BaseException:
  failure=dict(status='BLOCKED',time=time.time(),traceback=traceback.format_exc(),automatic_retry=False,performance_fallback_on_engineering_failure=False)
  write(N/'FAILURE.json',failure);write(N/'STATUS.json',failure);(N/'BLOCKED.md').write_text('# BLOCKED\n\nAn engineering assertion or phase failed. All attempts and original logs are retained. No automatic retry, new hyperparameters or performance fallback.\n\n```\n'+failure['traceback']+'\n```\n');raise
 finally:(N/'controller.exit').write_text(str(rc)+'\n')
if __name__=='__main__':main()
