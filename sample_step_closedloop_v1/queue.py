import os,sys,time,signal,subprocess,traceback,shutil,zipfile
from .common import *
def stage(label,args,budget):
 start=time.time();p=read(OUT/'PROTOCOL_FROZEN.json');r=p['resources']
 assert time.time()-float(os.environ['SAMPLE_QP_QUEUE_START'])<r['wall_hours']*3600
 with (OUT/(label+'.log')).open('w') as log:
  child=subprocess.Popen([sys.executable]+args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write(OUT/'QUEUE_STATUS.json',dict(status='RUNNING',stage=label,pid=os.getpid(),child_pid=child.pid,time=time.time()))
  try:
   while child.poll() is None:
    assert time.time()-start<budget and time.time()-float(os.environ['SAMPLE_QP_QUEUE_START'])<r['wall_hours']*3600,'Time watchdog'
    from functional_geometry_diagnostic_v1.runtime_guard import memory,size
    m=memory(child.pid);assert m['rss']<=r['host_rss_GiB']*2**30 and m['available']>=r['host_available_min_GiB']*2**30,'Memory watchdog'
    assert shutil.disk_usage(ROOT).free>=r['data_free_min_GiB']*2**30 and shutil.disk_usage('/root').free>=r['root_free_min_GiB']*2**30,'Disk watchdog'
    assert size(OUT)<r['new_reports_max_GiB']*2**30 and size(SCRATCH)<r['new_states_max_GiB']*2**30,'Output budget'
    time.sleep(5)
  except BaseException:
   if child.poll() is None:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(15)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   raise
  assert child.returncode==0,(label,child.returncode)

def main():
 verify();p=read(OUT/'PROTOCOL_FROZEN.json');os.environ['SAMPLE_QP_QUEUE_START']=str(time.time());SCRATCH.mkdir(exist_ok=True)
 write(OUT/'QUEUE_STARTED.json',dict(pid=os.getpid(),time=time.time(),seeds=p['seeds'],training_steps=300,trials=1350))
 stage('cpu_tests',['-m','sample_step_closedloop_v1.tests'],600)
 stage('solver_tests',['-m','unittest','sample_task_qp_v1.tests','-v'],600)
 stage('train_engineering',['-m','sample_step_closedloop_v1.train','engineering'],7200)
 assert read(OUT/'TRAIN_PREFLIGHT.json')['status']=='PASS'
 for tag,extra in [('trace_off',['no_trace']),('trace_on',[])]:
  stage('engineering_'+tag,['-m','sample_step_closedloop_v1.evaluate','0',tag]+extra,7200)
 off=read(OUT/'engineering/trace_off/ROUNDS.json');on=read(OUT/'engineering/trace_on/ROUNDS.json');assert len(off)==len(on)==5
 for x,y in zip(off,on):
  other=dict(y);other.pop('trace_file_sha256');assert x==other,('Identical-model evaluation parity failed',x['round'])
 write(OUT/'PREFLIGHT_COMPLETE.json',dict(status='PASS',origins=3,arms=3,first_step_exact=True,serialize_resume_step2_exact=True,identical_model_50trial_trace_parity=True,engineering_trials_excluded=100))
 (OUT/'PRECHECK_REPORT.md').write_text('PASS: CPU unit/solver tests,3origins x3arms first-step historical state parity, serialized step2 resume parity, identical-model50trial trace on/off parity. Engineering100trials excluded. Frozen300step3seed3arm/1350trial queue follows without outcome-based tuning.\n')
 write(OUT/'FORMAL_STARTED.json',dict(time=time.time(),pid=os.getpid(),jobs=9,trials=1350))
 for i,name in enumerate(p['points']):
  for a in p['train_orders'][i]:stage('train_'+name+'_'+a,['-m','sample_step_closedloop_v1.train',name,a],p['resources']['train_job_hours']*3600)
 for i,c in enumerate(p['cells']):stage('eval_'+str(i).zfill(2),['-m','sample_step_closedloop_v1.evaluate',str(i)],p['resources']['eval_cell_hours']*3600)
 stage('analysis',['-m','sample_step_closedloop_v1.analyze'],3600)
 stage('independent_audit',['-m','sample_step_closedloop_v1.audit'],3600)
 write(OUT/'QUEUE_STATUS.json',dict(status='COMPLETE',time=time.time(),seconds=time.time()-float(os.environ['SAMPLE_QP_QUEUE_START']),jobs=9,trials=1350))
 files=[f for f in OUT.rglob('*') if f.is_file()]+[ROOT/n for n in read(OUT/'FROZEN_INPUTS.json') if Path(n).suffix in ['.py','.yaml']]
 write(OUT/'DELIVERY_MANIFEST.json',{f.relative_to(ROOT).as_posix():sha(f) for f in files})
 with zipfile.ZipFile('/root/sample_step_closedloop_v1_delivery.zip','w',zipfile.ZIP_DEFLATED) as z:
  for f in files+[OUT/'DELIVERY_MANIFEST.json']:z.write(f,f.relative_to(ROOT).as_posix())
if __name__=='__main__':
 try:main()
 except BaseException:write(OUT/'QUEUE_STATUS.json',dict(status='FAILED_STOP',time=time.time(),pid=os.getpid(),error=traceback.format_exc()));raise
