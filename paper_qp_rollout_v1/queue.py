import os,sys,time,subprocess,signal,shutil,traceback
from .common import *
def run_child(stage,args,started):
    log=OUT/'logs'/(stage+'.log');log.parent.mkdir(exist_ok=True)
    with log.open('x') as f:
        child=subprocess.Popen([sys.executable]+args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        write(OUT/'QUEUE_STATUS.json',{'status':'RUNNING','stage':stage,'pid':os.getpid(),'child_pid':child.pid,'time':time.time(),'started':started,'formal_completed_cells':len(list((OUT/'evaluation').glob('*/*/*/COMPLETE.json')))})
        try:
            while child.poll() is None:
                assert time.time()-started<72*3600,'72hour watchdog'
                assert shutil.disk_usage('/root').free>=8*2**30,'root free guard'
                assert shutil.disk_usage(OUT).free>=2**30,'data free guard'
                time.sleep(5)
        except BaseException:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
            raise
    assert child.returncode==0,(stage,child.returncode)
def run():
    verify();p=read(OUT/'PROTOCOL_FROZEN.json');start=time.time()
    assert not (OUT/'QUEUE_STARTED.json').exists()
    write(OUT/'QUEUE_STARTED.json',{'pid':os.getpid(),'time':start,'protocol_sha256':sha(OUT/'PROTOCOL_FROZEN.json')})
    run_child('unit_tests',['-m','paper_qp_rollout_v1.tests'],start)
    run_child('recover_preflight6',['-m','paper_qp_rollout_v1.recover','preflight'],start)
    run_child('eval_trace_off',['-m','paper_qp_rollout_v1.evaluate','0','trace_off','no_trace'],start)
    run_child('eval_trace_on',['-m','paper_qp_rollout_v1.evaluate','0','trace_on'],start)
    off=read(OUT/'engineering/trace_off/ROUNDS.json');on=read(OUT/'engineering/trace_on/ROUNDS.json')
    for a,b in zip(off,on):
        bb=dict(b);bb.pop('trace_file_sha256');assert a==bb,('rollout duplicate trace parity failed',a,b)
    assert len(off)==len(on)==2
    write(OUT/'PREFLIGHT_COMPLETE.json',{'status':'PASS','recovery_origins':6,'recovered_witnesses':36,'duplicate_complete_rollout_trials':20,'identical':['actions','dones','initial observations and states','round RNG before/after'],'passive_logging_parity':True,'engineering_trials_excluded':40})
    (OUT/'PRECHECK_REPORT.md').write_text('Preflight PASS. All6registered origins reproduce both arms at1/5/20 including parameters,AdamW,scheduler,RNG and counters. Identical-model20-trial duplicate with trace retention on/off has identical actions,success flags,initial observations and RNG. No rollout effect was used to modify scientific rules. Formal queue includes36origins,180cells,3600trials.\n')
    run_child('recover_all36',['-m','paper_qp_rollout_v1.recover','main'],start)
    write(OUT/'FORMAL_STARTED.json',{'time':time.time(),'pid':os.getpid(),'cells':180,'trials':3600})
    for index,c in enumerate(p['cells']):run_child('cell_'+str(index).zfill(3),['-m','paper_qp_rollout_v1.evaluate',str(index)],start)
    run_child('analysis',['-m','paper_qp_rollout_v1.analyze'],start)
    write(OUT/'QUEUE_STATUS.json',{'status':'COMPLETE','seconds':time.time()-start,'time':time.time(),'formal_completed_cells':180,'trials':3600})
    files=[x for x in OUT.rglob('*') if x.is_file()]+list((ROOT/'paper_qp_rollout_v1').glob('*.py'))
    write(OUT/'DELIVERY_MANIFEST.json',{x.relative_to(ROOT).as_posix():sha(x) for x in files})
    import zipfile
    with zipfile.ZipFile('/root/paper_qp_rollout_v1_delivery.zip','w',zipfile.ZIP_DEFLATED) as z:
        for x in files+[OUT/'DELIVERY_MANIFEST.json']:z.write(x,x.relative_to(ROOT).as_posix())
if __name__=='__main__':
    try:run()
    except BaseException:
        write(OUT/'QUEUE_STATUS.json',{'status':'FAILED_STOP','time':time.time(),'traceback':traceback.format_exc()});raise
