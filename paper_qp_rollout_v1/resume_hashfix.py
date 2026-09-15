"""One documented technical recovery; no scientific protocol change."""
import os,time,sys,traceback,zipfile
from .common import *
from .queue import run_child
def run():
    verify();assert not list((OUT/'evaluation').glob('*/*/*/COMPLETE.json'))
    assert read(OUT/'RECOVERY_PREFLIGHT.json')['status']=='PASS'
    assert not (OUT/'HASHFIX_RESUME_STARTED.json').exists()
    start=time.time();write(OUT/'HASHFIX_RESUME_STARTED.json',{'pid':os.getpid(),'time':start,'reason':'Fix object-array pointer hash to recursive observation content hash; original engineering results preserved; no training or evaluation rule changed'})
    run_child('hashfix_unit_tests',['-m','paper_qp_rollout_v1.tests'],start)
    run_child('hashfix_trace_off',['-m','paper_qp_rollout_v1.evaluate','0','trace_off_hashfix','no_trace'],start)
    run_child('hashfix_trace_on',['-m','paper_qp_rollout_v1.evaluate','0','trace_on_hashfix'],start)
    off=read(OUT/'engineering/trace_off_hashfix/ROUNDS.json');on=read(OUT/'engineering/trace_on_hashfix/ROUNDS.json')
    assert len(off)==len(on)==2
    for a,b in zip(off,on):
        bb=dict(b);bb.pop('trace_file_sha256');assert a==bb,('corrected full-content parity failed',a,b)
    # The instrumentation fix must also preserve original complete rollout behavior.
    for tag,rr in [('trace_off',off),('trace_on',on)]:
        original=read(OUT/'engineering'/tag/'ROUNDS.json')
        for a,b in zip(original,rr):
            for k in ['init_sha256','policy_seed','policy_rng_before','policy_rng_after','count','success','first','action_sha256','done_sha256']:assert a[k]==b[k],('hashfix affected rollout',tag,k)
    write(OUT/'PREFLIGHT_COMPLETE.json',{'status':'PASS','recovery_origins':6,'recovered_witnesses':36,'duplicate_complete_rollout_trials':20,'identical':['actions','dones','initial observation CONTENT and states','round RNG before/after'],'passive_logging_parity':True,'original_rollout_actions_preserved_after_fix':True,'engineering_trials_excluded':80,'technical_amendment':'amendments/observation_hash_fix_v1'})
    (OUT/'PRECHECK_REPORT.md').write_text('PASS. Six original origins reproduce both arms at1/5/20 bitwise. Complete20trial duplicated evaluation has identical observation contents,actions,dones,RNG. Original pointer-hash preflight failure is preserved; content-hash instrumentation fix changes no rollout behavior. All80engineering trials excluded. Main180cells/3600trials unchanged.\n')
    run_child('recover_all36',['-m','paper_qp_rollout_v1.recover','main'],start)
    write(OUT/'FORMAL_STARTED.json',{'time':time.time(),'pid':os.getpid(),'cells':180,'trials':3600})
    for i,c in enumerate(read(OUT/'PROTOCOL_FROZEN.json')['cells']):run_child('cell_'+str(i).zfill(3),['-m','paper_qp_rollout_v1.evaluate',str(i)],start)
    run_child('analysis',['-m','paper_qp_rollout_v1.analyze'],start)
    write(OUT/'QUEUE_STATUS.json',{'status':'COMPLETE','seconds_since_technical_resume':time.time()-start,'time':time.time(),'formal_completed_cells':180,'trials':3600})
    files=[x for x in OUT.rglob('*') if x.is_file()]+list((ROOT/'paper_qp_rollout_v1').glob('*.py'))
    write(OUT/'DELIVERY_MANIFEST.json',{x.relative_to(ROOT).as_posix():sha(x) for x in files})
    with zipfile.ZipFile('/root/paper_qp_rollout_v1_delivery.zip','w',zipfile.ZIP_DEFLATED) as z:
        for x in files+[OUT/'DELIVERY_MANIFEST.json']:z.write(x,x.relative_to(ROOT).as_posix())
if __name__=='__main__':
    try:run()
    except BaseException:write(OUT/'QUEUE_STATUS.json',{'status':'FAILED_STOP','time':time.time(),'traceback':traceback.format_exc()});raise
