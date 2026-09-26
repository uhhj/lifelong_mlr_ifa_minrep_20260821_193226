"""Finite Stage A queue, technical repeats then freeze, resumable per unit."""
import fcntl
import subprocess
import sys
import time
import shutil
from .common import *

def freeze_registry():
    features=set(f for v in PACKAGES.values() for f in v)|{'b_target','b_source','b_progress','base_progress1','base_progress5','base_progress20'}
    bounds={'visual':[-1,1],'language':[-1,1],'gn_overlap':[0,1],'gradient_cosine':[-1,1],'coverage':[0,1],'risk_positive':[0,1],'mmd':[0,2]}
    noise={f:0. for f in features};counts={f:0 for f in features}
    for k,s,t in SMOKE:
        for phase,name in [('static','SIGNALS.json'),('probes','MEASUREMENTS.json')]:
            a=read(OUT/phase/unit(k,s,t)/name)['scores'];b=read(OUT/(phase+'_repeat')/unit(k,s,t)/name)['scores']
            for f in features:
                if a.get(f) is not None and b.get(f) is not None:
                    noise[f]=max(noise[f],abs(a[f]-b[f]));counts[f]+=1
    registry={f:dict(bound=bounds.get(f),scale_floor=max(1e-8,10*noise[f]),constant_floor=max(1e-8,10*noise[f]),repeat_abs_difference=noise[f],technical_observations=counts[f]) for f in sorted(features)}
    freeze(OUT/'SIGNAL_REGISTRY.json',dict(status='FROZEN',features=registry,packages=PACKAGES,technical_units=SMOKE,missing_rule='train-fold mean imputation, raw null retained; fully unavailable package excluded from finalist eligibility',primary='equal D1/D2 result; source/target losses trajectory-equal',new_C0_candidates=['S02','S03','S05','S06','S07','S08','S09','S10','S11'],formal_training_resumed=False))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'stage_A.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def execute(module,args,tag):
        if shutil.disk_usage(ROOT).free<2*1024**3:raise RuntimeError('STORAGE_RESERVE')
        prior_probe=any((OUT/'probes').glob('*/COMPLETE.json'))
        write(OUT/'STATUS.json',dict(stage=tag,status='RUNNING',started=time.time(),formal_training_resumed=False,diagnostic_training_performed=prior_probe or 'probe' in tag))
        with (OUT/(tag+'.log')).open('a') as f:
            run=subprocess.run([sys.executable,'-u','-m',module]+args,cwd=str(ROOT),stdout=f,stderr=subprocess.STDOUT)
        if run.returncode:raise RuntimeError('Child failed: '+tag)
    try:
        execute('unittest',['task_relation_signal_screen_v2.tests','-v'],'UNIT_TESTS')
        write(OUT/'TECHNICAL_TESTS.json',dict(status='CPU_PASS_GPU_SMOKE_PENDING',unit_tests=8,unit_test_log_sha256=sha(OUT/'UNIT_TESTS.log'),formal_training_resumed=False))
        for task in TASKS:
            execute('task_relation_signal_screen_v2.matching_cache',['--task',str(task)],'MATCHING_CACHE_t%d'%task)
        for triple in SMOKE+[q for q in PAIRS if q not in SMOKE]:
            for kind,phase in [('static','static'),('probe','probes')]:
                for repeat in ([False,True] if triple in SMOKE else [False]):
                    folder=phase+('_repeat' if repeat else '')
                    if (OUT/folder/unit(*triple)/'COMPLETE.json').exists():continue
                    args=['--kind',kind,'--seed',str(triple[0]),'--source',str(triple[1]),'--target',str(triple[2])]+(['--repeat'] if repeat else [])
                    execute('task_relation_signal_screen_v2.measure',args,folder+'_'+unit(*triple))
            if triple==SMOKE[-1]:
                freeze_registry()
                write(OUT/'TECHNICAL_TESTS.json',dict(status='PASS',unit_tests=8,technical_units=SMOKE,repeat_runs=True,registry_sha256=sha(OUT/'SIGNAL_REGISTRY.json'),checks=['model_optimizer_scheduler_rng_preservation','historical_prefix_batch_lr_loss_and_zero_step_parity','actual_effective_updates','role_restricted_data','near_constant_transform'],formal_training_resumed=False))
        execute('task_relation_signal_screen_v2.analyze_dev',[],'ANALYZE_DEV')
        execute('task_relation_signal_screen_v2.development_checks',[],'DEVELOPMENT_CHECKS')
        execute('task_relation_signal_screen_v2.report_dev',[],'REPORT_DEV')
        execute('task_relation_signal_screen_v2.audit_stage_a',[],'AUDIT_STAGE_A')
        write(OUT/'STATUS.json',dict(status='STAGE_A_COMPLETE',static_complete=36,probe_complete=36,finalists_frozen=True,stage_B_started=False,formal_training_resumed=False,diagnostic_training_performed=True))
    except Exception as e:
        prev=read(OUT/'STATUS.json');write(OUT/'FAILED_STOP.json',dict(error=str(e),previous_status=prev,time=time.time(),formal_training_resumed=False));raise

if __name__=='__main__':main()
