"""Finite prospective confirmation queue; all technical pilot work counts."""
import fcntl
import subprocess
import sys
import time
from .common import *
from . import confirm_runtime as rt

def main(sources_only=False):
    rt.REPORT.mkdir(parents=True,exist_ok=True)
    lock=(rt.REPORT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def execute(name,args,tag):
        write(rt.REPORT/'QUEUE_STATUS.json',dict(status='RUNNING',stage=tag,pid=os.getpid(),updated_unix=time.time(),formal_training_resumed=False,diagnostic_training_performed=bool(list((rt.REPORT/'sources').glob('*/steps/*.json')))))
        with (rt.REPORT/(tag+'.log')).open('a') as log:
            result=subprocess.run([sys.executable,'-u','-m','task_relation_signal_screen_v2.'+name]+args,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('FAILED '+tag+'; inspect '+str(rt.REPORT/(tag+'.log')))
    def cache(k,t,role):
        if not (rt.REPORT/('CACHE_s%d_t%d_%s.json'%(k,t,role))).exists():execute('confirm_cache',['--seed',str(k),'--task',str(t),'--role',role],'CACHE_s%d_t%d_%s'%(k,t,role))
    def source(k,t):
        for role in ('A_train','D_signal'):cache(k,t,role)
        if not (rt.REPORT/'sources'/('s%d_t%d'%(k,t))/'COMPLETE.json').exists():execute('confirm_source',['--seed',str(k),'--task',str(t)],'SOURCE_s%d_t%d'%(k,t))
    def pair(k,s,t):
        args=['--seed',str(k),'--source',str(s),'--target',str(t)];sid=unit(k,s,t)
        if not (rt.REPORT/'signals'/sid/'C0_SEALED.json').exists():execute('confirm_signals',args,'C0_'+sid)
        if not (rt.REPORT/'transfers'/sid/'COMPLETE.json').exists():execute('confirm_target',args,'TARGET_'+sid)
        for task in (s,t):cache(k,task,'E_eval')
        if not (rt.REPORT/'offline'/sid/'COMPLETE.json').exists():execute('confirm_offline',args,'OFFLINE_'+sid)
    def behavior(indices):
        from .confirm_behavior import cells,directory
        cc=cells()
        for i in indices:
            if not (directory(cc[i])/'COMPLETE.json').exists():execute('confirm_behavior',['--cell',str(i)],'BEHAVIOR_%03d'%i)
    try:
        assert read(OUT/'INDEPENDENT_STAGE_A_AUDIT.json')['status']=='PASS'
        for p,h in read(rt.REPORT/'FROZEN_INPUTS.json')['files'].items():assert sha(ROOT/p)==h
        for k,t in [(48,0),(48,3)]:source(k,t)
        if not sources_only:
            for s,t in ((0,3),(3,0)):pair(48,s,t)
            from .confirm_behavior import cells
            cc=cells();pilot=[i for i,c in enumerate(cc) if c['seed']==48 and c['task'] in (0,3) and (c['kind']!='transfer' or (c['source'],c['target']) in ((0,3),(3,0)))]
            assert len(pilot)==10
            behavior(pilot)
            write(rt.REPORT/'PILOT_COMPLETE.json',dict(status='PASS',pairs=[[48,0,3],[48,3,0]],behavior_cells=pilot,trials=400,counted_in_total=True,selection='Fixed cross-suite tasks0/3; no outcome-dependent continuation',formal_training_resumed=False))
        for k in (48,49,50):
            for t in range(6):source(k,t)
        if sources_only:
            write(rt.REPORT/'QUEUE_STATUS.json',dict(status='SOURCES_COMPLETE',sources=18,formal_training_resumed=False));return
        for k in (48,49,50):
            for s in range(6):
                for t in range(6):
                    if s!=t:pair(k,s,t)
        behavior(range(234))
        execute('adjudicate_confirmation',[],'ADJUDICATE')
        write(rt.REPORT/'QUEUE_STATUS.json',dict(status='MEASUREMENTS_AND_STATISTICS_COMPLETE_REVIEW_REQUIRED',formal_training_resumed=False,updated_unix=time.time()))
    except Exception as error:
        write(rt.REPORT/('FAILURE_%d.json'%time.time()),dict(error=str(error),state=read(rt.REPORT/'QUEUE_STATUS.json'),formal_training_resumed=False));raise

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--sources-only',action='store_true');a=p.parse_args();main(a.sources_only)
