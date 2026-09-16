"""User-requested seed45-only continuation; original frozen protocol is retained."""
import os, signal, time, traceback, shutil
from .common import *
from .queue import stage

def active(pid):
    p=Path('/proc')/str(pid)/'stat'
    return p.exists() and p.read_text().split(') ',1)[1].split()[0] != 'Z'

def main():
    verify()
    amendment=read(OUT/'SEED45_SCOPE_AMENDMENT.json')
    for name,h in amendment['implementation_files'].items():
        assert sha(ROOT/name)==h
    p=read(OUT/'PROTOCOL_FROZEN.json');name=p['points'][0]
    assert name=='s45_t8_e33_b285'
    assert read(OUT/'PREFLIGHT_COMPLETE.json')['status']=='PASS'
    old=amendment['previous_supervisor_pid'];child=amendment['adopted_training_pid']
    assert Path('/proc',str(old),'stat').read_text().split(') ',1)[1].split()[0]=='T'
    assert 'sample_step_closedloop_v1.queue' in Path('/proc',str(old),'cmdline').read_text()
    os.environ['SAMPLE_QP_QUEUE_START']=str(read(OUT/'QUEUE_STARTED.json')['time'])
    # Stop only the suspended scheduler. The training process has its own session.
    os.kill(old,signal.SIGKILL)
    write(OUT/'SEED45_HANDOFF.json',dict(time=time.time(),pid=os.getpid(),old_pid=old,child_pid=child,training_uninterrupted=True))
    (OUT/'QUEUE.pid').write_text(str(os.getpid())+'\n')
    d=OUT/'training'/name/'SAMPLE_STEP_QP';r=p['resources']
    started=read(d/'STARTED.json')['time']
    write(OUT/'QUEUE_STATUS.json',dict(status='RUNNING',stage='train_'+name+'_SAMPLE_STEP_QP',pid=os.getpid(),child_pid=child,seeds=[45],paused_seeds=[46,47],time=time.time()))
    try:
        while active(child):
            from functional_geometry_diagnostic_v1.runtime_guard import memory,size
            m=memory(child)
            assert time.time()-started<r['train_job_hours']*3600,'Time watchdog'
            assert time.time()-float(os.environ['SAMPLE_QP_QUEUE_START'])<r['wall_hours']*3600
            assert m['rss']<=r['host_rss_GiB']*2**30 and m['available']>=r['host_available_min_GiB']*2**30
            assert shutil.disk_usage(ROOT).free>=r['data_free_min_GiB']*2**30
            assert shutil.disk_usage('/root').free>=r['root_free_min_GiB']*2**30
            assert size(OUT)<r['new_reports_max_GiB']*2**30 and size(SCRATCH)<r['new_states_max_GiB']*2**30
            time.sleep(5)
    except BaseException:
        if active(child):os.killpg(child,signal.SIGTERM)
        raise
    for a in ARMS:
        meta=read(OUT/'training'/name/a/'COMPLETE.json')
        assert meta['status']=='PASS' and meta['steps']==300
        assert sha(meta['checkpoint']['path'])==meta['checkpoint']['sha256']
    cells=[(i,c) for i,c in enumerate(p['cells']) if c['environment_seed']==45]
    assert len(cells)==9
    for i,c in cells:
        stage('eval_'+str(i).zfill(2),['-m','sample_step_closedloop_v1.evaluate',str(i)],r['eval_cell_hours']*3600)
    stage('analysis_seed45',['-m','sample_step_closedloop_v1.analyze_seed45'],3600)
    stage('independent_audit_seed45',['-m','sample_step_closedloop_v1.audit_seed45'],3600)
    assert read(OUT/'INDEPENDENT_AUDIT_SEED45.json')['status']=='PASS'
    write(OUT/'QUEUE_STATUS.json',dict(status='SEED45_COMPLETE_OTHERS_PAUSED',time=time.time(),seeds=[45],paused_seeds=[46,47],jobs=3,trials=450))

if __name__=='__main__':
    try:main()
    except BaseException:
        write(OUT/'QUEUE_STATUS.json',dict(status='FAILED_STOP',pid=os.getpid(),time=time.time(),error=traceback.format_exc()))
        raise
