"""Run CPU gates now; wait for original GPU before any policy construction."""
import os,sys,time,subprocess,traceback,signal,zipfile,shutil
from .common import *

def stage(label,args,budget=12*3600):
    with (OUT/(label+'.log')).open('w') as f:
        child=subprocess.Popen([sys.executable]+args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        write(OUT/'QUEUE_STATUS.json',dict(status='RUNNING',stage=label,pid=os.getpid(),child_pid=child.pid,time=time.time()))
        started=time.time()
        try:
            while child.poll() is None:
                assert time.time()-started<budget,'Frozen stage budget exceeded'
                if label in ['preflight6','main36']:
                    from functional_geometry_diagnostic_v1.runtime_guard import memory,size
                    lim=read(OUT/'PROTOCOL_FROZEN.json')['resources'];m=memory(child.pid)
                    assert m['rss']<=lim['process_RSS_GiB']*2**30 and m['available']>=lim['system_MemAvailable_min_GiB']*2**30,'Host memory resource gate'
                    assert size(OUT)<=lim['new_data_disk_artifacts_max_GiB']*2**30,'Output resource gate'
                time.sleep(5)
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise
        assert child.returncode==0,(label,child.returncode)

def main():
    p=read(OUT/'PROTOCOL_FROZEN.json');write(OUT/'QUEUE_STARTED.json',dict(pid=os.getpid(),time=time.time(),no_rollout=True))
    stage('cpu_audit',['-m','sample_task_qp_v1.audit'],3600)
    stage('cpu_tests',['-m','sample_task_qp_v1.tests'],600)
    waiting=time.time();last=None
    while True:
        q=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader,nounits'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        text=q.stdout.strip()
        if q.returncode==0 and text:
            assert len(text.splitlines())==1 and text.split(',')[0].strip()==p['resources']['device'],('GPU differs from original carrier',text)
            from functional_geometry_diagnostic_v1.runtime_guard import memory
            if memory()['available']>=4*2**30:break
            why='GPU visible but host available RAM below4GiB startup gate'
        else:why='No usable GPU reported; no policy model, optimizer update, or forward measurement has started'
        write(OUT/'QUEUE_STATUS.json',dict(status='WAITING_FOR_GPU',pid=os.getpid(),reason=why,expected_device=p['resources']['device'],wait_started=waiting,time=time.time(),cpu_audit='PASS',cpu_tests='PASS'))
        if time.time()-waiting>=p['gpu_wait_hours']*3600:
            write(OUT/'QUEUE_STATUS.json',dict(status='GPU_WAIT_EXPIRED',pid=os.getpid(),time=time.time(),measurements_started=False));return
        time.sleep(60)
    os.environ['FG_QUEUE_STARTED']=str(time.time())
    stage('preflight6',['-m','sample_task_qp_v1.measure','preflight'])
    stage('main36',['-m','sample_task_qp_v1.measure','main'])
    stage('analysis',['-m','sample_task_qp_v1.analyze'],3600)
    stage('independent_audit',['-m','sample_task_qp_v1.finish'],3600)
    write(OUT/'QUEUE_STATUS.json',dict(status='COMPLETE',pid=os.getpid(),time=time.time(),points=36,horizon=1,no_rollout=True))
    files=list(OUT.rglob('*'))+list((ROOT/'sample_task_qp_v1').glob('*.py'));files=[x for x in files if x.is_file()]
    write(OUT/'DELIVERY_MANIFEST.json',{str(x.relative_to(ROOT)):sha(x) for x in files})
    with zipfile.ZipFile('/root/sample_task_qp_v1_delivery.zip','w',zipfile.ZIP_DEFLATED) as z:
        for x in files+[OUT/'DELIVERY_MANIFEST.json']:z.write(x,str(x.relative_to(ROOT)))

if __name__=='__main__':
    try:main()
    except BaseException:write(OUT/'QUEUE_STATUS.json',dict(status='FAILED_STOP',pid=os.getpid(),time=time.time(),error=traceback.format_exc()));raise
