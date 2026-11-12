"""Finite current-task supervision; no app schedule, retries or extra experiments.

Wait for predicted whole-run/suite checkpoints and audit them locally. Publish
the final acceptance for the authorized reverse runs.
Any unexpected failure stops this worker for agent engineering review.
"""
from pathlib import Path
import datetime,json,os,subprocess,sys,time,traceback
R=Path(__file__).resolve().parent
OLD=R.parent/'20261110_s_stripped_mlr_ifa_align'
SYS='C:/Users/Lenovo/AppData/Local/Programs/Python/Python313/python.exe'
BUNDLED='C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1104-2876-0_26.1104.2876.0_x64__3k8sg7r9htsxt/dependencies/python/python.exe'
STATE=R/'CURRENT_TASK_WORKER.json'
J=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(**kw):
    value=dict(observed=datetime.datetime.now().astimezone().isoformat(),pid=os.getpid(),app_automation_created=False,**kw)
    STATE.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def run(folder,script,*args,python=SYS):
    cmd=[python,'-X','utf8',str(folder/script),*map(str,args)]
    env=os.environ.copy();env.update(PYTHONIOENCODING='utf-8',GIT_CONFIG_COUNT='5')
    for i,(key,value) in enumerate([('http.sslBackend','openssl'),('http.version','HTTP/1.1'),('http.sslVerify','true'),('http.lowSpeedLimit','1'),('http.lowSpeedTime','20')]):env['GIT_CONFIG_KEY_'+str(i)]=key;env['GIT_CONFIG_VALUE_'+str(i)]=value
    save(status='RUNNING_AUTHORIZED_STEP',directory=str(folder),script=script,args=list(map(str,args)))
    p=subprocess.run(cmd,cwd=R.parents[1],env=env,capture_output=True,text=True,encoding='utf-8')
    if p.returncode:
        failure=dict(status='STOPPED_FOR_AGENT_ENGINEERING_REVIEW',directory=str(folder),script=script,args=list(map(str,args)),exit=p.returncode,stdout=p.stdout,stderr=p.stderr,observed=datetime.datetime.now().astimezone().isoformat(),no_automatic_science_retry=True)
        (R/'CURRENT_TASK_WORKER_FAILURE.json').write_text(json.dumps(failure,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        raise RuntimeError(json.dumps(failure,ensure_ascii=False))
    return p.stdout
def snapshot(folder,script):
    output=run(folder,'transport.py',script)
    paths=[Path(line.strip()) for line in output.splitlines() if line.strip() and not line.lstrip().startswith('{')]
    matches=[p for p in paths if p.is_dir() and (p/'EVIDENCE.json').exists()]
    assert len(matches)==1,('Missing actual transport output directory',output[:2000])
    return matches[0]
def publish(folder,message):
    run(folder,'publish.py','--message',message)
    receipt=J(folder/'PUBLICATION_RECEIPT.json');assert receipt['status']=='PUSH_VERIFIED'
    return receipt['commit']
def wait_until(iso,phase):
    target=datetime.datetime.fromisoformat(iso)
    save(status='WAITING_FOR_WHOLE_PHASE_CHECKPOINT',next_checkpoint_CST=iso,phase=phase)
    print(json.dumps(dict(status='WAITING_FOR_WHOLE_PHASE_CHECKPOINT',phase=phase,next_checkpoint_CST=iso)),flush=True)
    while True:
        remaining=target.timestamp()-time.time()
        if remaining<=0:return
        time.sleep(min(50,remaining))
def checkpoint(folder,old=False):
    previous=J(folder/'RUN_STATUS.json').get('evidence_directory')
    d=snapshot(folder,'check_readonly.py')
    run(folder,'audit_progress.py',d,*([previous] if previous else []))
    run(folder,'record_progress.py',d)
    if old:run(folder,'record_checkpoint.py',d,python=BUNDLED)
    status=J(folder/'RUN_STATUS.json')
    if old:next_info=J(folder/'TRACKING_STATUS.json')
    else:next_info=status
    print(json.dumps(dict(status='CHECKPOINT_AUDITED_LOCALLY',directory=str(folder),observed=status['observed'],totals=status.get('totals'),next_checkpoint_CST=next_info.get('next_checkpoint_CST')),ensure_ascii=False),flush=True)
    return d
def finish_old():
    status=J(OLD/'RUN_STATUS.json')
    if not status.get('actual_final_cpu_acceptance'):
        d=snapshot(OLD,'final_readonly_engineering02.py')
        run(OLD,'audit_final_local.py',d)
        run(OLD,'describe_final.py',d)
    assert J(OLD/'RUN_STATUS.json')['actual_final_cpu_acceptance']
    status=J(OLD/'RUN_STATUS.json');receipt=J(OLD/'PUBLICATION_RECEIPT.json')
    if status['status']=='COMPLETE_ACCEPTED_AND_PUBLISHED' and receipt['status']=='PUSH_VERIFIED':
        # Resuming this observer must not manufacture a new predecessor round.
        import hashlib
        final=Path(status['evidence_directory']);name=str((final/'LOCAL_FINAL_AUDIT.json').relative_to(OLD)).replace('\\','/')
        assert J(OLD/'PUBLICATION_REVIEW.json')['files_sha256'][name]['sha256']==hashlib.sha256((final/'LOCAL_FINAL_AUDIT.json').read_bytes()).hexdigest()
        commit=receipt['commit']
    else:commit=publish(OLD,'Accept twelve paired stripped-S runs and publish complete metrics')
    print(json.dumps(dict(status='PREDECESSOR_ACCEPTED_AND_PUBLISHED',commit=commit)),flush=True)
def main():
    # Existing prepared outputs and current scientific queue are never repeated.
    assert J(R/'PREPARATION_LOCAL_AUDIT.json')['status']=='PASS_PREPARED_NOT_LAUNCHED'
    if '--running-queue-only' in sys.argv:
        # Resume observation only. Do not revisit, republish or dispatch the
        # completed predecessor, or enter any deployment/launch path.
        assert J(R/'RUN_STATUS.json')['status']=='PARTIAL_EXECUTION_PENDING_FINAL_ACCEPTANCE'
        assert (R/'LAUNCH_AUTHORIZATION.json').exists()
    else:
        while not J(OLD/'RUN_STATUS.json').get('actual_final_cpu_acceptance'):
            tracking=J(OLD/'TRACKING_STATUS.json')
            if tracking['phase']=='READY_FINAL_CPU':break
            wait_until(tracking['next_checkpoint_CST'],'PREDECESSOR_'+tracking['phase'])
            checkpoint(OLD,old=True)
        finish_old()
    new=J(R/'RUN_STATUS.json')
    if new['status']=='PREPARED_FROZEN_NOT_LAUNCHED':
        assert not (R/'LAUNCH_AUTHORIZATION.json').exists(),'Review any previous launch result before continuing'
        run(R,'launch_after_current.py')
        # One startup admission observation is distinct from internal task polling.
        wait_until((datetime.datetime.now().astimezone()+datetime.timedelta(minutes=3)).isoformat(),'REVERSE_UNIQUE_STARTUP_ADMISSION')
        checkpoint(R)
    while True:
        status=J(R/'RUN_STATUS.json')
        if status.get('actual_final_cpu_acceptance'):break
        if status['phase']=='READY_ACTUAL_FINAL_CPU':
            d=snapshot(R,'final_readonly.py')
            run(R,'audit_final_local.py',d)
            run(R,'describe_final.py',d)
            break
        wait_until(status['next_checkpoint_CST'],'REVERSE_'+status['phase'])
        checkpoint(R)
    commit=publish(R,'Accept complete main-table-protocol reverse-order MLR and DEBR experiment')
    save(status='COMPLETE_ACCEPTED_AND_PUBLISHED',commit=commit,totals=J(R/'RUN_STATUS.json')['totals'])
    print(json.dumps(J(STATE),ensure_ascii=False),flush=True)
if __name__=='__main__':
    import msvcrt
    with (R/'CURRENT_TASK_WORKER.lock').open('a+b') as lock:
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:main()
        except BaseException:
            save(status='STOPPED_FOR_AGENT_ENGINEERING_REVIEW',traceback=traceback.format_exc(),no_automatic_science_retry=True)
            raise
