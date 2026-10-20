"""One detached finite invocation, no retries or new resources."""
import os, sys, subprocess, time
from server_runner import N, write, js, workers, verify
verify()
assert not (N/'supervisor.started').exists()
write(N/'supervisor.started',dict(pid=os.getpid(),time=time.time()))
with (N/'controller.log').open('x') as log:
    r=subprocess.run([sys.executable,'-u',str(N/'server_runner.py'),'run'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
(N/'supervisor.exit').write_text(str(r.returncode)+'\n')
if r.returncode==0:
    assert (N/'controller.exit').read_text().strip()=='0' and not workers()
    assert js(N/'FINAL_ACCEPTANCE.json')['status']=='PASS'
    write(N/'PROCESS_EXIT_ACCEPTANCE.json',dict(status='PASS',controller_exit=0,supervisor_exit=0,no_workers=True,time=time.time()))
raise SystemExit(r.returncode)
