"""One authorized formal launch, after frozen-input and GPU admission checks."""
import os,json,subprocess,time,shutil
from pathlib import Path
from coreset_exp.common import O,verify,sha,write
W=Path('/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed42_20261018')
verify();g=json.loads((O/'GPU_PREFLIGHT.json').read_text())
assert g['status']=='PASS' and g['isolated_optimizer_calls']==4 and g['environment_steps']==0
assert g['contract_sha256']==sha(O/'CONTRACT.json')
assert not (O/'pipeline.started').exists() and not (O/'supervisor.started').exists()
running=subprocess.check_output(['ps','-eo','args'],text=True).splitlines()
assert not any(' -m coreset_exp.train ' in l or ' -m coreset_exp.evaluate ' in l for l in running)
assert shutil.disk_usage('/root').free>4*1024**3
with (O/'launcher.log').open('xb') as f:
    p=subprocess.Popen(['bash',str(W/'supervise.sh')],cwd=str(W/'source'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=os.environ.copy())
write(O/'LAUNCH.json',dict(status='PARTIAL',time=time.time(),pid=p.pid,command=['bash',str(W/'supervise.sh')],preflight_sha256=sha(O/'GPU_PREFLIGHT.json'),contract_sha256=sha(O/'CONTRACT.json')))
time.sleep(2);assert p.poll() is None
print(json.dumps(dict(status='LAUNCHED_PARTIAL',pid=p.pid,output=str(O))))
