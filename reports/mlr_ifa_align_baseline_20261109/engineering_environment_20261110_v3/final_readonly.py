from pathlib import Path
import sys,json,hashlib
Q=Path('/data/mlr_ifa/experiment_workspaces/MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261110_v2')
assert hashlib.sha256((Q/'CONTRACT.json').read_bytes()).hexdigest()=='255748b47194a8ec8ce24485cd89399058620ab03a29c14690d8f4c652e568dd'
sys.path.insert(0,str(Q))
import subprocess
from runtime import PY,processes,P
assert (Q/'controller.exit').read_text().strip()=='0' and not processes([Q,P])
p=subprocess.run([PY,'-B',str(Q/'final_driver.py')],cwd=Q,capture_output=True,text=True)
assert p.returncode==0,p.stderr
r=json.loads(p.stdout.strip().splitlines()[-1])
from observe import snapshot
s=snapshot();s.update(status='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT',cpu=r['results'],remote_cpu_directory=r['directory'])
print(json.dumps(s))
