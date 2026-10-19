import os,json,subprocess,time,sys
from pathlib import Path
N=Path('/data/mlr_ifa/experiment_workspaces/RA_ZRv2_autonomous_20261019');sys.path.insert(0,str(N))
from flow_support import verify_controller,sha,alive
verify_controller()
for name in ['target_tests','admission_inventory','controller_paths']:
 assert (N/(name+'.exit')).read_text().strip()=='0',name
assert not (N/'AUTONOMOUS_LAUNCH.json').exists()
assert not (N/'wrapper.started').exists() and not (N/'controller.started').exists()
r=json.loads((N/'recovery_launch.json').read_text())
assert alive(r['pid']) or (N/'stage1/41/driver.exit').exists()
with (N/'supervisor.log').open('x') as f:
 p=subprocess.Popen(['/data/mlr_ifa/env/bin/python','-u',str(N/'supervisor.py')],cwd=str(N),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
x=dict(status='LAUNCHED_PARTIAL',pid=p.pid,time=time.time(),contract_sha256=sha(N/'AUTONOMOUS_CONTRACT.json'),command=['/data/mlr_ifa/env/bin/python','-u',str(N/'supervisor.py')],detached=True,existing_recovery_pid=r['pid'])
(N/'AUTONOMOUS_LAUNCH.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x,indent=2))
