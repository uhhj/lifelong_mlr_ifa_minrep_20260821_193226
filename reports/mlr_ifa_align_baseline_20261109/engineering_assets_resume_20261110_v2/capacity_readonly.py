from pathlib import Path
import sys,json,time,subprocess
Q=Path('/data/mlr_ifa/experiment_workspaces/MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261110_v2');E=Q/'engineering_assets_resume_20261110_v2'
sys.path.insert(0,str(E));from resume_runtime import *
verify();samples=[]
for i in range(5):
 r=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw,clocks.sm','--format=csv,noheader'],capture_output=True,text=True);samples.append(dict(time=time.time(),code=r.returncode,text=r.stdout))
 if i<4:time.sleep(1)
p=E/'OBJECT_41_evaluate.log'
print(json.dumps(dict(status='READONLY_LIVE_EVALUATION',observed=time.time(),gpu=samples,log_tail=p.read_text()[-5000:],processes=processes([Q,P]),new_optimizer=0,new_environment=0)))
