"""30-second resource observation; no signals, optimizer, environment or training writes."""
import json,time,subprocess,os
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1')
def run(args):
 p=subprocess.run(args,capture_output=True,text=True);return dict(exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
def state():
 out={}
 for c in ['L005_41','L005_42','L005_43','L020_41','L020_42','L020_43']:
  p=Q/c/'TRAIN_STATUS.json'
  if p.exists():out[c]=json.loads(p.read_text())
 return out
def processes():
 out=[]
 for p in Path('/proc').glob('[0-9]*'):
  try:
   cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
   if int(p.name)!=os.getpid() and str(Q) in cmd:
    s=(p/'stat').read_text().split(') ',1)[1].split();out.append(dict(pid=int(p.name),state=s[0],cpu_ticks=int(s[11])+int(s[12]),start=s[19],cmd=cmd))
  except OSError:pass
 return out
fields=['utilization.gpu','utilization.memory','memory.used','memory.total','temperature.gpu','power.draw','power.limit','clocks.current.sm','clocks.current.memory','pstate']
start=time.time();before=state();pb=processes();samples=[]
for i in range(30):
 x=run(['nvidia-smi','--query-gpu='+','.join(fields),'--format=csv,noheader,nounits']);samples.append(dict(time=time.time(),**x))
 if i<29:time.sleep(1)
out=dict(status='PASS_READONLY_RESOURCE_WINDOW' if all(x['exit']==0 for x in samples) else 'RESOURCE_QUERY_FAILURE',observed_start=start,observed_end=time.time(),fields=fields,samples=samples,training_before=before,training_after=state(),processes_before=pb,processes_after=processes(),cpu_clock_ticks=os.sysconf('SC_CLK_TCK'),gpu_processes=run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader']),gpu_detail=run(['nvidia-smi','-q','-d','PERFORMANCE,CLOCK,POWER,TEMPERATURE']),optimizer_calls=0,environments=0,signals=0,production_writes=0)
print(json.dumps(out))
