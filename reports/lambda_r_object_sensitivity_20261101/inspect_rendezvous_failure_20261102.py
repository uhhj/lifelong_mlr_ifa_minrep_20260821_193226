"""Read-only interruption, checkpoint and installed launcher audit; no optimizer/environment."""
import json,hashlib,inspect,time,subprocess,os
from pathlib import Path
import torch,torch.distributed.run as run
Q=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1');R=Q/'L005_43';P=Path('/data/mlr_ifa/experiments')/Q.name/'L005_43/S';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();js=lambda p:json.loads(Path(p).read_text())
contracts={}
for path in [Q/'CONTRACT.json',*Q.glob('engineering_*/CONTRACT.json')]:
 c=js(path);assert all(sha(p)==h for p,h in c['files'].items());contracts[str(path)]=sha(path)
procs=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if str(Q) in cmd and int(p.name)!=os.getpid():procs.append(dict(pid=int(p.name),cmd=cmd))
 except OSError:pass
assert not procs
cpfile=P/'recovery.pth';cp=torch.load(cpfile,map_location='cpu');history=js(R/'EPOCHS_S.json');gates=[json.loads(x) for x in (R/'GATES.jsonl').read_text().splitlines()];assert cp['history']==history and len(history)==289 and cp['stage']==8 and cp['next_epoch']==89
assert cp['physical_optimizer_calls']==116102 and cp['steps']==65901
partial=gates[cp['gate_blocks']:];assert len(partial)==20 and all(x['epoch']==89 and x['stage']==8 for x in partial)
completed={}
for case in ['L005_41','L005_42']:
 r=Q/case;assert (r/'train_S.exit').read_text().strip()=='0';assert len(js(r/'EPOCHS_S.json'))==400
 completed[case]={str(p):sha(p) for p in r.glob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.exit']}
 for st in range(6,10):
  p=Path('/data/mlr_ifa/experiments')/Q.name/case/'S'/('stage%d.pth'%st);completed[case][str(p)]=sha(p)
source=inspect.getsource(run.run);parser=run.get_args_parser();args=parser.parse_args(['--standalone','--nproc_per_node=1','dummy.py'])
gpu=subprocess.run(['nvidia-smi'],capture_output=True,text=True)
print(json.dumps(dict(status='PASS_READ_ONLY_INTERRUPTION_AUDIT',time=time.time(),contracts=contracts,processes=procs,gpu=gpu.stdout,checkpoint=dict(path=str(cpfile),sha256=sha(cpfile),bytes=cpfile.stat().st_size,keys=list(cp),state_keys=list(cp['state']),stage=cp['stage'],next_epoch=cp['next_epoch'],history_epochs=len(history),steps=cp['steps'],physical_optimizer_calls=cp['physical_optimizer_calls'],gate_blocks=cp['gate_blocks'],history_exact=True),partial_gates=partial,partial_recorded_optimizer_calls=sum(x['steps']*2 for x in partial),partial_unrecorded_optimizer_calls='UNKNOWN',completed_hashes=completed,torch_version=torch.__version__,installed_launcher_path=run.__file__,installed_launcher_sha256=sha(run.__file__),installed_run_function=source,parser_standalone_fields={k:v for k,v in vars(args).items() if k.startswith('rdzv') or k in ['standalone','master_port','master_addr']},failed_training_log=(R/'train_S.log').read_text(),optimizer_calls_added=0,environments_added=0)))
