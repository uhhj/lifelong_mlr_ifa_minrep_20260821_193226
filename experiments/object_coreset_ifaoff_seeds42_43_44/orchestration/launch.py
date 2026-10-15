import json,subprocess,hashlib,datetime,shutil
from pathlib import Path
W=Path('/data/mlr_ifa/experiment_workspaces/Coreset_validation_20261014/original_protocol')
Q=Path('/data/mlr_ifa/analyses/Coreset_IFAoff_original_protocol_20261014')
assert (Q/'prepare.exit').read_text().strip()=='0'
assert not (Q/'queue.started').exists() and not (Q/'launch_receipt.json').exists()
assert shutil.disk_usage(Q).free>8*1024**3
lines=subprocess.check_output(['ps','-eo','args'],text=True).splitlines()
assert not any(('evaluate_validation.py' in x and '--out' in x) or ' -m coreset_exp.train ' in x or x.startswith('bash '+str(W/'queue.sh')) for x in lines)
for p in W.glob('*.sh'):subprocess.run(['bash','-n',str(p)],check=True)
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in W.iterdir() if p.suffix in ('.py','.sh','.md')}
for n in ('preflight_noifa.py','accept_noifa_training.py'):
 p=W.parent/n;hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
with (Q/'orchestration_hashes.json').open('x') as f:json.dump(hashes,f,indent=2)
with (Q/'queue.log').open('xb') as f:
 p=subprocess.Popen(['bash',str(W/'queue.sh')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'pid':p.pid,'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':['bash',str(W/'queue.sh')],'new_formal_budget':1200,'cancelled_new_initial_eval_will_not_resume':True}
with (Q/'launch_receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
