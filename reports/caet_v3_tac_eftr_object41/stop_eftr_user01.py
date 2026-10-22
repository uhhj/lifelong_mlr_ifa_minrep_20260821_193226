"""Explicit user termination, retaining all outputs and original exit evidence."""
import os,signal,time,json,hashlib,subprocess
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/CAETv3_TAC_EFTR_ALIGN_OBJECT41_20261022_collate01')
O=Path('/data/mlr_ifa/experiments/CAETv3_EFTR_OBJECT_seed41_20261022_collate01')
D=Q/'eftr_user_stop01'
def read(p):
 if not p.exists():return None
 return json.loads(p.read_text()) if p.suffix=='.json' else p.read_text()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def processes():
 result=[]
 for p in Path('/proc').glob('[0-9]*'):
  try:
   cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode();cwd=str((p/'cwd').resolve())
   if int(p.name)!=os.getpid() and 'python -c' not in cmd and (str(Q) in cmd or str(Q) in cwd):result.append(dict(pid=int(p.name),command=cmd,cwd=cwd))
  except OSError:pass
 return result
assert not D.exists(),'Termination is not implicitly repeatable'
contract=read(Q/'ORCHESTRATION_CONTRACT.json')
assert all(sha(Path(p))==h for p,h in contract['files'].items())
assert read(Q/'RUN_STATUS.json')['current']=='EFTR_OBJECT_41'
assert not Path('/data/mlr_ifa/experiments/CAETv3_ALIGN_OBJECT_seed41_20261022_collate01').exists()
before=processes();targets=[x for x in before if x['cwd']==str(Q/'EFTR/source') and 'torch.distributed.run' in x['command'] and '-m caet_v3_exp.train ' in x['command']]
assert len(targets)==1,targets
pid=targets[0]['pid'];assert read(O/'train8.pid.json')['pid']==pid
D.mkdir()
request=dict(status='USER_TERMINATION_REQUESTED',unix=time.time(),reason='User explicitly stopped current EFTR; preserve checkpoints/logs/diagnostics; no complete negative-result comparison; continue frozen ALIGN only',signal='SIGTERM',target=targets[0],processes_before=before,original_contract_sha256=sha(Q/'ORCHESTRATION_CONTRACT.json'))
(D/'REQUEST.json').write_text(json.dumps(request,indent=2)+'\n')
os.kill(pid,signal.SIGTERM)
deadline=time.monotonic()+50
while processes() and time.monotonic()<deadline:time.sleep(1)
after=processes()
evidence=dict(request=request,processes_after=after,unix=time.time(),queue_exit=read(Q/'controller.exit'),queue_failure=read(Q/'FAILURE.json'),worker_exit=read(Q/'EFTR/supervisor.exit'),pipeline_exit=read(O/'pipeline.exit'),train_exit=read(O/'train8.exit'),gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True))
(D/'STOP_ATTEMPT.json').write_text(json.dumps(evidence,indent=2)+'\n')
assert not after,'Residual processes; inspect before any next action'
assert evidence['queue_exit'] is not None and evidence['queue_exit'].strip()!='0'
assert not evidence['gpu'].strip(),'GPU remains occupied'
assert all(sha(Path(p))==h for p,h in contract['files'].items())
history=read(O/'ra_coreset/epochs.json') or []
payload={str(p.relative_to(O)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(O.rglob('*.pth')) if p.is_file()}
assert 'ra_coreset/recovery/latest.pth' in payload
formal=sum(len(p.read_text().splitlines()) for p in (O/'ra_coreset/evaluation').glob('trials*.jsonl'))
evidence.update(status='EARLY_TERMINATED_BY_USER',complete_epochs=len(history),last_complete_epoch=history[-1] if history else None,formal_records=formal,checkpoint_inventory=payload,frozen_hashes=dict(count=len(contract['files']),status='PASS'),final_acceptance=False,complete_negative_result_comparison=False,preservation='No payload or log deleted; partial-epoch work is not counted as completed training; last complete recovery retained')
(D/'TERMINATION_ACCEPTANCE.json').write_text(json.dumps(evidence,indent=2)+'\n')
(O/'USER_TERMINATION.json').write_text(json.dumps(evidence,indent=2)+'\n')
print(json.dumps(evidence))
