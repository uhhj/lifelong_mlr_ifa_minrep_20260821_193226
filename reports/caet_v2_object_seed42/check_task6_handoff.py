"""Fixed read-only SSH snapshot, metadata only; never a resume command."""
import datetime,getpass,hashlib,json,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
SCRIPT=r'''
import json,time,hashlib
from pathlib import Path
W=Path('/data/mlr_ifa/experiment_workspaces/CAET_v2_OBJECT_seed42_20261021');O=Path('/data/mlr_ifa/experiments/CAET_v2_OBJECT_seed42_20261021');P=Path('/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed42_20261019')
def read(p):return json.loads(p.read_text()) if p.exists() else None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
contract=read(O/'CONTRACT.json');src={k:sha(W/'source'/k)==v for k,v in contract['source_sha256'].items()};assert all(src.values())
parity=read(O/'TASK6_TENSOR_PARITY.json');assert parity is None or parity['equal']
initial=read(O/'ra_coreset/INITIAL_STATE.json')==read(P/'ra_coreset/INITIAL_STATE.json');assert initial
storage=read(O/'ra_coreset/storage/task6.json');baseline=read(P/'ra_coreset/storage/task6.json');membership=None
if storage:
 key=lambda x:(Path(x['source']).name,x['destination'],x['sha256'])
 membership=[key(x) for x in storage['selected']]==[key(x) for x in baseline['selected']]
 assert membership and all(sha(O/'ra_coreset/buffer'/x['destination'])==x['sha256'] for x in storage['selected'])
h=read(O/'ra_coreset/epochs.json') or []
print(json.dumps(dict(observed_unix=time.time(),status=read(O/'RUN_STATUS.json'),epochs=len(h),last_epoch=h[-1] if h else None,task6_tensor_parity=parity,initial_equal=initial,task6_ordered_members_and_payloads=membership,source_hashes=src,phase_exits={p.name:p.read_text().strip() for p in O.glob('*.exit')},fit7=read(O/'effects/models/task7/ACCEPTANCE.json'))))
'''

def main():
 p=R/('handoff_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));p.mkdir()
 c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
 try:
  c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
  i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(SCRIPT),timeout=180);raw=o.read();err=e.read();rc=o.channel.recv_exit_status()
  (p/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err.decode()),indent=2)+'\n',encoding='utf8');assert rc==0,err.decode()
  d=json.loads(raw);(p/'SNAPSHOT.json').write_bytes(raw);(p/'RECEIPT.json').write_text(json.dumps(dict(sha256=hashlib.sha256(raw).hexdigest()),indent=2)+'\n')
  print(json.dumps(dict(directory=str(p),result=d)))
 finally:pw=None;c.close()

if __name__=='__main__':main()
