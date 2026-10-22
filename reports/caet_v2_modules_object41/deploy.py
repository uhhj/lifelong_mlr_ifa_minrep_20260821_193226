"""Create fresh immutable variant workspaces and start only the authorized queue."""
import datetime,getpass,hashlib,json,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
Q='/data/mlr_ifa/experiment_workspaces/CAETv2_TAC_RSET_EFTR_OBJECT41_20261022'
SCRIPT=r'''
import sys,json,hashlib,subprocess,shutil,time,os
from pathlib import Path
d=json.load(sys.stdin);Q=Path(d['q']);assert not Q.exists()
BASE=Path('/data/mlr_ifa/experiment_workspaces/CAET_v2_OBJECT_seed42_20261021/source')
P=Path('/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed41_20261019')
expected=json.loads(Path('/data/mlr_ifa/experiments/CAET_v2_OBJECT_seed42_20261021/CONTRACT.json').read_text())['source_sha256']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert all(sha(BASE/n)==h for n,h in expected.items())
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied'
assert Path('/data/mlr_ifa/experiment_workspaces/CAET_v2_OBJECT_extension_LIBERO50_20261021/libero50_taskmap01/PRETRAINING_CANCELLED_BY_USER.json').exists()
assert json.loads((P/'AUTONOMOUS_EXIT_ACCEPTANCE.json').read_text())['status']=='PASS'
Q.mkdir();files={};entries=[];differences={}
def wr(p,obj):p.write_text(json.dumps(obj,indent=2)+'\n')
for name,txt in d['top'].items():
 p=Q/name;p.write_text(txt);files[str(p)]=sha(p)
for variant in ('TAC','RSET','EFTR'):
 w=Q/variant;w.mkdir();s=w/'source';shutil.copytree(BASE,s,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for name,txt in d['source'].items():(s/name).write_text(txt)
 for n,txt in d['gates'][variant].items():(w/n).write_text(txt);files[str(w/n)]=sha(w/n)
 for n in ('PROTOCOL.md','IMPLEMENTATION_NOTES.md'):(w/n).write_bytes((Q/n).read_bytes());files[str(w/n)]=sha(w/n)
 out='/data/mlr_ifa/experiments/CAETv2_%s_OBJECT_seed41_20261022'%variant
 assert not Path(out).exists()
 wr(w/'ENTRY.json',dict(seed=41,variant=variant,parent=str(P),output=out));files[str(w/'ENTRY.json')]=sha(w/'ENTRY.json')
 actual={str(p.relative_to(s)):sha(p) for p in s.rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.sh') and '__pycache__' not in p.parts}
 changes={k:dict(before=expected.get(k),after=v) for k,v in actual.items() if expected.get(k)!=v}
 assert all(k.startswith('caet_v2_exp/') for k in changes)
 differences[variant]=dict(source_files=len(actual),unchanged=sum(expected.get(k)==v for k,v in actual.items()),changes=changes)
 files.update({str(s/k):h for k,h in actual.items()})
 entries.append(dict(key=variant+'_OBJECT_41',method='CAET-v2+'+variant,seed=41,output=out,storage_mount='/data',minimum_free_bytes=1600000000,argv=['/data/mlr_ifa/env/bin/python','-u',str(Q/'seed_supervisor.py'),str(w)]))
wr(Q/'QUEUE.json',entries);files[str(Q/'QUEUE.json')]=sha(Q/'QUEUE.json')
wr(Q/'SOURCE_DIFF.json',differences);wr(Q/'ORCHESTRATION_CONTRACT.json',dict(files=files,created_unix=time.time(),variants=['TAC','RSET','EFTR'],cancelled_excluded=['COVR','LIBERO50']))
with (Q/'controller.log').open('x') as f:
 p=subprocess.Popen(['/data/mlr_ifa/env/bin/python','-u',str(Q/'queue_controller.py')],cwd=Q,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='DISPATCHED_AWAITING_GPU_ADMISSION',controller_pid=p.pid,queue=entries,source_diff=differences,contract_sha256=sha(Q/'ORCHESTRATION_CONTRACT.json'),time=time.time())
wr(Q/'DISPATCH_RECEIPT.json',receipt);print(json.dumps(receipt))
'''
def main():
 data=dict(q=Q,top={n:(R/n).read_text(encoding='utf8') for n in ('PROTOCOL.md','IMPLEMENTATION_NOTES.md','queue_controller.py','seed_supervisor.py')},source={p.relative_to(R/'additions').as_posix():p.read_text(encoding='utf8') for p in (R/'additions').rglob('*.py')},gates={})
 for v in ('TAC','RSET','EFTR'):
  g=R/('gate1_'+v);a=json.loads((g/'LOCAL_AUDIT.json').read_text());assert a['status']=='GATE1_PASS'
  assert json.loads((g/'EVIDENCE.json').read_text())['protocol_sha256']==hashlib.sha256((R/'PROTOCOL.md').read_bytes()).hexdigest()
  data['gates'][v]={'GATE1_EVIDENCE.json':(g/'EVIDENCE.json').read_text(),'GATE1_LOCAL_AUDIT.json':(g/'LOCAL_AUDIT.json').read_text()}
 c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
 stamp=R/('launch_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));stamp.mkdir()
 try:
  c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
  i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(SCRIPT),timeout=240);i.write(json.dumps(data));i.flush();i.channel.shutdown_write()
  raw=o.read();err=e.read();rc=o.channel.recv_exit_status()
  (stamp/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err.decode()),indent=2)+'\n')
  assert rc==0,err.decode();(stamp/'DISPATCH_RECEIPT.json').write_bytes(raw);print(raw.decode())
 finally:pw=None;c.close()
if __name__=='__main__':main()
