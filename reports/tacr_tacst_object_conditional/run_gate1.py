"""CPU-only causal binding and fixed RA graph census, remote read-only."""
from pathlib import Path
import json,datetime,hashlib,getpass,shlex
import paramiko
R=Path(__file__).resolve().parent
source=(R/'additions/caet_v3_exp/science.py').read_text(encoding='utf8')
tail='''
import subprocess
out=dict(status='PASS',tasks={},stages={},optimizer_updates=0,model_forwards=0,environment_steps=0,server_writes=0)
old=[]
for t in range(6,10):
 rows,selected,proof=derive_task(t)
 current=[r for r in rows.values() if r['valid'] and not r['holdout']]
 g=graph(current,old)
 out['tasks'][str(t)]=proof;out['stages'][str(t)]=g
 old.extend(r for r in selected if r['valid'] and not r['holdout'])
out['gpu']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
out['cuda_initialized']=torch.cuda.is_initialized();assert not out['cuda_initialized']
out['disk']={m:dict(ordinary=(s:=os.statvfs(m)).f_bavail*s.f_frsize,root=s.f_bfree*s.f_frsize) for m in ('/data','/')}
print(json.dumps(out))
'''
p=R/('gate1_seed'+__import__('sys').argv[1]+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));p.mkdir()
(p/'PRECOMMIT.json').write_text(json.dumps(dict(protocol_sha256=hashlib.sha256((R/'PROTOCOL.md').read_bytes()).hexdigest(),science_sha256=hashlib.sha256(source.encode()).hexdigest()),indent=2))
seed=int(__import__('sys').argv[1]);assert seed in (41,42,43)
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
try:
 c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,allow_agent=False,look_for_keys=False,timeout=20);pw=None
 cmd=("env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 RA_BASE=/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed%d_20261019 /data/mlr_ifa/env/bin/python -c "+shlex.quote(source+tail)) % seed
 i,o,e=c.exec_command(cmd,timeout=600);i.channel.shutdown_write();raw=o.read();err=e.read().decode();rc=o.channel.recv_exit_status()
 (p/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err),indent=2));assert rc==0,err
 (p/'EVIDENCE.json').write_bytes(raw);d=json.loads(raw)
 assert d['status']=='PASS' and not d['cuda_initialized']
 for t,v in d['tasks'].items():
  assert v['valid_bc_labels']>=v['valid_next_labels']
  assert all(x['max_input_obs_index']+1==x['next_action_index']==x['from_state'] and x['to_state']==x['from_state']+1 for x in v['labels_metadata'])
  assert all(x['post_obs_alignment_exact'] for x in v['physical'].values())
 audit=dict(status='PASS',evidence_sha256=hashlib.sha256(raw).hexdigest(),tasks={t:{k:v[k] for k in ('total_windows','valid_bc_labels','valid_next_labels','invalid','ra_quota')} for t,v in d['tasks'].items()},stages={t:g['supported_current'] for t,g in d['stages'].items()})
 (p/'LOCAL_AUDIT.json').write_text(json.dumps(audit,indent=2))
 (R/('GATE1_seed%d_EVIDENCE.json'%seed)).write_bytes(raw);(R/('GATE1_seed%d_LOCAL_AUDIT.json'%seed)).write_text(json.dumps(audit,indent=2))
 print(p);print(json.dumps(audit,indent=2))
finally:pw=None;c.close()
