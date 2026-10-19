"""Read-only parent/resource inventory; no torch/model/environment calls."""
from pathlib import Path
import paramiko,getpass,json
R=Path(__file__).resolve().parent
code=r'''
from pathlib import Path
import json,shutil,subprocess,hashlib,time
parents={41:'/root/mlr_ifa_new_experiments/Seeds40_41_three_arm_20261016/seed41/rc',42:'/data/mlr_ifa/experiments/Coreset_paper_OBJECT_seed42_20261013',43:'/data/mlr_ifa/experiments/Seed43_Random_Coreset_repeat_20261016_v1'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
out={'time':time.time(),'free_bytes':{p:shutil.disk_usage(p).free for p in ['/data','/root']},'gpu':subprocess.check_output(['nvidia-smi'],text=True),'processes':[x for x in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines() if ' -m ' in x and any(w in x for w in ['coreset','zr2_exp','ra_exp'])],'parents':{}}
for s,r in parents.items():
 p=Path(r);v={'root':r,'exists':p.exists(),'files':{}}
 for n in ['CONTRACT.json','FEATURES.json','MODAL_CONTRACT.json','TASKS.json','SUMMARY.json','INDEPENDENT_ACCEPTANCE.json','resolved_coreset.json','resolved_eval.json','PRETRAIN_METADATA.json']:
  q=p/n
  if q.exists():v['files'][n]={'sha256':sha(q),'content':json.loads(q.read_text())}
 v['top_files']=[x.name for x in p.iterdir()] if p.exists() else []
 v['checkpoint_files']={str(x.relative_to(p)):{'bytes':x.stat().st_size,'sha256':sha(x)} for x in (p/'coreset/models').glob('*ep99.pth')}
 v['feature_counts']={str(t):len(list((p/'features'/str(t)).glob('*.pt'))) for t in range(6,10)}
 out['parents'][str(s)]=v
print(json.dumps(out))
'''
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
c.connect('jq1.9gpu.com',port=11920,username='root',password=getpass.getpass('SSH password: '),look_for_keys=False,allow_agent=False,timeout=25)
try:
 cmd="/data/mlr_ifa/env/bin/python - <<'PY'\n"+code+'\nPY';_,a,b=c.exec_command(cmd,timeout=180);raw=a.read();err=b.read().decode();rc=a.channel.recv_exit_status()
 (R/'PROBE_COMMAND.json').write_text(json.dumps(dict(command=cmd,exit_code=rc,stderr=err),indent=2)+'\n');assert rc==0,err
 (R/'RESOURCE_INVENTORY.json').write_bytes(raw);v=json.loads(raw)
 print(json.dumps({**{k:v[k] for k in ['free_bytes','gpu','processes']},'parents':{s:{k:p[k] for k in ['root','exists','feature_counts','top_files']} for s,p in v['parents'].items()}},indent=2))
finally:c.close()
