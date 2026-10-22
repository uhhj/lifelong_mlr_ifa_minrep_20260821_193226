import getpass,json,datetime,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
source=(R/'additions/caet_v3_exp/model.py').read_text(encoding='utf8')+'\n'+(R/'cpu_checks_tail.py').read_text(encoding='utf8')
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
out=R/('cpu_checks_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir()
try:
 c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
 i,o,e=c.exec_command("env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 LIBERO_CONFIG_PATH=/data/mlr_ifa/config HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/data/mlr_ifa/cache/huggingface MPLBACKEND=Agg /data/mlr_ifa/env/bin/python -c "+shlex.quote(source),timeout=180)
 i.channel.shutdown_write();raw=o.read();err=e.read().decode();rc=o.channel.recv_exit_status();(out/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err),indent=2)+'\n');assert rc==0,err
 (out/'EVIDENCE.json').write_bytes(raw);print(out);d=json.loads(raw);print(json.dumps({k:v for k,v in d.items() if k!='base_schedule_source'},indent=2))
finally:pw=None;c.close()
