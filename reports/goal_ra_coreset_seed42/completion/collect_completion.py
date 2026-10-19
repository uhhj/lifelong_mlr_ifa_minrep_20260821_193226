"""Read-only process inspection and metadata collection; never starts a scientific job."""
from pathlib import Path, PurePosixPath
import getpass, json, hashlib, datetime
import paramiko
R=Path(__file__).resolve().parent
W='/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed42_20261018'
O='/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed42_20261018'
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
c.connect('jq1.9gpu.com',port=11920,username='root',password=getpass.getpass('SSH password: '),look_for_keys=False,allow_agent=False,timeout=25)
commands=[]
def run(cmd):
    _,a,b=c.exec_command(cmd,timeout=600);out=a.read().decode();err=b.read().decode();code=a.channel.recv_exit_status()
    commands.append(dict(command=cmd,exit_code=code,stderr=err));assert code==0,(cmd,err)
    return out
try:
    prefix='source '+W+'/env.sh && /data/mlr_ifa/env/bin/python '
    snap=json.loads(run(prefix+W+'/monitor_snapshot.py'))
    print(json.dumps(snap,indent=2),flush=True)
    f=c.open_sftp()
    for n in ['pipeline','supervisor']:
        assert f.open(O+'/'+n+'.exit').read().strip()==b'0',n
    proof=json.loads(f.open(O+'/PROCESS_EXIT_ACCEPTANCE.json').read())
    assert proof['status']=='PASS' and not snap['processes']
    assert snap['records']==200 and snap['completed_vectors']==20 and snap['epochs']==400
    remote=run(prefix+W+'/package_light.py').strip()
    assert remote.startswith(W+'/light_') and '\n' not in remote
    package_raw=f.open(remote+'/PACKAGE_MANIFEST.json').read();package=json.loads(package_raw)
    E=R/'final_evidence';E.mkdir(exist_ok=False)
    for n,m in package['files'].items():
        p=PurePosixPath(n);assert not p.is_absolute() and '..' not in p.parts
        assert p.suffix in {'.json','.jsonl','.log','.exit','.pid','.started'}
        assert not {'features','buffer','models','recovery','smoke','bindings'}&set(p.parts)
        raw=f.open(remote+'/'+n).read();assert len(raw)==m['bytes'] and hashlib.sha256(raw).hexdigest()==m['sha256']
        dst=E/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(raw)
    (E/'PACKAGE_MANIFEST.json').write_bytes(package_raw)
    (R/'completion_snapshot.json').write_text(json.dumps(snap,indent=2)+'\n',encoding='utf-8')
    (R/'COLLECTION_RECEIPT.json').write_text(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),commands=commands,remote_package=remote,files=len(package['files']),weights_downloaded=False),indent=2)+'\n',encoding='utf-8')
    print('COLLECTED',len(package['files']),flush=True)
finally:c.close()
