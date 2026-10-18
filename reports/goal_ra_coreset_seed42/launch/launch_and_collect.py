"""Authorized one-time launch and metadata-only collection. Password only getpass."""
from pathlib import Path,PurePosixPath
import getpass,hashlib,json,subprocess,sys
import paramiko
R=Path(__file__).resolve().parent
W='/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed42_20261018'
O='/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed42_20261018'
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
c.connect('jq1.9gpu.com',port=11920,username='root',password=getpass.getpass('SSH password: '),look_for_keys=False,allow_agent=False,timeout=25)
def run(command):
    _,out,err=c.exec_command(command,timeout=600);text=out.read().decode();error=err.read().decode();code=out.channel.recv_exit_status()
    if code:raise RuntimeError((command,code,text,error))
    return text
try:
    f=c.open_sftp()
    for n in ['software','prepare','offline','freeze','preflight']:
        assert f.open(W+'/'+n+'.exit').read().strip()==b'0'
    g=json.loads(f.open(O+'/GPU_PREFLIGHT.json').read());assert g['status']=='PASS' and g['isolated_optimizer_calls']==4 and g['environment_steps']==0
    print(run('source '+W+'/env.sh && /data/mlr_ifa/env/bin/python '+W+'/launch_once.py'),flush=True)
    snap=run('source '+W+'/env.sh && /data/mlr_ifa/env/bin/python '+W+'/monitor_snapshot.py')
    (R/'launch_snapshot.json').write_text(snap,encoding='utf-8')
    remote=run('source '+W+'/env.sh && /data/mlr_ifa/env/bin/python '+W+'/package_light.py').strip()
    assert remote.startswith(W+'/light_') and '\n' not in remote
    package=json.loads(f.open(remote+'/PACKAGE_MANIFEST.json').read());E=R/'launch_evidence';E.mkdir(exist_ok=False)
    for n,m in package['files'].items():
        p=PurePosixPath(n);assert not p.is_absolute() and '..' not in p.parts
        assert p.suffix in {'.json','.jsonl','.log','.exit','.pid','.started'}
        assert not {'features','buffer','models','recovery','smoke','bindings'}&set(p.parts)
        raw=f.open(remote+'/'+n).read();assert len(raw)==m['bytes'] and hashlib.sha256(raw).hexdigest()==m['sha256']
        dest=E/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    (E/'PACKAGE_MANIFEST.json').write_bytes(f.open(remote+'/PACKAGE_MANIFEST.json').read())
    A=R/'admission_evidence';A.mkdir(exist_ok=True)
    for n in ['software.log','software.exit','prepare.log','prepare.exit','offline.log','offline.exit','freeze.log','freeze.exit','preflight.log','preflight.exit','admission_commands.log','INSTALL_FAILURE_1.json','INSTALL_FAILURE_2.json','DEPLOYMENT.json']:
        (A/n).write_bytes(f.open(W+'/'+n).read())
    print('METADATA_DOWNLOADED',len(package['files']),flush=True)
finally:c.close()
subprocess.run([sys.executable,'-X','utf8',str(R/'verify_launch.py')],check=True)
