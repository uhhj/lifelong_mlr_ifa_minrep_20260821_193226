"""Read progress and verified lightweight package. No restart actions."""
import json, getpass, datetime, hashlib, tarfile, io
from pathlib import Path, PurePosixPath
import paramiko
from audit_download import main as audit
R=Path(__file__).resolve().parent;N='/data/mlr_ifa/experiment_workspaces/ZRv3_OBJECT_RA_seed41_20261020'
D=R/('progress_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));D.mkdir()
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
c.connect('jq1.9gpu.com',port=11920,username='root',password=getpass.getpass('SSH password (not saved): '),look_for_keys=False,allow_agent=False,timeout=25)
commands=[]
def run(command):
    _,a,b=c.exec_command(command,timeout=600);a.channel.set_combine_stderr(True)
    raw=a.read().decode();rc=a.channel.recv_exit_status();commands.append(dict(command=command,exit_code=rc))
    assert rc==0,raw
    return raw
try:
    snap=json.loads(run('/data/mlr_ifa/env/bin/python '+N+'/monitor_snapshot.py'))
    package=json.loads(run('/data/mlr_ifa/env/bin/python '+N+'/package_light.py'))
    with c.open_sftp() as s:raw=s.open(N+'/light.tar.gz').read()
    assert hashlib.sha256(raw).hexdigest()==package['archive_sha256']
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as t:
        names=t.getnames();assert len(names)==len(set(names))
        manifest=json.loads(t.extractfile('PACKAGE_MANIFEST.json').read())
        assert set(names)==set(manifest)|{'PACKAGE_MANIFEST.json'}
        for name in names:
            p=PurePosixPath(name);assert not p.is_absolute() and '..' not in p.parts
            assert p.suffix in ('.json','.jsonl','.exit','.log','.md','.txt','.py','.yaml')
            assert not {'models','features','buffer','recovery'}&set(p.parts)
            assert t.getmember(name).isfile()
            b=t.extractfile(name).read()
            if name!='PACKAGE_MANIFEST.json':assert len(b)==manifest[name]['bytes'] and hashlib.sha256(b).hexdigest()==manifest[name]['sha256']
            dest=D/'evidence'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
    (D/'SNAPSHOT.json').write_text(json.dumps(snap,indent=2)+'\n')
    (D/'DOWNLOAD_RECEIPT.json').write_text(json.dumps(package,indent=2)+'\n')
    result=audit(D)
    (R/'RUN_STATUS.json').write_text(json.dumps(dict(status='COMPLETED' if result['status']=='PASS_FINAL' else 'PARTIAL',time=datetime.datetime.now().astimezone().isoformat(),evidence=D.name,variants=snap['variants']),indent=2)+'\n')
    print('EVIDENCE',D);print(json.dumps(snap,indent=2))
finally:
    (D/'COMMANDS.json').write_text(json.dumps(commands,indent=2)+'\n');c.close()
