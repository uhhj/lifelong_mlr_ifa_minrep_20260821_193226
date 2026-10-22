"""Resume only missing reviewed metadata; independent SFTP channels, no remote writes."""
import concurrent.futures
import datetime
import getpass
import hashlib
import json
import sys
from pathlib import Path
import paramiko

p=Path(sys.argv[1]).resolve()
manifest=json.loads((p/'PACKAGE_MANIFEST.json').read_text(encoding='utf8'))
assert manifest['status']=='LIGHT_MANIFEST_READY' and not manifest['residual_workers']
pending=[]
for name,item in manifest['entries'].items():
    target=(p/name).resolve()
    assert p in target.parents and target.suffix in ('.json','.jsonl','.exit')
    assert item['bytes']<8*1024**2
    assert str(item['path']).startswith(('/data/mlr_ifa/experiments/','/root/mlr_ifa/experiments/','/data/mlr_ifa/experiment_workspaces/'))
    if target.is_file():
        raw=target.read_bytes()
        assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
    else:
        pending.append((target,item))
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
pw=getpass.getpass('SSH password (memory only): ')
try:
    c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,look_for_keys=False,allow_agent=False,timeout=20)
    pw=None
    def download_batch(batch):
        s=c.open_sftp()
        try:
            for target,item in batch:
                with s.open(item['path'],'rb') as remote:
                    raw=remote.read()
                assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(raw)
        finally:
            s.close()
        return len(batch)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        completed=sum(pool.map(download_batch,[pending[i::8] for i in range(8)]))
    for name,item in manifest['entries'].items():
        raw=(p/name).read_bytes()
        assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
    result=dict(status='LIGHT_DOWNLOAD_RESUMED_VERIFIED',files=len(manifest['entries']),
        resumed_files=completed,manifest_sha256=hashlib.sha256((p/'PACKAGE_MANIFEST.json').read_bytes()).hexdigest(),
        time=datetime.datetime.now().astimezone().isoformat(),server_writes=0,experiment_restarts=0)
    (p/'DOWNLOAD_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))
finally:
    pw=None;c.close()
