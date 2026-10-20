"""Download only read-only audit JSON metadata."""
from pathlib import Path
import paramiko,getpass,json,hashlib
R=Path(__file__).resolve().parent
A="/data/mlr_ifa/analyses/OVR_OBJECT_seed42_history_readonly_20261020"
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
c.connect("jq1.9gpu.com",port=11920,username="root",password=getpass.getpass("SSH password (not saved): "),look_for_keys=False,allow_agent=False,timeout=25)
try:
    receipt={}
    with c.open_sftp() as s:
        for n in ("RNG_STATE_AUDIT.json","CONFIG_AUDIT.json","INITIAL_AUDIT.json","SOURCE_EVIDENCE.json","OUTCOMES.json","INPUT_MANIFEST.json","READONLY_ACCEPTANCE.json"):
            remote=A+"/"+n;assert s.stat(remote).st_size<8*1024**2
            with s.open(remote,"rb") as f:raw=f.read()
            json.loads(raw);(R/n).write_bytes(raw)
            receipt[n]=dict(source=remote,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    (R/"DOWNLOAD_MANIFEST.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("DOWNLOADED",len(receipt),"metadata files")
finally:c.close()
