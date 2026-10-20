import getpass,json,hashlib,datetime
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
A="/data/mlr_ifa/analyses/CZR_A_OBJECT_seed41_20261020_generalization"
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
c.connect("jq1.9gpu.com",port=11920,username="root",password=getpass.getpass("SSH password (not saved): "),
          allow_agent=False,look_for_keys=False,timeout=25)
try:
    with c.open_sftp() as s:
        names=["DIAGNOSTIC.json","QUERY_METADATA.json","CENTER_NEIGHBORS.json","SAME_DEMO_NEIGHBORS.json"]
        manifest={}
        for n in names:
            with s.open(A+"_v2/"+n,"rb") as f:raw=f.read()
            (R/n).write_bytes(raw);manifest[n]=hashlib.sha256(raw).hexdigest()
        for suffix in (".log",".exit","_v2.log","_v2.exit"):
            s.get(A+suffix,str(R/("remote"+suffix)))
    _,o,e=c.exec_command("sha256sum "+A+"_v2/*.json",timeout=30)
    check=o.read().decode();err=e.read().decode();rc=o.channel.recv_exit_status();assert rc==0,err
    remote={Path(l.split()[1]).name:l.split()[0] for l in check.splitlines()}
    assert manifest==remote
    receipt=dict(status="PASS",files=manifest,original_exit=int((R/"remote.exit").read_text()),
                 repaired_exit=int((R/"remote_v2.exit").read_text()),
                 time=datetime.datetime.now().astimezone().isoformat())
    assert receipt["original_exit"]==1 and receipt["repaired_exit"]==0
    (R/"DOWNLOAD_ACCEPTANCE.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))
finally:c.close()
