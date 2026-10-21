import getpass,json,shlex,datetime
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
SCRIPT=r'''
import os,json,time,zlib,hashlib,struct
from pathlib import Path
import numpy as np
os.nice(15)
O=Path('/data/mlr_ifa/experiments/CAET_v2_OBJECT_seed41_20261021')
def shuffle4(b):
 q=len(b)//4*4
 return np.frombuffer(b[:q],dtype=np.uint8).reshape(-1,4).T.tobytes()+b[q:]
def unshuffle4(b):
 q=len(b)//4*4
 return np.frombuffer(b[:q],dtype=np.uint8).reshape(4,-1).T.tobytes()+b[q:]
def probe(p):
 before=p.stat();a=hashlib.sha256();b=hashlib.sha256();n=packed=0
 with p.open('rb') as f:
  for raw in iter(lambda:f.read(4*1024**2),b''):
   c=zlib.compress(shuffle4(raw),6);restored=unshuffle4(zlib.decompress(c));assert restored==raw
   a.update(raw);b.update(restored);n+=len(raw);packed+=len(c)+8
 assert p.stat().st_mtime_ns==before.st_mtime_ns and n==before.st_size and a.hexdigest()==b.hexdigest()
 return dict(path=str(p),bytes=n,codec_bytes=packed,source_sha256=a.hexdigest(),restored_sha256=b.hexdigest(),byte_exact=True)
begin=time.monotonic();paths=[O/'preflight/recovery.pth']+sorted((O/'ra_coreset/buffer').glob('*.pt'))[:100]
rows=[probe(p) for p in paths]
print(json.dumps(dict(status='READ_ONLY_BYTE_EXACT_CODEC_PROBE_PASS',server_writes=0,model_forwards=0,optimizer_updates=0,environment_calls=0,seconds=time.monotonic()-begin,codec='4-byte plane transpose per 4MiB block + zlib6; inverse byte exact; no numerical casts',selection='one own immutable completed preflight recovery and first100 lexicographic own new buffer files',files=rows,original_bytes=sum(r['bytes'] for r in rows),codec_bytes=sum(r['codec_bytes'] for r in rows),does_not_authorize_deleting_active_payload=True)))
'''
dest=R/('storage_codec_probe_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));dest.mkdir()
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
try:
 c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
 i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(SCRIPT),timeout=180);raw=o.read();err=e.read();rc=o.channel.recv_exit_status();(dest/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err.decode()),indent=2),encoding='utf8');assert rc==0,err.decode();(dest/'SNAPSHOT.json').write_bytes(raw)
 print(str(dest))
 print('INSPECT_OK')
finally:pw=None;c.close()
