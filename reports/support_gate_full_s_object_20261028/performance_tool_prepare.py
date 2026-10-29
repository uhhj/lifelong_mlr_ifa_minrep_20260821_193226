"""Install a standalone profiler outside frozen science code, verify PyPI SHA256."""
import json,urllib.request,hashlib,zipfile,io,subprocess,time,os
from pathlib import Path
D=Path('/data/mlr_ifa/diagnostics/support_gate_performance_20261029_v1')
D.mkdir(parents=True,exist_ok=True)
meta=json.load(urllib.request.urlopen('https://pypi.org/pypi/py-spy/0.4.1/json',timeout=30))
files=[x for x in meta['urls'] if x['filename'].endswith('.whl') and 'manylinux' in x['filename'] and 'x86_64' in x['filename']]
assert len(files)==1,[x['filename'] for x in files]
f=files[0];assert f['url'].startswith('https://files.pythonhosted.org/') and f['size']<20*1024**2
data=urllib.request.urlopen(f['url'],timeout=60).read();assert hashlib.sha256(data).hexdigest()==f['digests']['sha256']
z=zipfile.ZipFile(io.BytesIO(data));names=[n for n in z.namelist() if n.endswith('/scripts/py-spy')];assert len(names)==1
binary=z.read(names[0]);p=D/'py-spy-0.4.1'
if p.exists():assert p.read_bytes()==binary
else:p.write_bytes(binary);p.chmod(0o700)
r=subprocess.run([str(p),'record','--help'],capture_output=True,text=True,timeout=10);assert r.returncode==0
assert '--nonblocking' in r.stdout
receipt=dict(status='STANDALONE_PROFILER_READY',time=time.time(),source='https://pypi.org/pypi/py-spy/0.4.1/json',wheel=f['filename'],wheel_bytes=len(data),wheel_sha256=f['digests']['sha256'],binary=str(p),binary_bytes=len(binary),binary_sha256=hashlib.sha256(binary).hexdigest(),record_help=r.stdout,training_environment_modified=False,scientific_files_changed=0,optimizer_calls=0,environments=0)
(D/'TOOL_RECEIPT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
