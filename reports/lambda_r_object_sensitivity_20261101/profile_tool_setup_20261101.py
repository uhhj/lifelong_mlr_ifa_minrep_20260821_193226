"""Extract pinned standalone profiler outside training env; do not attach yet."""
import json,hashlib,io,zipfile,urllib.request,subprocess,os,time
from pathlib import Path
base=Path('/data/mlr_ifa/diagnostics/lambda_r_profile_20261101_v1');base.mkdir(parents=True,exist_ok=True)
url='https://files.pythonhosted.org/packages/f9/34/dd7d3c763a00b7b965e25a5eab0acd1a345dbaf0f45fffe595278873a1c0/py_spy-0.4.2-py2.py3-none-manylinux_2_5_x86_64.manylinux1_x86_64.whl'
expected='aeb0323409199c785f730645e9f4bb7a7b9ca2c481f2c331a55642b5d13fa52f'
v=os.statvfs('/data');assert v.f_bavail*v.f_frsize>15*1024**3
raw=urllib.request.urlopen(url,timeout=45).read(4*1024**2);assert hashlib.sha256(raw).hexdigest()==expected
z=zipfile.ZipFile(io.BytesIO(raw));names=[n for n in z.namelist() if n.endswith('/py-spy')];assert len(names)==1
binary=z.read(names[0]);target=base/'py-spy-0.4.2'
if target.exists():assert target.read_bytes()==binary
else:target.write_bytes(binary);target.chmod(0o755)
p=subprocess.run([str(target),'record','--help'],capture_output=True,text=True,timeout=10)
assert p.returncode==0 and '--nonblocking' in p.stdout and '--idle' in p.stdout
print(json.dumps(dict(status='PASS_STANDALONE_TOOL_ONLY_NO_TARGET_ACCESS',observed=time.time(),url=url,wheel_sha256=expected,binary_sha256=hashlib.sha256(binary).hexdigest(),binary_bytes=len(binary),path=str(target),help=p.stdout,training_environment_modified=False,target_signals=0,optimizer_calls=0,environments=0)))
