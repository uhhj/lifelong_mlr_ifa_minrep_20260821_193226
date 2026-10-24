import subprocess,sys,os,json,time,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
assert not (R/'supervisor.started').exists()
(R/'supervisor.started').write_text(str(os.getpid()))
c=json.loads((R/'CONTRACT.json').read_text())
for n,h in c['files'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h
with (R/'worker.log').open('xb') as f:
 rc=subprocess.call([sys.executable,'-B',str(R/'paired_worker.py')],cwd=R,stdout=f,stderr=subprocess.STDOUT)
(R/'worker.exit').write_text(str(rc))
(R/'supervisor.exit').write_text(str(rc))
sys.exit(rc)
