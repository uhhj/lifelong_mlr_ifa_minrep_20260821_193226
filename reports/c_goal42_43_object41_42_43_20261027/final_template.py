"""Read-only aggregate final CPU audits; no training, model writes or environments."""
import subprocess,json,hashlib,time,os
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/CCBP_C_GOAL42_43_OBJECT41_42_43_20261027_v1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
c=json.loads((Q/'CONTRACT.json').read_text());assert all(sha(p)==h for p,h in c['files'].items())
assert (Q/'controller.exit').read_text()=='0'
for p in Path('/proc').glob('[0-9]*'):
 try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
 except OSError:continue
 assert int(p.name)==os.getpid() or str(Q) not in cmd
results={};diagnostics={};light={}
for case,script in SCRIPTS.items():
 p=subprocess.run(['/data/mlr_ifa/env/bin/python','-B','-'],input=script,text=True,capture_output=True);assert p.returncode==0,(case,p.stderr)
 v=json.loads(p.stdout)
 for k,rows in v.pop('diagnostics').items():diagnostics[case+'_'+k]=rows
 for k,x in v.pop('light').items():light[case+'/'+k]=x
 results[case]=v
for n in ['STATUS.json','controller.exit']:
 p=Q/n;light[n]=dict(text=p.read_text(),sha256=sha(p))
assert all(sha(p)==h for p,h in c['files'].items())
print(json.dumps(dict(status='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT',observed=time.time(),contract_sha256=sha(Q/'CONTRACT.json'),children=c['children'],source_files=len(c['files']),light=light,processes=[],cases=results,diagnostics=diagnostics)))
