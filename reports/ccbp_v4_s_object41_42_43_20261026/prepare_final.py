"""External final wrapper composition. Never alters frozen server source."""
from pathlib import Path
R=Path(__file__).resolve().parent
scripts={str(seed):(R/f'final_seed{seed}_readonly.py').read_text(encoding='utf8') for seed in (41,42,43)}
prefix='''import subprocess,json,hashlib,time
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/CCBP_V4_S_OBJECT41_42_43_20261026_v1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
c=json.loads((Q/'CONTRACT.json').read_text());assert all(sha(p)==h for p,h in c['files'].items())
assert (Q/'controller.exit').read_text()=='0'
'''
suffix='''
results={};diagnostics={};light={}
for seed,script in SCRIPTS.items():
 p=subprocess.run(['/data/mlr_ifa/env/bin/python','-B','-'],input=script,text=True,capture_output=True);assert p.returncode==0,(seed,p.stderr)
 v=json.loads(p.stdout)
 for k,rows in v.pop('diagnostics').items():diagnostics['s'+seed+'_'+k]=rows
 for k,x in v.pop('light').items():light['seed'+seed+'/'+k]=x
 results[seed]=v
for n in ['STATUS.json','controller.exit']:
 p=Q/n;light[n]=dict(text=p.read_text(),sha256=sha(p))
assert all(sha(p)==h for p,h in c['files'].items())
print(json.dumps(dict(status='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT',observed=time.time(),contract_sha256=sha(Q/'CONTRACT.json'),children=c['children'],source_files=len(c['files']),light=light,processes=[],seeds=results,diagnostics=diagnostics)))
'''
(R/'final_readonly.py').write_text(prefix+'SCRIPTS='+ascii(scripts)+'\n'+suffix,encoding='utf8')
print('External final CPU wrapper composed; still awaits real final validation.')
