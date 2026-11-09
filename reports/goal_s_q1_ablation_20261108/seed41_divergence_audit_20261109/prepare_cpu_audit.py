"""Bind read-only tensor inspection to previously accepted artifact hashes."""
from pathlib import Path
import json,hashlib
A=Path(__file__).resolve().parent;R=A.parent;B=R.parent/'20261029_support_gate_full_s_goal'
spec={'Q1':(R/'final_readonly_20261109_120106/EVIDENCE.json','Q1_41'),'S':(B/'final_readonly_20261030_111546/EVIDENCE.json','seed41')}
files={};expected={};sources={}
for tag,(p,case) in spec.items():
    sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();d=json.loads(p.read_text(encoding='utf8'));files[tag]={}
    for t in range(6,10):
        cp=json.loads(d['light'][case+'/STAGE_S_%d.json'%t]['text']);mem=json.loads(d['light'][case+'/MEMORY_S_%d.json'%t]['text'])
        paths=dict(cp=cp['model'],cache=str(Path(cp['model']).parent/'memory'/('task%d.pth'%t)).replace('\\','/'))
        files[tag][str(t)]=paths;expected[paths['cp']]=cp['sha256'];expected[paths['cache']]=mem['cache_sha256']
body=(A/'cpu_compare_body.py').read_text(encoding='utf8')
(A/'capacity_readonly.py').write_text('FILES='+repr(files)+'\nEXPECTED='+repr(expected)+'\n'+body,encoding='utf8',newline='\n')
(A/'CPU_AUDIT_BINDING.json').write_text(json.dumps(dict(sources=sources,files=files,expected=expected,script_sha256=hashlib.sha256((A/'capacity_readonly.py').read_bytes()).hexdigest(),no_training=True,no_environment=True,no_forward=True),indent=2)+'\n',encoding='utf8')
print('Prepared CPU-only audit of 8 checkpoints and 8 cache files; not executed.')
