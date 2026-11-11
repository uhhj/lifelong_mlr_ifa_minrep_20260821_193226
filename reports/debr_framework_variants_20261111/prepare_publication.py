from pathlib import Path
import json, hashlib
p=Path(__file__).resolve().parent
d=json.loads((p/'PROMPTS.json').read_text(encoding='utf8'))
for e in d['outputs']:
    e['source']=e['source'].split(' as ')[-1]
    e['sha256']=hashlib.sha256((p/e['file']).read_bytes()).hexdigest()
(p/'PROMPTS.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
s=(p.parent/'20261111_gate_figure_horizontal/publish.py').read_text(encoding='utf8')
s=s.replace('gate_figure_horizontal_20261111','debr_framework_variants_20261111')
a=s.index('audit = ')
b=s.index('manifest = {}')
s=s[:a]+"files = ['DEBR_A.png', 'DEBR_B.png', 'DEBR_C.png', 'PROMPTS.json', 'README.md', 'prepare_publication.py', 'publish.py']\n"+s[b:]
s=s.replace('Restyle gate figure in Times New Roman with horizontal panels and bottom legend','Add three DEBR framework concept variants for author selection')
(p/'publish.py').write_text(s,encoding='utf8')
