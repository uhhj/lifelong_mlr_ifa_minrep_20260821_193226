from pathlib import Path
import json
p=Path(__file__).resolve().parent
d=json.loads((p/'INVENTORY.json').read_text(encoding='utf8'))
(p/'REVIEW_AUDIT.json').write_text(json.dumps(dict(status='COMPLETE_CONTENT_AND_EMBEDDED_FIGURE_REVIEW',source=d['source'],source_sha256=d['sha256'],body_blocks=len(d['blocks']),embedded_figures=2,tables=sum(b['kind']=='tbl' for b in d['blocks']),docx_modified=False,page_layout_reviewed=False,new_experiments=0),ensure_ascii=False,indent=2),encoding='utf8')
s=(p.parent/'20261111_gate_figure_horizontal/publish.py').read_text(encoding='utf8')
s=s.replace('gate_figure_horizontal_20261111','manuscript_figure_review_20261111')
a=s.index('audit = ')
b=s.index('manifest = {}')
s=s[:a]+"files = ['inspect_doc.py', 'prepare_publication.py', 'publish.py', 'REPORT.md', 'REVIEW_AUDIT.json']\n"+s[b:]
s=s.replace('Restyle gate figure in Times New Roman with horizontal panels and bottom legend','Review DEBR manuscript figure needs and prioritize framework and learning curves')
(p/'publish.py').write_text(s,encoding='utf8')
