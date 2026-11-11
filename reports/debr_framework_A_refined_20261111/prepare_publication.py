from pathlib import Path
import hashlib,json
p=Path(__file__).resolve().parent
f=p/'DEBR_A_refined.png'
(p/'REVIEW.json').write_text(json.dumps(dict(status='REFINED_RASTER_REVIEWED',image_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),preserves_original=True,gate_conditions=['BC_Q <= BC_C','Exec_Q <= Exec_C','Ret_Q < Ret_C'],new_experiments=0,editable_vector=False),indent=2)+'\n',encoding='utf8')
s=(p.parent/'20261111_gate_figure_horizontal/publish.py').read_text(encoding='utf8')
s=s.replace('gate_figure_horizontal_20261111','debr_framework_A_refined_20261111')
a=s.index('audit = ')
b=s.index('manifest = {}')
s=s[:a]+"files = ['DEBR_A_refined.png', 'PROMPTS.json', 'README.md', 'REVIEW.json', 'prepare_publication.py', 'publish.py', 'make_svg.py', 'render_svg.cjs', 'DEBR_A_editable.svg', 'DEBR_A_editable_preview.png', 'VECTOR_REVIEW.json']\n"+s[b:]
s=s.replace("['.py', '.md', '.json', '.svg']", "['.py', '.cjs', '.md', '.json', '.svg']")
s=s.replace('Restyle gate figure in Times New Roman with horizontal panels and bottom legend','Refine selected DEBR dual-experience framework A and clarify memory and gate flows')
(p/'publish.py').write_text(s,encoding='utf8')
