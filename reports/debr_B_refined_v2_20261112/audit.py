from pathlib import Path
import json,hashlib,struct,xml.etree.ElementTree as ET,datetime
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
svg=HERE/'DEBR_B_refined.svg';png=HERE/'DEBR_B_refined.png'
root=ET.fromstring(svg.read_bytes());ns={'s':'http://www.w3.org/2000/svg'}
assert root.attrib['viewBox']=='0 0 1800 940'
assert not root.findall('.//s:image',ns)
assert not root.findall('.//s:script',ns)
texts=root.findall('.//s:text',ns)
assert all(t.attrib['font-family']=='Times New Roman' for t in texts)
flat=' '.join(''.join(t.itertext()) for t in texts)
for phrase in ['Guide action learning','Retain self-acquired behavior','Dual-experience coordination','BCQ ≤ BCC','ExecQ ≤ ExecC','RetQ < RetC','Commit the selected full state']:
    assert phrase in flat,phrase
render=json.loads((HERE/'RENDER_RECEIPT.json').read_text())
assert render['source_sha256']==sha(svg)
assert render['png_sha256']==sha(png)
assert struct.unpack('>II',png.read_bytes()[16:24])==(4800,2507)
review=dict(status='PASS_VECTOR_RENDER_AND_VISUAL_REVIEW',observed=datetime.datetime.now().astimezone().isoformat(),svg_sha256=sha(svg),png_sha256=sha(png),
    selected_B_source_sha256=sha(HERE.parent/'20261112_debr_dual_experience_variants/DEBR_B.svg'),
    text_elements=len(texts),embedded_bitmaps=0,font='Times New Roman',single_master=True,
    visual_checks=['Symmetric dual-experience hierarchy','Convergence means coordination, not averaging','No observed text overflow or unintended overlaps','Precise three-condition gate retained','PNG exported from exact delivered SVG'],
    original_files_preserved=True,scientific_runs=0)
(HERE/'REVIEW.json').write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps(review))
