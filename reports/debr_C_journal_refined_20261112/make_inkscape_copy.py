"""Create an editable Inkscape copy using SVG 1.1 shadow primitives."""
from pathlib import Path
import hashlib, json, xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent
src=HERE/'DEBR_C_journal.svg'
dst=HERE/'DEBR_C_journal_inkscape.svg'
original=src.read_bytes()
old='<feDropShadow dx="0" dy="4" stdDeviation="7" flood-color="#17364D" flood-opacity=".07"/>'
new='''<feGaussianBlur in="SourceAlpha" stdDeviation="7" result="shadowBlur"/>
<feOffset in="shadowBlur" dx="0" dy="4" result="shadowOffset"/>
<feFlood flood-color="#17364D" flood-opacity=".07" result="shadowColor"/>
<feComposite in="shadowColor" in2="shadowOffset" operator="in" result="shadowComposite"/>
<feMerge><feMergeNode in="shadowComposite"/><feMergeNode in="SourceGraphic"/></feMerge>'''
text=original.decode('utf8')
assert text.count(old)==1
updated=text.replace(old,new)
assert updated.replace(new,old)==text
ET.fromstring(updated)
dst.write_text(updated,encoding='utf8')
assert src.read_bytes()==original
r=dict(source_sha256=hashlib.sha256(original).hexdigest(),
       compatible_sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),
       change='Replace feDropShadow with equivalent SVG 1.1 filter primitives',
       layout_text_images_unchanged=True,original_untouched=True,
       executable='D:/APP/Inkscape/bin/inkscape.com',version='1.4.4',
       publication='Image-bearing SVG and rendered PNG stay local')
(HERE/'INKSCAPE_COMPATIBILITY.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
