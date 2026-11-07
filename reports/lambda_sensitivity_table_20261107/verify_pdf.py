"""Check the compiled table and render a cropped, high-resolution preview."""
from pathlib import Path
import hashlib
import json
import re
import pypdfium2 as pdfium
from pypdf import PdfReader
from PIL import ImageChops

ROOT = Path(__file__).resolve().parent
pdf = ROOT / 'lambda_sensitivity.pdf'
reader = PdfReader(pdf)
assert len(reader.pages) == 1
text = reader.pages[0].extract_text()
assert 'LIBERO-OBJECT' in text and 'LIBERO-GOAL' in text
assert 'LIBERO-50' not in text
assert '??' not in text
assert not re.search(r'\[\d', text)
source=(ROOT/'lambda_sensitivity.tex').read_text(encoding='utf8')
assert r'\cite{' not in source and r'\begin{thebibliography}' not in source
assert not re.search(r'^\s*(C|Q|S)\s*&', source, re.M)
assert source.count(r'\multirow{3}{*}')==2
assert source.count(r'\bestresult{')==6
assert not r'\rowcolor{' in source
for value in ['0.05', '0.1', '0.2', 'FWT', 'NBT', 'AUC']:
    assert value in text, value
log = (ROOT / 'lambda_sensitivity.log').read_text(encoding='utf8', errors='replace')
assert not re.search(r'Overfull \\[hv]box|undefined references|Citation .* undefined|^!', log, re.M)
doc = pdfium.PdfDocument(str(pdf))
im = doc[0].render(scale=4).to_pil().convert('RGB')
bbox = ImageChops.difference(im, im.copy().point(lambda x: 255)).getbbox()
assert bbox
x0,y0,x1,y1 = bbox
im.crop((max(0,x0-28),max(0,y0-28),min(im.width,x1+28),min(im.height,y1+28))).save(ROOT/'lambda_sensitivity_preview.png', dpi=(288,288))
audit = dict(status='PASS_COMPILED_TEXT_AND_LAYOUT_CHECKS', compiler='Tectonic 0.17.0', pages=1,
             pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
             tex_sha256=hashlib.sha256((ROOT/'lambda_sensitivity.tex').read_bytes()).hexdigest(),
             preview_sha256=hashlib.sha256((ROOT/'lambda_sensitivity_preview.png').read_bytes()).hexdigest(),
             unresolved_citations=False, overfull_boxes=False, visual_review='PENDING')
(ROOT/'COMPILE_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')
print(json.dumps(audit,indent=2))
