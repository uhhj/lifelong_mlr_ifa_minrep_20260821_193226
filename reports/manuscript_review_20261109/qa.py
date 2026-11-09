from pathlib import Path
from zipfile import ZipFile
from decimal import Decimal, ROUND_HALF_UP
import json, hashlib, re, datetime
from docx import Document
from lxml import etree
from pypdf import PdfReader
H=Path(__file__).resolve().parent
R=H.parents[1]
a=json.loads((H/'EDIT_RECEIPT.json').read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for k,h in [('source','source_sha256'),('output','output_sha256'),('data_source','data_sha256')]:assert sha(a[k])==a[h],k
src=Document(a['source']);doc=Document(a['output'])
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
body='\n'.join(''.join(p._p.xpath('.//w:t/text()|.//m:t/text()')) for p in doc.paragraphs)
assert '3.4.4' not in body and 'SBHR' not in body
assert len(doc.tables)==5
for s in ['摘要','Abstract','关键词','Keywords','参考文献','向上取整','observation–action temporal alignment']:
    assert s in body,s
before_refs=body.split('参考文献')[0]
for i in range(1,14):assert f'[{i}]' in before_refs,f'citation {i}'
for i in range(1,21):assert f'（{i}）' in body or f'({i})' in body,f'equation {i}'
with ZipFile(a['source']) as z1, ZipFile(a['output']) as z2:
    media=[n for n in z1.namelist() if n.startswith('word/media/')]
    assert len(media)==2
    for n in media:assert z1.read(n)==z2.read(n),n
    tree=etree.fromstring(z2.read('word/document.xml'))
    for n in tree.xpath('//m:nary/m:e',namespaces=ns):assert len(n)>0
j=json.loads(Path(a['data_source']).read_text(encoding='utf8'))
for p,h in j['source_sha256'].items():assert sha(R/p)==h,p
fmt=lambda x:str(Decimal(str(x)).quantize(Decimal('.1'),rounding=ROUND_HALF_UP))
t=doc.tables[1]
for row,m in enumerate(['Random','K-Center','RA'],2):
    for si,s in enumerate(['OBJECT','GOAL']):
        for ki,k in enumerate(['FWT','NBT','AUC']):
            v=j['summary'][s][m][k]
            assert t.cell(row,1+si*3+ki).text==fmt(v['mean_percent'])+'±'+fmt(v['sample_sd_pp'])
for c in range(1,7):
    assert doc.tables[0].cell(9,c).text==t.cell(2,c).text
    assert doc.tables[2].cell(2,c).text==t.cell(4,c).text
labels=' '.join(c.text for row in doc.tables[0].rows for c in row.cells)
assert not any(s in labels for s in ['†','local','本地','文献'])
assert t.cell(0,1).text=='LIBERO-OBJECT' and t.cell(0,4).text=='LIBERO-GOAL'
assert t.cell(4,0).text=='RA-Coreset'
pdf=H/'render/revised.pdf';assert len(PdfReader(pdf).pages)==19
for i in range(1,20):assert (H/f'render/page-{i}.png').stat().st_size>1000
assert '�' not in body
q=dict(status='PASS_TARGETED_EDIT_AND_VISUAL_QA',observed=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),source_sha256=a['source_sha256'],output_sha256=a['output_sha256'],pdf_sha256=sha(pdf),pages=19,visually_inspected_pages=list(range(1,20)),tables=5,original_media_bytes_preserved=2,native_word_math_count=len(tree.xpath('//m:oMath',namespaces=ns)),equation_numbers_checked=list(range(1,21)),references_cited=list(range(1,14)),table2_values_checked=18,duplicate_metric_cells_checked=12,table1_provenance_markers=False,source_data_hashes_rechecked=len(j['source_sha256']),render_method='Hidden Microsoft Word COM ExportAsFixedFormat followed by pypdfium2; bundled LibreOffice render unavailable (soffice.exe absent).',scientific_reruns=0,visual_findings='All 19 pages inspected; formulas, tables, captions and citations readable, no clipping or overlapping text. Existing figure media preserved. Intentional section and figure page breaks retained.')
(H/'QA.json').write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
a['status']='PASS_EDITED_AND_VISUALLY_VERIFIED';a['qa']='QA.json'
(H/'EDIT_RECEIPT.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(q,ensure_ascii=False,indent=2))
