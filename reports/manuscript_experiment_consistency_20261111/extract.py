from pathlib import Path
from zipfile import ZipFile
from lxml import etree
import hashlib,json

ROOT=Path(__file__).resolve().parent
SOURCE=Path('C:/Users/Lenovo/Desktop/论文/论文写作')
names=['DEBR_Complete_Manuscript_Method_Rewritten_20261110.docx','DEBR_Complete_Manuscript_CADL_SHBR_20261110.docx']
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
meta={}
for name in names:
    p=SOURCE/name
    with ZipFile(p) as z: root=etree.fromstring(z.read('word/document.xml'))
    def text(el):
        return ''.join(el.xpath('.//w:t/text() | .//m:t/text()',namespaces=ns))
    blocks=[]; paras=0; tables=0
    for i,el in enumerate(root.find('w:body',ns)):
        tag=etree.QName(el).localname
        if tag=='p':
            paras+=1; content=text(el)
            blocks.append({'body_index':i,'kind':'paragraph','index':paras,'text':content})
        elif tag=='tbl':
            tables+=1
            rows=[[text(c) for c in r.findall('w:tc',ns)] for r in el.findall('w:tr',ns)]
            blocks.append({'body_index':i,'kind':'table','index':tables,'rows':rows})
    slug='latest' if 'Rewritten' in name else 'previous'
    (ROOT/(slug+'_extracted.json')).write_text(json.dumps(blocks,ensure_ascii=False,indent=2),encoding='utf8')
    out=[]
    for b in blocks:
        if b['kind']=='paragraph':out.append(f"P{b['index']:03d} [body {b['body_index']}] {b['text']}")
        else:
            out.append(f"TABLE{b['index']} [body {b['body_index']}]")
            out.extend(' | '.join(r) for r in b['rows'])
    (ROOT/(slug+'_extracted.txt')).write_text('\n'.join(out),encoding='utf8')
    meta[slug]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'paragraphs':paras,'tables':tables}
(ROOT/'SOURCE_DOCUMENTS.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(meta,ensure_ascii=False,indent=2))
