from pathlib import Path
import zipfile, xml.etree.ElementTree as ET, json, hashlib
P=Path('C:/Users/Lenovo/Desktop/论文/论文写作/DEBR_Complete_Manuscript_Method_ABC_20261110_消融表更新 - 副本.docx')
OUT=Path(__file__).resolve().parent
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
with zipfile.ZipFile(P) as z:
    root=ET.fromstring(z.read('word/document.xml'))
    rels={r.attrib['Id']:r.attrib.get('Target') for r in ET.fromstring(z.read('word/_rels/document.xml.rels'))}
    blocks=[]
    for i,e in enumerate(root.find('w:body',NS)):
        text=''.join(e.itertext()) if False else ''.join(t.text or '' for t in e.iter() if t.tag in ['{'+NS['w']+'}t','{http://schemas.openxmlformats.org/officeDocument/2006/math}t'])
        imgs=[rels.get(b.attrib.get('{'+NS['r']+'}embed')) for b in e.findall('.//a:blip',NS)]
        if e.tag.endswith('}tbl'):
            text='\n'.join(' | '.join(''.join(c.itertext()) if False else ''.join(t.text or '' for t in c.findall('.//w:t',NS)) for c in row.findall('w:tc',NS)) for row in e.findall('w:tr',NS))
        blocks.append(dict(block=i,kind=e.tag.split('}')[-1],text=text,images=imgs))
    media=[]
    for name in z.namelist():
        if name.startswith('word/media/'):
            p=OUT/'media'/Path(name).name;p.parent.mkdir(exist_ok=True);p.write_bytes(z.read(name));media.append(name)
(OUT/'extracted.txt').write_text('\n\n'.join(f"[{b['block']} {b['kind']}] {b['text']}"+(f"\nIMAGES {b['images']}" if b['images'] else '') for b in blocks),encoding='utf8')
(OUT/'INVENTORY.json').write_text(json.dumps(dict(source=str(P),sha256=hashlib.sha256(P.read_bytes()).hexdigest(),blocks=blocks,media=media),ensure_ascii=False,indent=2),encoding='utf8')
print('Blocks',len(blocks),'media',media)
