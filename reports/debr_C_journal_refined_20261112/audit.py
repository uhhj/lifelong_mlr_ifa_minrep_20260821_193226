"""Read-only checks of the delivered figure and its unmodified image sources."""
from pathlib import Path
import hashlib, json, struct, xml.etree.ElementTree as ET, base64, datetime
HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
source = json.loads((HERE/'ASSET_PROVENANCE.json').read_text(encoding='utf8'))
expected = {r['png_sha256'] for group in ['expert','success'] for r in source[group]}
file = HERE/'DEBR_C_journal.svg'
root = ET.fromstring(file.read_bytes())
ns = {'s':'http://www.w3.org/2000/svg'}
assert root.attrib['viewBox'] == '0 0 1800 860'
assert not root.findall('.//s:script', ns)
texts = root.findall('.//s:text', ns)
assert all(t.attrib.get('font-family') == 'Times New Roman' for t in texts)
flat = ' '.join(''.join(t.itertext()) for t in texts)
for token in ['DUAL EXPERIENCE', 'Learn from demonstrations.',
              'DEBR: Dual-Experience Behavioral Retention', 'Simulation imagery illustrates']:
    assert token not in flat, token
for token in ['BC + IFA +', 'execution supervision', 'Expert memory', 'Success history',
              '(MD)', '(MS)', 'Base Candidate ΩC', 'Retention Candidate ΩQ',
              '(LQ = LC + λRLSHBR)', 'Policy πθk']:
    assert token in flat, token
for token in ['LBC,new(θQ)', 'LBC,new(θC)', 'Enew(θQ)', 'Enew(θC)',
              'Rold(θQ)', 'Rold(θC)']:
    assert token in flat, token
lines = [''.join(t.itertext()) for t in texts]
assert lines.count('≤') == 2 and lines.count('<') == 1
policy = next(t for t in texts if ''.join(t.itertext()).startswith('Policy π'))
subs = [t for t in policy.findall('s:tspan', ns) if t.attrib.get('baseline-shift') == 'sub']
assert any(n.attrib.get('baseline-shift') == 'sub' and ''.join(n.itertext()) == 'k'
           for t in subs for n in t.findall('s:tspan', ns))
images = root.findall('.//s:image', ns)
assert len(images) == 6
actual = set()
for im in images:
    val = im.attrib['href']
    assert val.startswith('data:image/png;base64,')
    actual.add(hashlib.sha256(base64.b64decode(val.split(',',1)[1])).hexdigest())
assert actual == expected
render = json.loads((HERE/'RENDER_RECEIPT.json').read_text())
assert render['single_master'] and sha(file) == render['svg_sha256']
assert sha(HERE/'DEBR_C_journal.png') == render['png_sha256']
for name, size in [('DEBR_C_journal.png',(3600,1720)),
                   ('DEBR_C_journal_preview.png',(1800,860)),
                   ('SVG_PNG_comparison.png',(3660,930))]:
    assert struct.unpack('>II',(HERE/name).read_bytes()[16:24]) == size
doc = Path('C:/Users/Lenovo/Desktop/论文/论文写作/DEBR_Complete_Manuscript_Method_ABC_20261110_消融表更新 - 副本.docx')
assert sha(doc) == 'b6c7809bbc6d47b17c8fdacaf44814a193c10e8c1d22d3ccf06e602af657235e'
review = dict(status='PASS_JOURNAL_FIGURE_REVIEW',
    observed=datetime.datetime.now().astimezone().isoformat(),
    source_images_unchanged=True, source_image_count=6, svg_png_same_master=True,
    height_reduction_percent=14, editable_text_elements=len(texts),
    notation='Latest explicit user specification; older DOCX differences documented separately',
    manuscript_unchanged=True, image_source_context='External caption; different archival tasks',
    visual_review=['Full preview inspected by author and independent reviewer',
                   'No global title banners, formula clipping or overlapping text',
                   'Three gate probes vertically aligned; nested policy subscript visible',
                   'CADL icon raised 7px after review to improve lower padding'],
    new_rollouts=0, new_model_calls=0,
    publication='Code and lightweight notes/hashes only; image-bearing files stay local')
(HERE/'REVIEW.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
manifest={}
for name in ['DEBR_C_journal.svg','DEBR_C_journal.png','DEBR_C_journal_preview.png',
             'SVG_PNG_comparison.png','FIGURE_CAPTION.md','NOTATION_NOTE.md']:
    p=HERE/name
    manifest[name]=dict(bytes=p.stat().st_size,sha256=sha(p),
        publication='LOCAL_ONLY_IMAGE_BEARING_ARTIFACT' if p.suffix in ['.svg','.png'] else 'LIGHTWEIGHT_NOTE')
(HERE/'LOCAL_ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(review,ensure_ascii=False))
