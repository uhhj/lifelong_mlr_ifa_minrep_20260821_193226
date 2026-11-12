from pathlib import Path
import hashlib,json,struct,xml.etree.ElementTree as ET,base64,datetime
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=json.loads((HERE/'ASSET_PROVENANCE.json').read_text(encoding='utf8'))
expected={r['png_sha256'] for group in ['expert','success'] for r in source[group]}
render=json.loads((HERE/'RENDER_RECEIPT.json').read_text())
ns={'s':'http://www.w3.org/2000/svg'}
variants={};manifest={}
for v in 'ABC':
    file=HERE/f'DEBR_{v}.svg';raw=file.read_bytes();root=ET.fromstring(raw)
    assert root.attrib['viewBox']=='0 0 1800 1000'
    assert not root.findall('.//s:script',ns)
    txt=root.findall('.//s:text',ns)
    assert all(t.attrib.get('font-family')=='Times New Roman' for t in txt)
    embedded=root.findall('.//s:image',ns);assert len(embedded)==6
    hashes=[]
    for im in embedded:
        val=im.attrib['href'];assert val.startswith('data:image/png;base64,')
        h=hashlib.sha256(base64.b64decode(val.split(',',1)[1])).hexdigest()
        assert h in expected;hashes.append(h)
    assert set(hashes)==expected
    flat=' '.join(''.join(t.itertext()) for t in txt)
    for token in ['CADL','SHBR','≤','<','RA-Coreset','not a paired performance comparison']:
        assert token in flat,(v,token)
    assert sha(file)==render[v]['svg_sha256']
    assert sha(HERE/f'DEBR_{v}_refined.png')==render[v]['png_sha256']
    for suffix,size in [('preview',(1800,1000)),('refined',(3600,2000)),('comparison',(3660,1100))]:
        p=HERE/f'DEBR_{v}_{suffix}.png';assert struct.unpack('>II',p.read_bytes()[16:24])==size
    variants[v]=dict(images=len(embedded),unaltered_source_images=True,editable_text_elements=len(txt),svg_sha256=sha(file),png_sha256=sha(HERE/f'DEBR_{v}_refined.png'))
    for p in [file,*[HERE/f'DEBR_{v}_{s}.png' for s in ['preview','refined','comparison']]]:
        manifest[p.name]=dict(bytes=p.stat().st_size,sha256=sha(p),publication='LOCAL_ONLY_IMAGE_BEARING_ARTIFACT')
review=dict(status='PASS_MIXED_VECTOR_SOURCE_AND_VISUAL_REVIEW',observed=datetime.datetime.now().astimezone().isoformat(),
    variants=variants,source_images='Official demonstrations + archived successful policy rollout; different tasks; illustrative',
    matching='Refined PNG is rendered from the exact delivered SVG, including identical embedded frames',
    visual_review=['All three full previews inspected','Frame sources and representative role labels verified','No photo substituted for a latest DEBR result','Three gate conditions and single-policy commit preserved','Memory reuse and source captions separated from connectors'],
    raw_frame_size=[128,128],frame_enhancement=False,new_rollouts=0,new_model_calls=0,publication='Images, image-bearing SVG/PNG and local gallery withheld; only reviewed code and lightweight provenance published')
(HERE/'REVIEW.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
(HERE/'LOCAL_ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'status':review['status'],'variants':list(variants),'source_images_per_variant':6,'new_rollouts':0}))
