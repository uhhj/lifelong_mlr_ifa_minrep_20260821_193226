from pathlib import Path
import hashlib,json,struct,xml.etree.ElementTree as ET,datetime

HERE=Path(__file__).resolve().parent
ns={'s':'http://www.w3.org/2000/svg'}
variants={}
for v in 'ABC':
    p=HERE/f'DEBR_{v}.svg'; raw=p.read_bytes(); root=ET.fromstring(raw)
    assert root.attrib['viewBox']=='0 0 1800 940'
    assert not root.findall('.//s:image',ns)
    texts=root.findall('.//s:text',ns)
    assert all(t.attrib.get('font-family')=='Times New Roman' for t in texts)
    assert not root.findall('.//s:script',ns)
    flat=' '.join(''.join(t.itertext()) for t in texts)
    for required in ['CADL','SHBR','C','Q','≤','<','RA-Coreset']:
        assert required in flat,(v,required)
    images={}
    for suffix,expected in [('preview',(1800,940)),('refined',(3600,1880)),('comparison',(3660,1040))]:
        q=HERE/f'DEBR_{v}_{suffix}.png'; b=q.read_bytes()
        assert b[:8]==b'\x89PNG\r\n\x1a\n'
        size=struct.unpack('>II',b[16:24]); assert size==expected
        images[suffix]=dict(pixels=size,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
    variants[v]=dict(svg_sha256=hashlib.sha256(raw).hexdigest(),native_text_elements=len(texts),embedded_rasters=0,images=images)

source=HERE.parent/'20261111_manuscript_figure_review/extracted.txt'
review=dict(status='PASS_VECTOR_AND_VISUAL_REVIEW',observed=datetime.datetime.now().astimezone().isoformat(),
    authoring='Native SVG master; refined PNG is a 2x export of the identical master, not an independent AI-generated image.',
    comparison='Left: SVG raster preview. Right: downsampled high-resolution PNG. Both panels derive from the same SVG.',
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),variants=variants,
    semantic_review=['Complementary expert and autonomous-success experience','CADL execution supervision and stage-end RA-Coreset','SHBR fixed references, task support and fixed retention denominator','Same-state same-main-batch C/Q comparison','BC <= C; execution <= C; retention strictly < C; all pass selects Q else C','Stage-end memories used in next task; original inference at deployment'],
    visual_review=['All final previews inspected','A source-label widths and Q-arrow input/output separation corrected','B balanced wings independently reviewed','C long subtitles and independent formula positions corrected','No observed text cropping or unintended overlap'],
    reviewers=['root','variant_b independent A/B reviewer','variant_c C visual correction'],
    experiments_modified=False,word_modified=False,prior_artifacts_preserved=True)
(HERE/'REVIEW.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({'status':review['status'],'variants':list(variants),'PNG_size':[3600,1880],'embedded_rasters':0}))
