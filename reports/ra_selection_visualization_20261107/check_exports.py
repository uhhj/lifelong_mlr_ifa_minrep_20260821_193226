"""Check existing CSV aggregates, source hashes, PDF embedding and render exports."""
from pathlib import Path
import json,csv,hashlib,math
from pypdf import PdfReader
import pypdfium2 as pdfium

R=Path(__file__).resolve().parent;W=R.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((R/'SOURCE_MANIFEST.json').read_text(encoding='utf8'))
assert all(sha(W/p)==h for p,h in manifest.items())
rows=list(csv.DictReader((R/'data/all_cases_original_distance.csv').open(encoding='utf8')))
assert len(rows)==24 and len({(r['suite'],r['seed'],r['task']) for r in rows})==24
for r in rows:
    for key in ['mean','radius']:
        assert math.isclose(float(r[key+'_ratio']),float(r[key+'_ra'])/float(r[key+'_kcenter']),rel_tol=1e-14)
coverage=list(csv.DictReader((R/'data/illustrated_trajectory_coverage.csv').open(encoding='utf8')))
windows=list(csv.DictReader((R/'data/illustrated_window_membership.csv').open(encoding='utf8')))
for r in coverage:
    ss=[x for x in windows if x['suite']==r['suite'] and x['demo']==r['demo']]
    field={'Random':'random','K-Center':'kcenter','RA-Coreset':'ra'}[r['method']]
    assert len(ss)==int(r['candidates']) and sum(int(x[field]) for x in ss)==int(r['selected'])
    assert math.isclose(float(r['fraction']),int(r['selected'])/len(ss),rel_tol=1e-14)
out=R/'review';out.mkdir(exist_ok=True)
pdfs=[]
for item in json.loads((R/'EXPORT_AUDIT.json').read_text())['figures']:
    p=R/item['file'];assert sha(p)==item['sha256']
    if p.suffix!='.pdf':continue
    reader=PdfReader(p);assert len(reader.pages)==1
    pg=reader.pages[0];fonts=[]
    for ref in pg['/Resources']['/Font'].values():
        f=ref.get_object();assert f['/Subtype']!='/Type3'
        if '/DescendantFonts' in f:
            for df in f['/DescendantFonts']:
                desc=df.get_object()['/FontDescriptor'].get_object();assert any(k in desc for k in ['/FontFile','/FontFile2','/FontFile3'])
        fonts.append(str(f['/BaseFont']))
    doc=pdfium.PdfDocument(str(p));page=doc[0];page.render(scale=2).to_pil().save(out/(p.stem+'.png'))
    text=page.get_textpage().get_text_range();assert 'OBJECT' in text and 'GOAL' in text
    pdfs.append(dict(file=item['file'],pages=1,fonts=fonts,width_points=float(pg.mediabox.width),height_points=float(pg.mediabox.height),render=str((out/(p.stem+'.png')).relative_to(R))))
distance_source=W/'execution/20261019_ra_object_conditional/source/libero/lifelong/selectors/greedy_kcenter.py'
selector_source=W/'execution/20261019_ra_object_conditional/source/ra_exp/selector_reference.py'
res=dict(status='PASS_SOURCE_CSV_AND_PDF_CHECKS',cases=24,illustrated_windows=len(windows),raw_distance_recomputed=False,pdfs=pdfs,implementation_sha256={p.relative_to(W).as_posix():sha(p) for p in [distance_source,selector_source]})
(R/'RECOMPUTATION_AUDIT.json').write_text(json.dumps(res,indent=2)+'\n',encoding='utf8')
print(json.dumps(res,indent=2))
