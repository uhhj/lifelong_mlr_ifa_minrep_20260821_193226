"""Independent scalar/ECDF recomputation and PDF export checks; no model calls."""
from pathlib import Path
import json,csv,hashlib,math,statistics
from pypdf import PdfReader
import pypdfium2 as pdfium
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
b=json.loads((R/'DISTANCE_LOCAL_BINDING.json').read_text());P=R/b['directory']
assert sha(P/'EVIDENCE.json')==b['evidence_sha256'] and sha(R/'distance_readonly.py')==b['request_sha256']
ev=json.loads((P/'EVIDENCE.json').read_text());assert ev['status']=='PASS_ALL_24_ORIGINAL_DISTANCE_RECOMPUTATIONS'
assert len(ev['results'])==24
for p,entry in ev['diagnostic_files'].items():assert sha(P/p)==entry['sha256']
stats=list(csv.DictReader((R/'data_distance/nearest_distance_summary.csv').open(encoding='utf8')))
ecdf=list(csv.DictReader((R/'data_distance/nearest_distance_ecdf.csv').open(encoding='utf8')))
assert len(stats)==72
fields={'Random':'random_distance','K-Center':'kcenter_distance','RA-Coreset':'ra_distance'}
for s in stats:
    key=f'{s["suite"]}_{s["seed"]}_{s["task"]}'
    rows=json.loads((P/f'diagnostics/nearest_{key}.json').read_text());a=sorted(r[fields[s['method']]] for r in rows);n=len(a)
    assert n==int(s['N'])
    calc={'mean':statistics.fmean(a),'maximum':a[-1],'zero_fraction':sum(x==0 for x in a)/n}
    for pct in [50,90,95,99]:
        at=(n-1)*pct/100;lo=int(at);hi=min(lo+1,n-1);calc['p'+str(pct)]=a[lo]+(a[hi]-a[lo])*(at-lo)
    assert all(math.isclose(v,float(s[k]),rel_tol=1e-12,abs_tol=1e-12) for k,v in calc.items())
    er=[r for r in ecdf if (r['suite'],r['seed'],r['task'],r['method'])==(s['suite'],s['seed'],s['task'],s['method'])]
    cursor=0;prev=-1
    for r in er:
        value=float(r['distance']);count=int(r['count_at_distance']);assert value>prev
        assert all(x==value for x in a[cursor:cursor+count]);cursor+=count;prev=value
        assert math.isclose(float(r['fraction']),cursor/n,abs_tol=1e-15)
    assert cursor==n and float(er[-1]['fraction'])==1
summary=json.loads((R/'DISTANCE_SUMMARY.json').read_text())
for suite in ['OBJECT','GOAL']:
    cases=[c for c in ev['results'] if c['suite']==suite]
    for comp in ['Random','K-Center']:
        for metric in ['mean','p95','maximum']:
            changes=[100*(c['arms']['RA-Coreset'][metric]/c['arms'][comp][metric]-1) for c in cases]
            reported=summary[suite]['RA_vs_'+comp][metric]
            assert math.isclose(statistics.fmean(changes),reported['mean_paired_change_percent'],abs_tol=1e-12)
            assert [min(changes),max(changes)]==reported['change_percent_range']
out=R/'review_distance';out.mkdir(exist_ok=True);pdfs=[]
for x in json.loads((R/'DISTANCE_EXPORT_AUDIT.json').read_text())['figures']:
    p=R/x['file'];assert sha(p)==x['sha256']
    if p.suffix!='.pdf':continue
    reader=PdfReader(p);assert len(reader.pages)==1;fonts=[]
    for ref in reader.pages[0]['/Resources']['/Font'].values():
        f=ref.get_object();assert f['/Subtype']!='/Type3'
        for df in f.get('/DescendantFonts',[]):
            desc=df.get_object()['/FontDescriptor'].get_object();assert any(k in desc for k in ['/FontFile','/FontFile2','/FontFile3'])
        fonts.append(str(f['/BaseFont']))
    doc=pdfium.PdfDocument(str(p));page=doc[0];page.render(scale=2).to_pil().save(out/(p.stem+'.png'))
    pdfs.append(dict(file=x['file'],sha256=x['sha256'],embedded_fonts=fonts,pages=1))
result=dict(status='PASS_REMOTE_HASHES_SCALARS_ECDF_AND_PDF_EXPORT',cases=24,method_cases=72,pdfs=pdfs,all_input_hashes_rechecked_remotely=ev['inputs_before_after_unchanged'],raw_distance_recomputed=True,optimizer_calls=0,environment_steps=0,model_forward_calls=0,selection_calls=0)
(R/'DISTANCE_RECOMPUTATION_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result,indent=2))
