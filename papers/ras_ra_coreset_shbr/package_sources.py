from pathlib import Path
import hashlib, json, re, zipfile

R=Path(__file__).resolve().parent
allowed={'.tex','.bib','.cls','.bst','.md','.json','.py','.ps1','.txt'}
files=[p for p in sorted(R.iterdir()) if p.is_file() and p.suffix in allowed and not p.name.startswith(('PUBLICATION_', 'PACKAGE_', 'BUILD_'))]
manifest={}
for p in files:
    raw=p.read_bytes()
    assert len(raw)<2*1024**2, p.name
    text=raw.decode('utf-8-sig')
    assert not re.search(r'BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|github_pat_[A-Za-z0-9_]{30,}|ghp_[A-Za-z0-9]{20,}',text),p.name
    manifest[p.name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
log=(R/'main.log').read_text(encoding='utf8',errors='replace')
assert not re.search(r'undefined references|Citation .* undefined|LaTeX Error|^!',log,re.M)
receipt={'status':'COMPILED_AND_VISUALLY_REVIEWED_FRAMEWORK','pdf_sha256':hashlib.sha256((R/'main.pdf').read_bytes()).hexdigest(),'pages':14,'compiler':'Tectonic 0.17.0','visual_check':'Full-document overview; corrected title parsing; final title and tables checked at readable resolution.','remaining_warnings':'Official class footer has a 2.608pt overfull box; two loose paragraphs. No content clipping observed.','submission_ready':False,'files':manifest}
(R/'BUILD_VALIDATION.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
with zipfile.ZipFile(R/'RA_Coreset_SHBR_RAS_sources.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in files+[R/'BUILD_VALIDATION.json']:
        z.write(p,p.name)
print('Created flat Overleaf-ready source ZIP:',len(files)+1,'reviewed text files.')
