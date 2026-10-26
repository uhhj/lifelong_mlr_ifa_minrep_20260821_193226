from pathlib import Path
import json,hashlib,zipfile
R=Path(__file__).resolve().parent
files={}
for name in ['README.md','REAL_FRAME_PROVENANCE.json','EVIDENCE_AUDIT.json','all_metrics.csv','EPOCHS_S41.json','train_S41_original_excerpt.jsonl','training_scalars_S41.csv','training_curves_S41.png','training_curves_S41.pdf']:
 files[name]=R/name
for p in (R/'latex_document').rglob('*'):
 if p.is_file() and p.suffix in ['.tex','.png','.pdf']:files[p.relative_to(R).as_posix()]=p
for p in (R/'assets_private').glob('*.png'):files[p.relative_to(R).as_posix()]=p
manifest={}
for name,p in files.items():
 raw=p.read_bytes();assert len(raw)<5*1024**2
 if p.suffix=='.png':assert raw.startswith(b'\x89PNG\r\n\x1a\n')
 elif p.suffix=='.pdf':assert raw.startswith(b'%PDF')
 else:raw.decode('utf-8-sig')
 manifest[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
target=R/'output/论文前期工作_真实图像日志与LaTeX素材.zip'
with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for name,p in files.items():z.write(p,name)
 z.writestr('MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
with zipfile.ZipFile(target) as z:
 assert z.testzip() is None
 assert set(z.namelist())==set(manifest)|{'MANIFEST.json'}
 for name,m in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==m['sha256']
receipt={'status':'REVIEWED_PACKAGE','archive':target.name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'files':manifest,'publication':'Local user delivery only; no private RGB or archive pushed to Git.'}
(R/'MATERIALS_MANIFEST.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({'status':receipt['status'],'files':len(files),'bytes':target.stat().st_size}))
