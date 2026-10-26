"""Publish only exact reviewed text sources, data and hashes via publish_mlr_branch."""
from pathlib import Path
import subprocess,json,hashlib,re,datetime
R=Path(__file__).resolve().parent;repo=R.parents[1]/'publish_mlr_branch';dest=repo/'reports/thesis_preliminary_20261026'
def git(*a):return subprocess.check_output(['git','-C',str(repo),*a],text=True,encoding='utf8').strip()
assert git('branch','--show-current')=='MLR'
assert git('remote','get-url','origin')=='https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226.git'
assert not git('diff','--cached','--name-only')
names=['README.md','build_slides.mjs','prepare_evidence.py','export_real_frames.py','package_materials.py','publish.py','EVIDENCE_AUDIT.json','REAL_FRAME_PROVENANCE.json','MATERIALS_MANIFEST.json','all_metrics.csv','EPOCHS_S41.json','training_scalars_S41.csv','train_S41_original_excerpt.jsonl','latex_document/main.tex','latex_document/table_baselines.tex','latex_document/table_S41.tex','latex_document/metric_definitions.tex','.build/render.mjs','.build/finalize.mjs']
manifest={}
for name in names:
 p=R/name;raw=p.read_bytes();assert len(raw)<2*1024**2;txt=raw.decode('utf-8-sig')
 assert p.suffix in ['.md','.mjs','.py','.json','.csv','.jsonl','.tex']
 assert not re.search(r'BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}',txt)
 assert not re.search(r'password\s*=\s*[\'\"][^\'\"]+[\'\"]',txt,re.I)
 out=dest/name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(raw)
 assert hashlib.sha256(out.read_bytes()).hexdigest()==hashlib.sha256(raw).hexdigest()
 manifest[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
(dest/'PUBLICATION_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
(dest/'.gitattributes').write_bytes(b'* -text -whitespace\n')
paths=[(dest/n).relative_to(repo).as_posix() for n in names+['PUBLICATION_MANIFEST.json','.gitattributes']]
subprocess.run(['git','-C',str(repo),'add','-f','--pathspec-from-file=-','--pathspec-file-nul'],input=b'\0'.join(p.encode() for p in paths)+b'\0',check=True)
assert set(git('diff','--cached','--name-only').splitlines())<=set(paths)
subprocess.run(['git','-C',str(repo),'diff','--cached','--check'],check=True)
if git('diff','--cached','--name-only'):
 subprocess.run(['git','-C',str(repo),'commit','-q','-m','Add verified thesis preliminary-work materials and scalar tables'],check=True)
 subprocess.run(['git','-C',str(repo),'push','origin','MLR'],check=True)
head=git('rev-parse','HEAD');remote=git('ls-remote','origin','refs/heads/MLR').split()[0];assert head==remote
receipt={'status':'PUSH_VERIFIED','commit':head,'remote':remote,'files':len(paths),'time':datetime.datetime.now().astimezone().isoformat(),'active_experiment':'S42/43 remains PARTIAL; no experiment state changed','private_images_published':False}
(R/'PUBLICATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8');print(json.dumps(receipt))
