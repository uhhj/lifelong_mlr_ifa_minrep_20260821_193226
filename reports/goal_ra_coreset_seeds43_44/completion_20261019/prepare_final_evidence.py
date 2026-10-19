"""Copy only the archived seed42 metadata package into a new final evidence set."""
from pathlib import Path
import json,hashlib,shutil,sys
R=Path(__file__).resolve().parent;D=R/sys.argv[1]
old=R.parent/'20261018_ra_coreset_goal_seed42';E=old/'final_evidence'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((E/'PACKAGE_MANIFEST.json').read_text());dest=D/'seed42/evidence'
for n,v in m['files'].items():
 p=Path(n);assert not p.is_absolute() and '..' not in p.parts
 assert p.suffix in {'.json','.jsonl','.log','.exit','.pid','.started'}
 assert not {'features','buffer','models','recovery','smoke','bindings'}&set(p.parts)
 assert sha(E/n)==v['sha256'] and (E/n).stat().st_size==v['bytes']
 out=dest/n;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(E/n,out)
shutil.copyfile(E/'PACKAGE_MANIFEST.json',dest/'PACKAGE_MANIFEST.json')
shutil.copytree(old/'baseline_evidence',D/'seed42/baseline_evidence',dirs_exist_ok=True)
provenance=dict(source=str(E),package_manifest_sha256=sha(E/'PACKAGE_MANIFEST.json'),original_local_acceptance_sha256=sha(old/'LOCAL_FINAL_ACCEPTANCE.json'),note='Archived completed seed42 evidence; no new model load or rollout. Current remote three-seed aggregate independently rechecks protected parent files.')
(D/'SEED42_PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('SEED42_ARCHIVED_PACKAGE_VERIFIED',len(m['files']))
