"""Review scoped RA completion report, recomputation and metadata; preserve prior launch."""
from pathlib import Path
import hashlib,json,re,subprocess
R=Path(__file__).resolve().parent;REPO=R.parents[1]/'publish_mlr_branch';D=REPO/'reports/goal_ra_coreset_seed42/completion'
assert subprocess.check_output(['git','-C',str(REPO),'branch','--show-current'],text=True).strip()=='MLR'
assert subprocess.check_output(['git','-C',str(REPO),'remote','get-url','origin'],text=True).strip()=='https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226.git'
allowed={'.py','.json','.jsonl','.md','.sh','.yaml','.txt','.csv','.patch','.exit','.log','.pid','.started','.sbatch'}
secret=re.compile(r'BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}')
def check(v):
    if isinstance(v,dict):
        assert not {'last_action','password','private_key','latent_tensor'}&set(v)
        for x in v.values():check(x)
    elif isinstance(v,list):
        for x in v:check(x)
manifest={}
for p in sorted(R.rglob('*')):
    if not p.is_file() or '__pycache__' in p.parts or p.name in {'code_bundle.zip','PUBLICATION_RECEIPT.json'}:continue
    rel=p.relative_to(R)
    roots={'final_evidence','baseline_evidence','monitor_20261019_1200'}
    names={'FINAL_REPORT.md','LOCAL_FINAL_ACCEPTANCE.json','RUN_STATUS.json','MONITOR_HANDOFF.md','final_matrix.csv','current_snapshot.json','completion_snapshot.json','COLLECTION_RECEIPT.json','LOCAL_RECOMPUTE_REPAIR.json','AUTOMATION_DELETION.json','collect_completion.py','check_status.py','recompute_final.py','prepare_comparison_evidence.py','write_final_report.py','publish_completion.py'}
    if rel.parts[0] not in roots and rel.as_posix() not in names:continue
    assert p.suffix in allowed or p.name=='LICENSE',str(p)
    assert not {'features','buffer','bindings','recovery','smoke'}&set(rel.parts),str(p)
    raw=p.read_bytes();assert len(raw)<5*1024**2
    txt=raw.decode('utf-8-sig');assert not secret.search(txt),str(p)
    assert not re.search(r'password\s*=\s*[\x22\x27][^\x22\x27]+[\x22\x27]',txt,re.I),str(p)
    if p.suffix=='.json':check(json.loads(txt))
    if p.suffix=='.jsonl':
        for line in txt.splitlines():
            if line.strip():check(json.loads(line))
    out=D/rel;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(raw)
    manifest[rel.as_posix()]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
(D/'.gitattributes').write_bytes(b'* -text -whitespace\n')
(D/'PUBLICATION_MANIFEST.json').write_bytes((json.dumps(manifest,indent=2)+'\n').encode())
print(json.dumps(dict(files=len(manifest),bytes=sum(v['bytes'] for v in manifest.values()),destination=str(D))))
