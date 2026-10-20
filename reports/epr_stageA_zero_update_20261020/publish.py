"""Publish only reviewed scalar evidence/source; no model or dataset payloads."""
from pathlib import Path
import subprocess,hashlib,json,re,datetime
R=Path(__file__).resolve().parent;repo=R.parents[1]/"publish_mlr_branch"
dest=repo/"reports/epr_stageA_zero_update_20261020"
def git(*args):return subprocess.check_output(["git","-C",str(repo),*args],text=True).strip()
assert git("branch","--show-current")=="MLR"
assert git("remote","get-url","origin")=="https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226.git"
assert not git("diff","--cached","--name-only"),"Preserve unrelated staging"
assert json.loads((R/"LOCAL_ACCEPTANCE.json").read_text())["status"]=="PASS"
assert json.loads((R/"RUN_STATUS.json").read_text())["stage_B"]=="NOT_STARTED"
top=["PROTOCOL.md","REPORT.md","run_diagnostic.py","run_all.py","final_snapshot.py",
     "analyze.py","test_metrics.py","finalize_local.py","publish.py","RUN_STATUS.json","LOCAL_ACCEPTANCE.json"]
files=[R/n for n in top]+sorted(p for p in (R/"evidence").rglob("*") if p.is_file())
secret=re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}")
manifest={}
for p in files:
    rel=p.relative_to(R).as_posix();raw=p.read_bytes()
    assert p.suffix in (".py",".md",".json",".jsonl",".csv",".log",".exit")
    assert len(raw)<25*1024**2,(rel,len(raw))
    text=raw.decode("utf-8-sig")
    assert not secret.search(text),rel
    assert not re.search(r"(?:password|passwd)\s*=\s*['\"][^'\"]+['\"]",text,re.I),rel
    if p.suffix in (".json",".jsonl"):
        objects=[json.loads(l) for l in text.splitlines() if l.strip()] if p.suffix==".jsonl" else [json.loads(text)]
        def check(x):
            if isinstance(x,dict):
                assert not set(x)&{"last_action","actions","feature","features_tensor","state_dict","optimizer_state","password","private_key"},rel
                for v in x.values():check(v)
            elif isinstance(x,list):
                for v in x:check(v)
        for obj in objects:check(obj)
    dst=dest/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(raw)
    manifest[rel]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
(dest/".gitattributes").write_bytes(b"* -text -whitespace\n")
(dest/"PUBLICATION_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
paths=[(dest/n).relative_to(repo).as_posix() for n in list(manifest)+[".gitattributes","PUBLICATION_MANIFEST.json"]]
subprocess.run(["git","-C",str(repo),"add","-f","--pathspec-from-file=-","--pathspec-file-nul"],
               input=b"\0".join(p.encode("utf8") for p in paths)+b"\0",check=True)
assert set(git("diff","--cached","--name-only").splitlines())<=set(paths)
subprocess.run(["git","-C",str(repo),"diff","--cached","--check"],check=True)
if git("diff","--cached","--name-only"):
    subprocess.run(["git","-C",str(repo),"commit","-q","-m","Audit EPR zero-update opportunity on RA GOAL42 and OBJECT41"],check=True)
    subprocess.run(["git","-C",str(repo),"push","origin","MLR"],check=True)
head=git("rev-parse","HEAD");remote=git("ls-remote","origin","refs/heads/MLR").split()[0];assert head==remote
receipt=dict(status="PUSH_VERIFIED",commit=head,remote=remote,files=len(paths),
    bytes=sum(x["bytes"] for x in manifest.values()),stage="A_COMPLETED_B_NOT_STARTED",
    time=datetime.datetime.now().astimezone().isoformat())
(R/("PUBLICATION_RECEIPT_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".json")).write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))
